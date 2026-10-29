// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_builder.hpp"

#include <bitset>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>

#include <tt_stl/assert.hpp>
#include "circular_buffer_constants.h"
#include "core_coord.hpp"
#include "hal_types.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "program_allocation_support.hpp"
#include "program_impl.hpp"

namespace tt::tt_metal::detail {

namespace {

std::bitset<MAX_PROCESSOR_TYPES_COUNT> get_kernel_processor_set(const Kernel& kernel) {
    std::bitset<MAX_PROCESSOR_TYPES_COUNT> set;
    for (int i = 0; i < kernel.expected_num_binaries(); i++) {
        int processor_id = kernel.get_kernel_processor_type(i);
        TT_ASSERT(0 <= processor_id && processor_id < MAX_PROCESSOR_TYPES_COUNT);
        set.set(processor_id);
    }
    return set;
}

}  // namespace

KernelHandle Metal1ProgramBuilder::add_kernel(
    const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& programmable_core_type) {
    TT_FATAL(program_.compiled_.empty(), "Cannot add kernel to an already compiled program {}", program_.id);

    // Metal 2.0 kernels (with named bindings, e.g. dfb::/tensor::/args::) are only legal on Metal 2.0 Programs
    if (kernel->is_metal2_kernel()) {
        TT_FATAL(
            program_.created_from_spec_,
            "Internal error: Metal 2.0 named bindings (dfb::/sem::/tensor::/args::) are only valid in a "
            "Metal 2.0 Program (created from a ProgramSpec).");
    }

    // Id is unique across all kernels on all core types
    KernelHandle id = program_.num_kernels();
    uint32_t index =
        MetalContext::instance(program_.context_id_).hal().get_programmable_core_type_index(programmable_core_type);

    auto new_kernel_core_type = kernel->get_kernel_programmable_core_type();
    auto new_kernel_processor_class = kernel->get_kernel_processor_class();

    std::set<CoreCoord> kernel_logical_cores = kernel->logical_cores();
    auto new_kernel_processor_set = get_kernel_processor_set(*kernel);
    for (size_t i = 0; i < program_.num_kernels(); i++) {
        // Note, looks like id is program specific, and increments naturally as kernels are added.
        //  add_kernel -> id = num_kernels -> kernel is inserted -> next num_kernels() increments.
        std::shared_ptr<Kernel> check_kernel = program_.get_kernel(i);
        auto check_kernel_core_type = check_kernel->get_kernel_programmable_core_type();
        auto check_kernel_processor_class = check_kernel->get_kernel_processor_class();
        if (check_kernel_core_type == new_kernel_core_type &&
            check_kernel_processor_class == new_kernel_processor_class &&
            (new_kernel_processor_set & get_kernel_processor_set(*check_kernel)).any()) {
            // Two kernels are using the same processor, need to check core ranges.
            std::set<CoreCoord> check_kernel_logical_cores = check_kernel->logical_cores();
            for (CoreCoord coreCoord : kernel_logical_cores) {
                TT_FATAL(
                    !check_kernel_logical_cores.contains(coreCoord),
                    "Core Overlap Between (\"{}\") and new kernel (\"{}\") at {}",
                    check_kernel->name(),
                    kernel->name(),
                    coreCoord.str());
            }
        }
    }

    if (!kernel->scratchpad_binding_handles().empty()) {
        register_ephemeral_allocation(program_.dfb_allocation_metadata_, kernel->core_range_set());
    }

    program_.kernels_[index].insert({id, kernel});
    program_.kernel_groups_[index].resize(0);
    program_.core_to_kernel_group_index_table_[index].clear();
    return id;
}

CBHandle Metal1ProgramBuilder::add_circular_buffer_(const std::shared_ptr<CircularBufferImpl>& circular_buffer) {
    // Metal 2.0 programs use DataflowBuffers, never legacy circular buffers.
    TT_FATAL(
        !program_.created_from_spec_,
        "Cannot add a legacy circular buffer to a Metal 2.0 Program; "
        "Metal 2.0 Programs use DataflowBuffers, and cannot be modified after construction.");

    // The deserialization constructor sets these sets independently, so local and remote are not
    // guaranteed to be subsets of buffer_indices; all three index the per-core bitsets below.
    const CircularBufferConfig& config = circular_buffer->config();
    for (const auto* indices :
         {&config.buffer_indices(), &config.local_buffer_indices(), &config.remote_buffer_indices()}) {
        for (uint32_t buffer_index : *indices) {
            if (buffer_index >= program_.max_dfbs_) {
                TT_THROW(
                    "Buffer index ({}) exceeds max number of circular buffers per core ({})",
                    buffer_index,
                    program_.max_dfbs_);
            }
        }
    }

    // Globally allocated circular buffer do not invalidate allocation because their addresses are tracked by memory
    // allocator
    if (not circular_buffer->globally_allocated()) {
        program_.invalidate_circular_buffer_allocation();
    } else {
        circular_buffer->assign_global_address();
        // invalidate_circular_buffer_allocation() would have done this; adding a buffer still means
        // the program has to be laid out again.
        program_.compile_and_allocate_needed_ = true;
    }

    // Mark which buffer indices are being used on each core the circular buffer is used on
    for (const CoreRange& core_range : circular_buffer->core_ranges().ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                std::bitset<NUM_CIRCULAR_BUFFERS>& cb_indices = program_.per_core_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& local_cb_indices = program_.per_core_local_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& remote_cb_indices =
                    program_.per_core_remote_cb_indices_[logical_core];
                auto add_buffer_indices = [&cb_indices](
                                              const std::unordered_set<uint8_t>& buffer_indices,
                                              std::bitset<NUM_CIRCULAR_BUFFERS>& target_cb_indices) {
                    for (uint32_t buffer_index : buffer_indices) {
                        if (cb_indices[buffer_index]) {
                            TT_THROW(
                                "Invalid circular buffer index: Cannot add circular buffer at index {}, another "
                                "circular "
                                "buffer already exists",
                                buffer_index);
                        }
                        cb_indices[buffer_index] = true;
                        target_cb_indices[buffer_index] = true;
                    }
                };
                add_buffer_indices(circular_buffer->config().local_buffer_indices(), local_cb_indices);
                add_buffer_indices(circular_buffer->config().remote_buffer_indices(), remote_cb_indices);
            }
        }
    }
    // A globally allocated CB does need l1 allocation
    if (!circular_buffer->globally_allocated()) {
        register_ephemeral_allocation(program_.cb_allocation_metadata_, circular_buffer->core_ranges());
    }

    program_.circular_buffers_.push_back(circular_buffer);
    program_.circular_buffer_by_id_.insert({circular_buffer->id(), circular_buffer});
    return circular_buffer->id();
}

CBHandle Metal1ProgramBuilder::add_circular_buffer(
    const CoreRangeSet& core_range_set, const CircularBufferConfig& config) {
    TT_FATAL(program_.compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", program_.id);
    TT_FATAL(
        program_.dataflow_buffers_.empty(),
        "Cannot add circular buffer to a program that already has dataflow buffers");
    // Merge ranges to reduce the number of multicasts needed to initialize CBs.
    std::shared_ptr<CircularBufferImpl> circular_buffer =
        std::make_shared<CircularBufferImpl>(core_range_set.merge_ranges(), config);
    return add_circular_buffer_(circular_buffer);
}

CBHandle Metal1ProgramBuilder::add_circular_buffer(
    const CoreRangeSet& core_range_set,
    const CircularBufferConfig& config,
    const experimental::GlobalCircularBuffer& global_circular_buffer) {
    TT_FATAL(program_.compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", program_.id);
    TT_FATAL(
        program_.dataflow_buffers_.empty(),
        "Cannot add circular buffer to a program that already has dataflow buffers");
    TT_FATAL(
        program_.per_core_cross_node_dfbs_.empty(),
        "Cannot add a GlobalCircularBuffer to a program that already has CrossNodeDFB participants. "
        "GlobalCircularBuffer and CrossNodeDFB are mutually exclusive within a program.");
    TT_FATAL(
        program_.per_core_prefetcher_pipes_.empty(),
        "Cannot add a GlobalCircularBuffer to a program that already has PrefetcherPipe attachments. "
        "GlobalCircularBuffer and PrefetcherPipe are mutually exclusive within a program.");
    // Merge ranges to reduce the number of multicasts needed to initialize CBs.
    std::shared_ptr<CircularBufferImpl> circular_buffer =
        std::make_shared<CircularBufferImpl>(core_range_set.merge_ranges(), config, global_circular_buffer);
    return add_circular_buffer_(circular_buffer);
}

void Metal1ProgramBuilder::validate_semaphore_id(
    const CoreRangeSet& crs, uint32_t semaphore_id, CoreType core_type) const {
    TT_FATAL(semaphore_id < NUM_SEMAPHORES, "Semaphore id {} exceeds max value {}", semaphore_id, NUM_SEMAPHORES - 1);

    for (const auto& core_range : crs.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                auto existing_semaphores = program_.semaphores_on_core(logical_core, core_type);
                for (const auto& semaphore : existing_semaphores) {
                    TT_FATAL(
                        semaphore.get().id() != semaphore_id,
                        "Semaphore id {} already in use on core {}",
                        semaphore_id,
                        logical_core.str());
                }
            }
        }
    }
}

void Metal1ProgramBuilder::add_semaphore(
    const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    TT_FATAL(program_.compiled_.empty(), "Cannot add semaphore to an already compiled program {}", program_.id);
    validate_semaphore_id(crs, semaphore_id, core_type);
    const uint32_t l1_alignment = MetalContext::instance(program_.context_id_).hal().get_alignment(HalMemType::L1);
    program_.semaphores_.emplace_back(Semaphore(crs, semaphore_id, init_value, l1_alignment, core_type));
}

uint32_t Metal1ProgramBuilder::create_semaphore(const CoreRangeSet& crs, uint32_t initial_value, CoreType core_type) {
    TT_FATAL(!crs.ranges().empty(), "Expecting a non-empty CoreRangeSet!");
    TT_FATAL(
        MetalContext::instance(program_.context_id_).is_coord_in_range(crs.ranges().back().end_coord, core_type),
        "Coordinates out of range");

    std::optional<uint32_t> semaphore_id;
    std::bitset<NUM_SEMAPHORES> used_semaphore_ids;
    for (const auto& core_range : crs.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                auto existing = program_.semaphores_on_core(logical_core, core_type);
                if (existing.size() == NUM_SEMAPHORES) {
                    TT_THROW(
                        "Cannot add semaphore on core {}. Max number of semaphores ({}) reached!",
                        logical_core.str(),
                        NUM_SEMAPHORES);
                }
                for (const auto& semaphore : existing) {
                    used_semaphore_ids.set(semaphore.get().id());
                }
            }
        }
    }
    for (uint32_t sem_id = 0; sem_id < NUM_SEMAPHORES; sem_id++) {
        if (!used_semaphore_ids.test(sem_id)) {
            semaphore_id = sem_id;
            break;
        }
    }
    TT_FATAL(
        semaphore_id.has_value(),
        "Unable to initialize semaphore on CoreRangeSet {}: all {} IDs are in use",
        crs.str(),
        NUM_SEMAPHORES);

    add_semaphore(crs, *semaphore_id, initial_value, core_type);
    return *semaphore_id;
}

}  // namespace tt::tt_metal::detail
