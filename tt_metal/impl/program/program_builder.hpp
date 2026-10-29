// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

class CircularBufferImpl;
class Kernel;

namespace experimental {
class GlobalCircularBuffer;
}

namespace detail {

class ProgramImpl;

// Builds up a ProgramImpl for the Metal 1.0 host API (CreateKernel / CreateCircularBuffer / CreateSemaphore and the
// ProgramDescriptor constructor). Covers kernels, circular buffers and semaphores only; it is a non-owning view and
// the ProgramImpl keeps all of the state. Metal 2.0 construction reuses the kernel and semaphore primitives.
class Metal1ProgramBuilder {
public:
    explicit Metal1ProgramBuilder(ProgramImpl& program) : program_(program) {}

    KernelHandle add_kernel(const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& core_type);

    // Takes ownership of an already-built CB (used by the ProgramDescriptor path); returns its handle.
    CBHandle add_circular_buffer_(const std::shared_ptr<CircularBufferImpl>& circular_buffer);
    CBHandle add_circular_buffer(const CoreRangeSet& core_range_set, const CircularBufferConfig& config);
    CBHandle add_circular_buffer(
        const CoreRangeSet& core_range_set,
        const CircularBufferConfig& config,
        const experimental::GlobalCircularBuffer& global_circular_buffer);

    // Allocate the next free semaphore ID on the given cores and insert the semaphore.
    // Returns the allocated ID; TT_FATALs if no slot can be found.
    uint32_t create_semaphore(const CoreRangeSet& crs, uint32_t initial_value, CoreType core_type);

    // Insert a semaphore at the caller-specified ID. Used by:
    //   - Construction paths where the ID is already decided upstream (ProgramDescriptor ctor path).
    //   - create_semaphore above
    // The ProgramDescriptor path is something of a hack; it will eventually be deprecated in favor of Metal 2.0.
    void add_semaphore(const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type);

    // Validates that a semaphore ID is within bounds and not already in use on overlapping cores
    void validate_semaphore_id(const CoreRangeSet& crs, uint32_t semaphore_id, CoreType core_type) const;

private:
    ProgramImpl& program_;
};

}  // namespace detail
}  // namespace tt::tt_metal
