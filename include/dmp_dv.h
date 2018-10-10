/*
 *  Copyright 2018 Digital Media Professionals Inc.

 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at

 *      http://www.apache.org/licenses/LICENSE-2.0

 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
/*
 * @brief User-space driver interface.
 */
#pragma once

#include <stdlib.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


#ifdef __GNUC__
#pragma GCC visibility push(default)
#endif


/// @brief Device execution context.
typedef struct dmp_dv_context_impl dmp_dv_context;

/// @brief Device-accessible memory allocation.
typedef struct dmp_dv_mem_impl dmp_dv_mem;

/// @brief Command list for execution.
/// @details Contains prepacked in device specific format commands for execution,
///          thus reducing argument packing overhead.
typedef struct dmp_dv_cmdlist_impl dmp_dv_cmdlist;


/// @brief Returns version string of the driver interface.
/// @details Starts with MAJOR.MINOR.SUB for example "0.1.0 Initial release."
///          It is thread-safe.
const char* dmp_dv_get_version_string();


/// @brief Returns last error message.
/// @details It might return garbage if several functions will fail from multiple threads
///          simultaneously during this function call.
const char* dmp_dv_get_last_error_message();


/// @brief Creates context for working with DV accelerator.
/// @return Non-NULL on success, NULL on error.
/// @details It is thread-safe.
dmp_dv_context* dmp_dv_context_create();


/// @brief Returns information about context as human-readable string.
/// @details It is thread-safe.
const char *dmp_dv_context_get_info_string(dmp_dv_context* ctx);


/// @brief Structure with information about the context.
typedef struct dmp_dv_info_impl {
  uint32_t size;            // size of this structure
  uint32_t version;         // version of this structure
} dmp_dv_info;


/// @brief Structure with information about the context (version 0).
typedef struct dmp_dv_info_v0_impl {
  uint32_t size;               // size of this structure
  uint32_t version;            // version of this structure (set to 0)
  int32_t ub_size;             // unified buffer size
  int32_t max_kernel_size;     // maximum supported convolutional kernel size
  int32_t conv_freq;           // convolutional block frequency in MHz
  int32_t fc_freq;             // fully connected block frequency in MHz
  int32_t max_fc_vector_size;  // fully connected block maximum input vector size in elements
  int32_t rsvd;                // padding to 64-bits
} dmp_dv_info_v0;


/// @brief Fills structure with information about the context.
/// @param ctx Context for working with DV accelerator, when NULL it is ignored.
/// @param info Structure to be filled, fields size and version must be set.
/// @return 0 on success, non-zero otherwise.
/// @details On return, the version field will be set to maximum supported version less or equal to the requested,
///          the fields of the corresponding structure will be set only if the size is enough.
///          It is thread-safe.
int dmp_dv_context_get_info(dmp_dv_context* ctx, dmp_dv_info *info);


/// @brief Releases context for working with DV accelerator (decreases reference counter).
/// @param ctx Context for working with DV accelerator, when NULL it is ignored.
/// @details Call this when "ctx" is no longer needed.
///          It is thread-safe.
void dmp_dv_context_release(dmp_dv_context *ctx);


/// @brief Retains context for working with DV accelerator (increases reference counter).
/// @param ctx Context for working with DV accelerator, when NULL it is ignored.
/// @details It is thread-safe.
void dmp_dv_context_retain(dmp_dv_context *ctx);


/// @brief Allocates physically continuous chunk of memory.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @param size Memory size in bytes.
/// @return Handle for the allocated memory or NULL on error.
/// @details Memory is allocated using ION with CMA and is not yet mapped to user or kernel address space.
///          It is thread-safe.
dmp_dv_mem* dmp_dv_mem_alloc(dmp_dv_context *ctx, size_t size);


/// @brief Releases allocated memory (decreses reference counter).
/// @param mem Handle for the allocated memory, when NULL it is ignored.
/// @details Call this when "mem" is no longer needed.
///          dmp_dv_mem_unmap() will be called automatically before the memory is returned to the system.
///          It is thread-safe.
void dmp_dv_mem_release(dmp_dv_mem *mem);


/// @brief Retains allocated memory (increases reference counter).
/// @param mem Handle for the allocated memory, when NULL it is ignored.
/// @details It is thread-safe.
void dmp_dv_mem_retain(dmp_dv_mem *mem);


/// @brief Maps previously allocated memory to the user address space.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @return Pointer to memory region in user address space or NULL on error.
/// @details Retuned memory can be read or written, executable flag is not set.
///          If the memory was already mapped, the same pointer will be returned.
///          It is thread-safe only on different memory handles.
uint8_t *dmp_dv_mem_map(dmp_dv_mem *mem);


/// @brief Unmaps previously allocated and mapped memory from the user address space.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @details Function can be called repeatedly.
///          dmp_dv_mem_sync_end() will be called automatically before unmapping.
///          It is thread-safe only on different memory handles.
void dmp_dv_mem_unmap(dmp_dv_mem *mem);


/// @brief Starts Device <-> CPU synchronization of the memory buffer.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @param rd If non-zero, the Device -> CPU synchronization will occur before this function returns.
/// @param wr If non-zero, the CPU -> Device synchronization will occur on dmp_dv_mem_sync_end().
/// @return 0 on success, non-zero otherwise.
/// @details When called multiple times with the same or less flags rd | wr, the function does nothing.
///          It is thread-safe only on different memory handles.
int dmp_dv_mem_sync_start(dmp_dv_mem *mem, int rd, int wr);


/// @brief Finishes the last started Device <-> CPU synchronization.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @return 0 on success, non-zero otherwise.
/// @details When calling second time before next call to dmp_dv_mem_sync_start(), the function does nothing.
///          It is thread-safe only on different memory handles.
int dmp_dv_mem_sync_end(dmp_dv_mem *mem);


/// @brief Returns allocated size in bytes for the provided memory handle.
/// @param mem Handle to the allocated memory, when NULL the function will return 0.
/// @return Size in bytes (can be greater than requested in dmp_dv_mem_alloc()) or 0 if mem is NULL.
/// @details It is thread-safe.
size_t dmp_dv_mem_get_size(dmp_dv_mem *mem);


/// @brief Returns total per-process allocated size in bytes.
/// @details It is thread-safe.
int64_t dmp_dv_mem_get_total_size();


/// @brief Creates command list.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @return Handle to command list or NULL on error.
/// @details It is thread-safe.
dmp_dv_cmdlist *dmp_dv_cmdlist_create(dmp_dv_context *ctx);


/// @brief Releases the command list (decreases reference counter).
/// @param cmdlist Handle to command list, when NULL it is ignored.
/// @details Call this when "cmdlist" is no longer needed.
///          It is thread-safe.
void dmp_dv_cmdlist_release(dmp_dv_cmdlist *cmdlist);


/// @brief Retains the command list (increases reference counter).
/// @param cmdlist Handle to command list, when NULL it is ignored.
/// @details It is thread-safe.
void dmp_dv_cmdlist_retain(dmp_dv_cmdlist *cmdlist);


/// @brief Commits the command list, preparing device-specific structures for further execution.
/// @param cmdlist Handle to command list, when NULL the error is returned.
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe only on different command lists.
int dmp_dv_cmdlist_commit(dmp_dv_cmdlist *cmdlist);


/// @brief Schedules command list for execution.
/// @param cmdlist Handle to command list, when NULL the error is returned.
/// @return exec_id >= 0 for this execution on success, < 0 on error.
/// @details Each context is associated with a single execution queue.
///          It is thread-safe.
int64_t dmp_dv_cmdlist_exec(dmp_dv_cmdlist *cmdlist);


/// @brief Waits for the specific scheduled command to be completed.
/// @param cmdlist Handle to command list, when NULL the error is returned.
/// @param exec_id Id of the scheduled command to wait for completion.
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe.
int dmp_dv_cmdlist_wait(dmp_dv_cmdlist *cmdlist, int64_t exec_id);


/// @brief Memory buffer specification.
typedef struct dmp_dmp_dv_buf_impl {
  union {
    dmp_dv_mem *mem;  // memory handle
    uint64_t rsvd;    // padding to 64-bit size
  };
  uint64_t offs;      // offset from the start of the buffer, must be 16-bit aligned
} dmp_dv_buf;

/// @brief Convolutional device type id.
#define DMP_DV_DEV_CONV 1

/// @brief Fully connected device type id.
#define DMP_DV_DEV_FC 2

/// @brief Upper bound of different device type ids.
#define DMP_DV_DEV_COUNT 3

/// @brief Raw command for execution.
typedef struct dmp_dmp_dv_cmdraw_impl {
  uint32_t size;        // size of this structure
  uint8_t device_type;  // device type
  uint8_t version;      // version of this structure
  uint8_t rsvd[2];      // padding to 64-bit size
} dmp_dv_cmdraw;


/// @brief Adds raw command to the command list.
/// @param cmdlist Handle to command list, when NULL the error is returned.
/// @param cmd Raw command for execution, see dmp_dv_cmdraw_v0.h for the description of command version 0.
/// @return 0 on success, non-zero otherwise, known error codes:
///         EINVAL - invalid argument such as structure size,
///         ENOTSUP - raw command version is not supported.
/// @details It is thread-safe only on different command lists.
int dmp_dv_cmdlist_add_raw(dmp_dv_cmdlist *cmdlist, dmp_dv_cmdraw *cmd);


/// @brief Packs convolution layer weights and biases into output array.
/// @param n_channels Number of input channels, for depthwise convolution this must be set to 1.
/// @param kx Kernel width.
/// @param ky Kernel height.
/// @param n_kernels Number of output channels.
/// @param quant_map Quantization table for weights (but not bias), 256 elements, can be NULL.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format, else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size n_kernels.
/// @param packed_weights Output buffer for packed weights information (can be NULL if packed_weights_size is 0).
/// @param packed_weights_size On input, contains the size of the packed_weights buffer in bytes (can be 0, in such case it will be filled with the required buffer size), on output will contain the required buffer size.
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe.
int dmp_dv_pack_conv_weights(
    int n_channels, int kx, int ky, int n_kernels,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias,
    uint8_t *packed_weights, size_t *packed_weights_size);


/// @brief Packs dilated convolution layer weights and biases into output array.
/// @param n_channels Number of input channels.
/// @param kx Kernel width.
/// @param ky Kernel height.
/// @param n_kernels Number of output channels.
/// @param quant_map Quantization table for weights (but not bias), can be NULL.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format, else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size n_kernels.
/// @param packed_weights Output buffer for packed weights information (can be NULL if packed_weights_size is 0).
/// @param packed_weights_size On input, contains the size of the packed_weights buffer in bytes (can be 0, in such case it will be filled with the required buffer size), on output will contain the required buffer size.
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe.
int dmp_dv_pack_dil_weights(
    int n_channels, int kx, int ky, int n_kernels,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias,
    uint8_t *packed_weights, size_t *packed_weights_size);


/// @brief Packs fully connected layer weights and biases into output array possibly rearranging them to match input and output shapes.
/// @param c_input Number of input channels.
/// @param h_input Input height (set to 1 for 1D input).
/// @param w_input Input width (set to 1 for 1D input).
/// @param c_output Number of output channels.
/// @param h_output Output height (set to 1 for 1D output).
/// @param w_output Output width (set to 1 for 1D output).
/// @param quant_map Quantization table for weights (but not bias), 256 elements, can be NULL.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format (N=output_size), else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size output_size.
/// @param packed_weights Output buffer for packed weights information (can be NULL if packed_weights_size is 0).
/// @param packed_weights_size On input, contains the size of the packed_weights buffer in bytes (can be 0, in such case it will be filled with the required buffer size), on output will contain the required buffer size.
/// @return 0 on success, non-zero otherwise.
/// @details The function packs weights in NCHW format to the DV input format WHC8 (n_channels / 8, width, height, 8 channels)
///          with rearranging to produce output in DV format WHC8.
///          It is thread-safe.
int dmp_dv_pack_fc_weights(
    int c_input, int h_input, int w_input,
    int c_output, int h_output, int w_output,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias,
    uint8_t *packed_weights, size_t *packed_weights_size);


#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#ifdef __cplusplus
}  // extern "C"
#endif
