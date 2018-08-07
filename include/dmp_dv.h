/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
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
/// @details Context is bound to the specific device and has single execution queue.
///          Multiple contexts can coexist, commands are executed in exclusive mode:
///          execution from several contexts is possible but occure sequentially.
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
/// @param path Path to the device, use NULL or empty string to select default device.
/// @return Non-NULL on success, NULL on error.
/// @details It is thread-safe.
dmp_dv_context* dmp_dv_context_create(const char *path);


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
  uint32_t size;            // size of this structure
  uint32_t version;         // version of this structure (set to 0)
  int32_t ub_size;          // unified buffer size
  int32_t max_kernel_size;  // maximum supported convolutional kernel size
  int32_t conv_freq;        // convolutional block frequency in MHz
  int32_t fc_freq;          // fully connected block frequency in MHz
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
///          It is thread-safe only on different memory handles.
void dmp_dv_mem_unmap(dmp_dv_mem *mem);


/// @brief Starts Device <-> CPU synchronization of the memory buffer.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @param rd If non-zero, the Device -> CPU synchronization will occure.
/// @param wr If non-zero, the CPU -> Device synchronization will occure on dmp_dv_mem_sync_end().
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe only on different memory handles.
int dmp_dv_mem_sync_start(dmp_dv_mem *mem, int rd, int wr);


/// @brief Finishes the last started Device <-> CPU synchronization.
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe only on different memory handles.
int dmp_dv_mem_sync_end(dmp_dv_mem *mem);


/// @brief Returns allocated size in bytes for provided memory handle.
/// @details It is thread-safe.
size_t dmp_dv_mem_get_size(dmp_dv_mem *mem);


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
  uint64_t offs;      // offset from the start of the buffer
} dmp_dv_buf;


/// @brief Raw command for execution.
typedef struct dmp_dmp_dv_cmdraw_impl {
  uint32_t size;     // size of this structure
  uint32_t version;  // version of this structure
} dmp_dv_cmdraw;


/// @brief Adds raw command for convolutional block to the command list.
/// @return 0 on success, non-zero otherwise, known error codes:
///         EINVAL - invalid argument such as structure size,
///         ENOTSUP - raw command version is not supported.
/// @details It is thread-safe only on different command lists.
int dmp_dv_cmdlist_add_raw_conv(dmp_dv_cmdlist *cmdlist, dmp_dv_cmdraw *cmd);


/// @brief Adds raw command for fully connected block to the command list.
/// @return 0 on success, non-zero otherwise, known error codes:
///         EINVAL - invalid argument such as structure size,
///         ENOTSUP - raw command version is not supported.
/// @details It is thread-safe only on different command lists.
int dmp_dv_cmdlist_add_raw_fc(dmp_dv_cmdlist *cmdlist, dmp_dv_cmdraw *cmd);


/// @brief Packs convolution layer weights and biases into output array.
/// @param n_channels Number of input channels.
/// @param kx Kernel width.
/// @param ky Kernel height.
/// @param n_kernels Number of output channels.
/// @param quant_map Quantization table for weights (but not bias), can be NULL.
/// @param weights If quant_map is NULL, array of half precision floating point weights in NCHW format, else array of 1-byte indices.
/// @param bias Array of half precision floating point biases of size n_kernels.
/// @param output Output buffer for packed weights information (can be NULL if output_size is 0).
/// @param output_size On input, contains the size of the output buffer in bytes (can be 0, in such case it will be filled with the required output size), on output will contain the required output size.
/// @return 0 on success, non-zero otherwise.
/// @details It is thread-safe.
int dmp_dv_pack_conv_weights(
    int n_channels, int kx, int ky, int n_kernels,
    const uint16_t quant_map[256],
    const void *weights, const uint16_t *bias,
    uint8_t *output, size_t *output_size);


#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#ifdef __cplusplus
}  // extern "C"
#endif
