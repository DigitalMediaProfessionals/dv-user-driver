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
/// @details Contains prepacked in device specific format commands for execution, thus reducing argument packing overhead.
typedef struct dmp_dv_cmdlist_impl dmp_dv_cmdlist;


/// @brief Returns version string of the driver interface.
/// @details Starts with MAJOR.MINOR.SUB for example "0.1.0 Initial release."
const char* dmp_dv_get_version_string();


/// @brief Returns last error message.
const char* dmp_dv_get_last_error_message();


/// @brief Creates context for working with DV accelerator.
/// @param path Path to the device, use NULL or empty string to select default device.
/// @return Non-NULL on success, NULL on error.
dmp_dv_context* dmp_dv_context_create(const char *path);


/// @brief Returns some information about context.
const char *dmp_dv_context_get_info_string(dmp_dv_context* ctx);


/// @brief Destroys context for working with DV accelerator.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
void dmp_dv_context_destroy(dmp_dv_context *ctx);


/// @brief Allocates physically continuous chunk of memory.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @param size Memory size in bytes.
/// @return Handle for the allocated memory or NULL on error.
/// @details Memory is allocated using ION with CMA and is not yet mapped to user or kernel address space.
dmp_dv_mem* dmp_dv_mem_alloc(dmp_dv_context *ctx, size_t size);


/// @brief Frees previously allocated memory.
/// @param mem Handle for the allocated memory, when NULL the error is returned.
void dmp_dv_mem_free(dmp_dv_mem *mem);


/// @brief Maps previously allocated memory to the user address space.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @return Pointer to memory region in user address space or NULL on error.
/// @details Retuned memory can be read or written, executable flag is not set.
///          If the memory was already mapped, the same pointer will be returned.
uint8_t *dmp_dv_mem_map(dmp_dv_mem *mem);


/// @brief Unmaps previously allocated and mapped memory from the user address space.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @detail Function can be called repeatedly.
void dmp_dv_mem_unmap(dmp_dv_mem *mem);


/// @brief Starts Device <-> CPU synchronization of the memory buffer.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @param rd If non-zero, the Device -> CPU synchronization will occure.
/// @param wr If non-zero, the CPU -> Device synchronization will occure on dmp_dv_mem_sync_end().
/// @return 0 on success, non-zero otherwise.
int dmp_dv_mem_sync_start(dmp_dv_mem *mem, int rd, int wr);


/// @brief Finishes the last started Device <-> CPU synchronization.
/// @return 0 on success, non-zero otherwise.
int dmp_dv_mem_sync_end(dmp_dv_mem *mem);


/// @brief Returns allocated size in bytes for provided memory handle.
size_t dmp_dv_mem_get_size(dmp_dv_mem *mem);


/// @brief Waits for all scheduled commands to be executed.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @return 0 on success, non-zero otherwise.
int dmp_dv_sync(dmp_dv_context *ctx);


/// @brief Creates command list.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @return Handle to command list or NULL on error.
dmp_dv_cmdlist *dmp_dv_cmdlist_create(dmp_dv_context *ctx);


/// @brief Destroys command list.
/// @param cmdlist Handle to command list, when NULL it is ignored.
void dmp_dv_cmdlist_destroy(dmp_dv_cmdlist *cmdlist);


/// @brief Ends the command list, preparing device-specific structures for further execution.
/// @param cmdlist Handle to command list, when NULL the error is returned.
/// @return 0 on success, non-zero otherwise.
int dmp_dv_cmdlist_end(dmp_dv_cmdlist *cmdlist);


/// @brief Schedules command list for execution.
/// @param cmdlist Handle to command list, when NULL the error is returned.
/// @return 0 on success, non-zero otherwise.
int dmp_dv_cmdlist_exec(dmp_dv_cmdlist *cmdlist);



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


/// @brief Adds raw command to the command list.
int dmp_dv_cmdlist_add_raw(dmp_dv_cmdlist *cmdlist, dmp_dv_cmdraw *cmd);


/// @brief Returns maximum supported version of dmp_dv_cmdraw structure.
int32_t dmp_dv_get_cmdraw_max_version();


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
