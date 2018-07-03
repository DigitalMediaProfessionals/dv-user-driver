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


/// @brief Forward reference for dv_context structure.
typedef struct dv_context_impl dv_context;

/// @brief Forward reference for dv_mem structure.
typedef struct dv_mem_impl dv_mem;


/// @brief Returns version string of the driver interface.
/// @details Starts with MAJOR.MINOR.SUB for example "0.1.0 Initial release."
const char* dv_get_version_string();


/// @brief Returns last error message.
const char* dv_get_last_error_message();


/// @brief Creates context for working with DV accelerator.
/// @param path Path to the device, use NULL or empty string to select default device.
/// @return Non-NULL on success, NULL on error.
dv_context* dv_context_create(const char *path);


/// @brief Destroys context for working with DV accelerator.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
void dv_context_destroy(dv_context *ctx);


/// @brief Allocates physically continuous chunk of memory.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @param size Memory size in bytes.
/// @return Handle for the allocated memory or NULL on error.
/// @details Memory is allocated using ION with CMA and is not yet mapped to user or kernel address space.
dv_mem* dv_mem_alloc(dv_context *ctx, size_t size);


/// @brief Frees previously allocated memory.
/// @param mem Handle for the allocated memory, when NULL the error is returned.
void dv_mem_free(dv_mem *mem);


/// @brief Maps previously allocated memory to the user address space.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @return Pointer to memory region in user address space or NULL on error.
/// @details Retuned memory can be read or written, executable flag is not set.
///          If the memory was already mapped, the same pointer will be returned.
uint8_t *dv_mem_map(dv_mem *mem);


/// @brief Unmaps previously allocated and mapped memory from the user address space.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @detail Function can be called repeatedly.
void dv_mem_unmap(dv_mem *mem);


/// @brief Starts Device <-> CPU synchronization of the memory buffer.
/// @param mem Handle to the allocated memory, when NULL the error is returned.
/// @param rd If non-zero, the Device -> CPU synchronization will occure.
/// @param wr If non-zero, the CPU -> Device synchronization will occure on dv_mem_sync_end().
/// @return 0 on success, non-zero otherwise.
int dv_mem_sync_start(dv_mem *mem, int rd, int wr);


/// @brief Finishes the last started Device <-> CPU synchronization.
/// @return 0 on success, non-zero otherwise.
int dv_mem_sync_end(dv_mem *mem);


/// @brief Flushes queued tasks for execution if any.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @return 0 on success, non-zero otherwise.
int dv_flush(dv_context *ctx);


/// @brief Waits until all scheduled tasks are complete.
/// @param ctx Context for working with DV accelerator, when NULL the error is returned.
/// @return 0 on success, non-zero otherwise.
int dv_finish(dv_context *ctx);


#ifdef __cplusplus
}
#endif
