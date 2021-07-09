# DV User Driver

This lib implements the user space API for DV700 cores.

## HOWTO Build
Run `make`.

## HOWTO Install
Run `sudo make install` on your AI FPGA Module.
Or copy `libdmpdv.so` under`/usr/lib/` on the Module.

## Basic usage flow

1. Create context with `dmp_dv_context_create()`
2. Prepare input/output:
    * Allocate input/output buffers with `dmp_dv_mem_alloc()`
    * Obtain pointers to input/output with `dmp_dv_mem_map()`
3. Prepare weights:
    * Obtain the size of packed weights with `dmp_dv_pack_conv_weights()` (can be done offline)
    * Allocate weights buffers with `dmp_dv_mem_alloc()`
    * Obtain pointer to packjed weights with `dmp_dv_mem_map()`
    * Fill packed weights either from offline prepared data or by calling `dmp_dv_pack_conv_weights()`
4. Prepare command lists:
    * Create command list with `dmp_dv_cmdlist_create()`
    * Add commands to command list with `dmp_dv_cmdlist_add_raw()`
    * Commit command list with `dmp_dv_cmdlist_commit()`
5. Do the inference:
    * Fill input buffer and call `dmp_dv_mem_to_device()`
    * Execute command list with `dmp_dv_cmdlist_exec()`
    * Wait for execution completion with `dmp_dv_cmdlist_wait()`
    * Synchronize output data with CPU by calling `dmp_dv_mem_to_cpu()`
6. Release resources with `dmp_dv_cmdlist_release()`, `dmp_dv_mem_release()`, `dmp_dv_context_release()`
