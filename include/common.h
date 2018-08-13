/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief Common staff.
 */
#pragma once

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>

#include <linux/dma-buf.h>
#include "ion.h"

#include "dmp_dv.h"
#include "../../dv-kernel-driver/uapi/dmp-dv.h"
#include "../../dv-kernel-driver/uapi/dimensions.h"


#ifdef __cplusplus
extern "C" {
#endif


/// @brief Last error message (forward declaration).
extern char s_last_error_message[256];


/// @brief Helper to set the last error message.
#define SET_ERR(...) snprintf(s_last_error_message, sizeof(s_last_error_message), __VA_ARGS__)


/// @brief Helper to set the last error message for ioctl call.
#define SET_IOCTL_ERR(dev, cmd) SET_ERR("ioctl(%s) failed for %s", cmd, dev)


/// @brief Helper to set the last error message on implementation logic error.
#define SET_LOGIC_ERR() SET_ERR("Control should not reach line %d of file %s", __LINE__, __FILE__)


#ifdef __cplusplus
}  // extern "C"
#endif
