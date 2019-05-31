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
#include "../../dv-kernel-driver/uapi/dmp_dv_cmdraw_v1.h"


#ifdef __cplusplus
extern "C" {
#endif


/// @brief Last error message (forward declaration).
extern char s_last_error_message[256];


/// @brief Helper to set the last error message.
#define SET_ERR(...) snprintf(s_last_error_message, sizeof(s_last_error_message), __VA_ARGS__)


/// @brief Helper to set the last error message for ioctl call.
#define SET_IOCTL_ERR(retval, dev, cmd) SET_ERR("ioctl(%s) returned %d for %s with errno=%d: %s", cmd, retval, dev, errno, strerror(errno))


/// @brief Helper to set the last error message on implementation logic error.
#define SET_LOGIC_ERR() SET_ERR("%s(): Control should not reach line %d of file %s", __func__, __LINE__, __FILE__)


/// @brief Path to convolutional character device file.
#define DMP_DV_DEV_PATH_CONV  "/dev/dv_conv"

/// @brief Path to fully connected character device file.
#define DMP_DV_DEV_PATH_FC  "/dev/dv_fc"

/// @brief Path to IPU character device file.
#define DMP_DV_DEV_PATH_IPU  "/dev/dv_ipu"

#ifdef __cplusplus
}  // extern "C"
#endif
