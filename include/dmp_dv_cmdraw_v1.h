/*
 *  Copyright 2019 Digital Media Professionals Inc.

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
 * @brief User-space definition of dv_cmdraw structure version v0.
 */
#pragma once

#ifndef _DMP_DV_CMDRAW_V1_H_
#define _DMP_DV_CMDRAW_V1_H_

#include "dmp_dv.h"

/// @brief Raw command for convolutional block version 0.
struct dmp_dv_cmdraw_conv_v1 {
  struct dmp_dv_cmdraw header;  // General structure information

	struct dmp_dv_buf u8tofp16_table;

  struct dmp_dv_cmdraw_conv_v0 conv_cmd;
};

#endif  // _DMP_DV_CMDRAW_V1_H_
