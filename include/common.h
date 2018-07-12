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

#include <string>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include <linux/dma-buf.h>
#include "ion.h"

#include "dmp_dv.h"


/// @brief Last error message (forward declaration).
extern char s_last_error_message[256];


/// @brief Helper to set the last error message.
#define SET_ERR(...) snprintf(s_last_error_message, sizeof(s_last_error_message), __VA_ARGS__)
