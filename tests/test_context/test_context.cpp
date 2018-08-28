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
 * @brief Test for correctness of structures sizes and context creation/deletion.
 */
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <dirent.h>

#include <stdio.h>
#include <string.h>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"
#include "../../dv-kernel-driver/uapi/dmp_dv_cmdraw_v0.h"


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)
#define CHECK_SIZEOF(name, got, expected) if (got != expected) { ERR("sizeof(%s) is %zu while expecting %d\n", name, got, expected); return -1; }

int test_context() {
  LOG("ENTER: test_context\n");
  LOG("dmp_dv_get_version_string(): %s\n", dmp_dv_get_version_string());

  if ((sizeof(dmp_dv_info_v0) & 7) || (sizeof(dmp_dv_buf) & 7) || (sizeof(dmp_dv_cmdraw) & 7) ||
      (sizeof(dmp_dv_cmdraw_conv_v0_run) & 7) || (sizeof(dmp_dv_cmdraw_conv_v0) & 7) ||
      (sizeof(dmp_dv_cmdraw_fc_v0) & 7)) {
    ERR("Detected structure with size not multiple of 8\n");
    return -1;
  }

  if ((sizeof(dmp_dv_kbuf) & 7) ||
      (sizeof(dmp_dv_kcmdraw_conv_v0_run) & 7) || (sizeof(dmp_dv_kcmdraw_conv_v0) & 7) ||
      (sizeof(dmp_dv_kcmdraw_fc_v0) & 7)) {
    ERR("Detected structure for communication with kernel module with size not multiple of 8\n");
    return -1;
  }

  CHECK_SIZEOF("__fp16", sizeof(__fp16),  2);

  CHECK_SIZEOF("dmp_dv_info_v0", sizeof(dmp_dv_info_v0), 32);
  CHECK_SIZEOF("dmp_dv_buf", sizeof(dmp_dv_buf), 16);
  CHECK_SIZEOF("dmp_dv_cmdraw", sizeof(dmp_dv_cmdraw), 8);
  CHECK_SIZEOF("dmp_dv_cmdraw_conv_v0_run", sizeof(dmp_dv_cmdraw_conv_v0_run), 56);
  CHECK_SIZEOF("dmp_dv_cmdraw_conv_v0", sizeof(dmp_dv_cmdraw_conv_v0), 1864);
  CHECK_SIZEOF("dmp_dv_cmdraw_fc_v0", sizeof(dmp_dv_cmdraw_fc_v0), 72);

  CHECK_SIZEOF("dmp_dv_kbuf", sizeof(dmp_dv_kbuf), 16);
  CHECK_SIZEOF("dmp_dv_kcmdraw_conv_v0_run", sizeof(dmp_dv_kcmdraw_conv_v0_run), 56);
  CHECK_SIZEOF("dmp_dv_kcmdraw_conv_v0", sizeof(dmp_dv_kcmdraw_conv_v0), 1864);
  CHECK_SIZEOF("dmp_dv_kcmdraw_fc_v0", sizeof(dmp_dv_kcmdraw_fc_v0), 72);

  dmp_dv_context *ctx = dmp_dv_context_create();
  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    return -1;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  dmp_dv_info_v0 info;
  info.size = sizeof(info);
  info.version = 0;
  info.ub_size = -1;
  info.max_kernel_size = -1;
  info.conv_freq = -1;
  info.fc_freq = -1;
  info.max_fc_vector_size = -1;
  if (dmp_dv_context_get_info(ctx, (dmp_dv_info*)&info)) {
    ERR("dmp_dv_context_get_info() failed: %s\n", dmp_dv_get_last_error_message());
    dmp_dv_context_release(ctx);
    return -1;
  }

  LOG("ub_size=%d\nmax_kernel_size=%d\nconv_freq=%d\nfc_freq=%d\nmax_fc_vector_size=%d\n",
      info.ub_size, info.max_kernel_size, info.conv_freq, info.fc_freq, info.max_fc_vector_size);

  if ((info.ub_size < 0) || (info.max_kernel_size < 0) ||
      (info.conv_freq < 0) || (info.fc_freq < 0) || (info.max_fc_vector_size < 0)) {
    ERR("dmp_dv_context_get_info() returned some invalid values\n");
    dmp_dv_context_release(ctx);
    return -1;
  }

  dmp_dv_context_release(ctx);

  static int s_n_fd = -1;
  int result = 0;
  int n_fd = 0;
  DIR *d;
  struct dirent *dir;
  d = opendir("/proc/self/fd");
  if (!d) {
    ERR("Could not open \"/proc/self/fd\" folder\n");
    return -1;
  }
  while ((dir = readdir(d))) {
    char *fnme = dir->d_name;
    bool num = true;
    for (; *fnme; ++fnme) {
      if ((*fnme >= '0') && (*fnme <= '9')) {
        continue;
      }
      num = false;
      break;
    }
    if (num) {
      ++n_fd;
    }
  }
  closedir(d);

  if (s_n_fd == -1) {
    s_n_fd = n_fd;
  }
  if (n_fd != s_n_fd) {
    ERR("Inconsistent file descriptor count detected, memory leak is probable\n");
    result = -1;
  }

  LOG("EXIT: test_context: %d FDs\n", n_fd);
  return result;
}


int main(int argc, char **argv) {
  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  for (int i = 0; i < 3; ++i) {
    res = test_context();
    if (res) {
      ++n_err;
    }
    else {
      ++n_ok;
    }
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);
  return n_err;
}
