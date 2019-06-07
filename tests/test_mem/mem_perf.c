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
 * @brief Tests for memory allocation/mapping/updating.
 */
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <dirent.h>

#include <stdio.h>
#include <string.h>

#include <dmp_dv.h>


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


/* The state array must be initialized to not be all zero */
uint32_t xorshift128(uint32_t state[4]) {
    /* Algorithm "xor128" from p. 5 of Marsaglia, "Xorshift RNGs" */
    uint32_t s, t = state[3];
    t ^= t << 11;
    t ^= t >> 8;
    state[3] = state[2]; state[2] = state[1]; state[1] = s = state[0];
    t ^= s;
    t ^= s >> 19;
    state[0] = t;
    return t;
}


static double get_ms(struct timespec *ts0, struct timespec *ts1) {
  time_t dt_sec = ts1->tv_sec - ts0->tv_sec;
  long dt_nsec = ts1->tv_nsec - ts0->tv_nsec;
  if (dt_nsec < 0) {
    dt_sec -= 1;
    dt_nsec += 1000000000;
  }
  return 1.0e3 * dt_sec + 1.0e-6 * dt_nsec;
}


int mem_perf(size_t size, uint32_t state[4]) {
  LOG("ENTER: mem_perf(%zu)\n", size);

  uint32_t saved_state[4];
  int result = -1;
  struct timespec ts0, ts1;
  uint32_t *arr = NULL;
  dmp_dv_mem mem = NULL;

  dmp_dv_context ctx = dmp_dv_context_create();
  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  mem = dmp_dv_mem_alloc(ctx, size);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  if (!mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("mem_alloc(%zu): %.3f msec\n", size, get_ms(&ts0, &ts1));

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  arr = (uint32_t*)dmp_dv_mem_map(mem);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  if (!arr) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("mem_map(%zu): %.3f msec\n", size, get_ms(&ts0, &ts1));

  // WRITE
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (dmp_dv_mem_sync_start(mem, 0, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for WRITE: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  LOG("sync_start(%zu, %s): %.3f msec\n", size, "WRITE", get_ms(&ts0, &ts1));
  memcpy(saved_state, state, 4 * 4);
  for (int i = 0; i < (size >> 2); ++i) {
    arr[i] = xorshift128(state);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  LOG("sync_end(%zu, %s): %.3f msec\n", size, "WRITE", get_ms(&ts0, &ts1));

  // READ WRITE
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (dmp_dv_mem_sync_start(mem, 1, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for RW: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  LOG("sync_start(%zu, %s): %.3f msec\n", size, "RW", get_ms(&ts0, &ts1));
  memcpy(state, saved_state, 4 * 4);
  for (int i = 0; i < (size >> 2); ++i) {
    if (arr[i] != xorshift128(state)) {
      ERR("Memory contents changed\n");
      goto L_EXIT;
    }
  }
  memcpy(saved_state, state, 4 * 4);
  for (int i = 0; i < (size >> 2); ++i) {
    arr[i] = xorshift128(state);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  LOG("sync_end(%zu, %s): %.3f msec\n", size, "RW", get_ms(&ts0, &ts1));

  // READ
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (dmp_dv_mem_sync_start(mem, 1, 0)) {
    ERR("dmp_dv_mem_sync_start() failed for READ: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  LOG("sync_start(%zu, %s): %.3f msec\n", size, "READ", get_ms(&ts0, &ts1));
  memcpy(state, saved_state, 4 * 4);
  for (int i = 0; i < (size >> 2); ++i) {
    if (arr[i] != xorshift128(state)) {
      ERR("Memory contents changed\n");
      goto L_EXIT;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  LOG("sync_end(%zu, %s): %.3f msec\n", size, "READ", get_ms(&ts0, &ts1));

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  dmp_dv_mem_unmap(mem);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  LOG("unmap(%zu): %.3f msec\n", size, get_ms(&ts0, &ts1));

  result = 0;

  L_EXIT:

  dmp_dv_mem_release(mem);
  dmp_dv_context_release(ctx);

  if (dmp_dv_mem_get_total_size()) {
    ERR("dmp_dv_mem_get_total_size() returned non-zero: %lld\n",
        (long long)dmp_dv_mem_get_total_size());
    result = -1;
  }

  static int s_n_fd = -1;
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
    int num = 1;
    for (; *fnme; ++fnme) {
      if ((*fnme >= '0') && (*fnme <= '9')) {
        continue;
      }
      num = 0;
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

  LOG("EXIT%s: mem_perf(%zu): %d FDs\n", result ? "(FAILED)" : "", size, n_fd);
  return result;
}


int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stdout, "USAGE: ./test_mem N_KB\n");
    return 1;
  }
  int n_kb = atoi(argv[1]);
  if (n_kb < 4) {
    n_kb = 4;
  }
  if ((n_kb & 3)) {
    n_kb += 4 - (n_kb & 3);
  }

  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  struct timespec ts0;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  uint32_t state[4] = {(uint32_t)ts0.tv_sec, (uint32_t)ts0.tv_nsec, 3, 4};

  for (int i = 0; i < 1; ++i) {
    res = mem_perf(n_kb << 10, state);
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
