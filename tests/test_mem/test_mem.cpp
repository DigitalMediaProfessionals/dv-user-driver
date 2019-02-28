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

#include <memory>

#include "dmp_dv.h"


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


inline uint32_t xorshift32(uint32_t state[1]) {
  uint32_t x = state[0];
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state[0] = x;
  return x;
}


int test_mem(size_t size) {
  LOG("ENTER: test_mem(%zu)\n", size);

  dmp_dv_context ctx = NULL;
  dmp_dv_mem mem = NULL;
  int result = -1;
  uint8_t *arr = NULL;
  void *ptr = NULL;
  const size_t sz_big = 1 * 1024 * 1024;
  std::unique_ptr<uint8_t> big_ptr;
  uint8_t *big = NULL;
  const int n = size >> 2;
  volatile uint32_t *buf = NULL;
  struct timespec ts;
  uint32_t state[1];
  uint32_t s0;
  size_t a_start, a_end, a_map;
  uint32_t *big32 = NULL;

  ctx = dmp_dv_context_create();
  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  mem = dmp_dv_mem_alloc(ctx, size);
  if (!mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully allocated %zu bytes of memory (%zu requested)\n", dmp_dv_mem_get_size(mem), size);

  arr = dmp_dv_mem_map(mem);
  if (!arr) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully mapped to user space %zu bytes of memory, address is %zu\n", size, (size_t)arr);
  if (dmp_dv_mem_sync_start(mem, 0, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for writing: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  memset(arr, 0, size);
  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  // Start Read-Write memory transaction
  if (dmp_dv_mem_sync_start(mem, 1, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for writing: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  buf = (uint32_t*)arr;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  s0 = (uint32_t)(ts.tv_sec * 1000000 + ts.tv_nsec / 1000);  // in microseconds
  state[0] = {s0};
  for (int i = 0; i < n; ++i) {
    uint32_t gold = xorshift32(state);
    buf[i] = gold;
  }
  state[0] = {s0};
  for (int i = 0; i < n; ++i) {
    uint32_t gold = xorshift32(state);
    if (buf[i] != gold) {
      ERR("Cache incoherence detected at first stage: %u != %u at %d\n", buf[i], gold, i);
      goto L_EXIT;
    }
  }
  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully filled the buffer and synced with physical memory\n");

  dmp_dv_mem_unmap(mem);
  LOG("Unmapped the buffer\n");

  // Map at the previous address
  ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!ptr) {
    ERR("mmap() failed at the previous address\n");
    goto L_EXIT;
  }
  memset(ptr, 0, 4096);

  big_ptr.reset(new uint8_t[sz_big]);
  big = big_ptr.get();
  if (!big) {
    ERR("malloc() failed for 64Mb\n");
    goto L_EXIT;
  }
  memset(big, 0xFF, sz_big);

  arr = dmp_dv_mem_map(mem);
  if (!arr) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully mapped to user space %zu bytes of memory, address is %zu\n", size, (size_t)arr);
  munmap(ptr, 4096);

  a_start = (size_t)big;
  a_end = a_start + sz_big;
  a_map = (size_t)arr;
  if (!((a_map + size <= a_start) || (a_map >= a_end))) {
    ERR("Memory allocator returned bad address: [%zu, %zu] map=%zu\n", a_start, a_end, a_map);
    goto L_EXIT;
  }
  big32 = (uint32_t*)big;
  for (int i = 0; i < (int)(sz_big >> 2); ++i) {
    if (big32[i] != 0xFFFFFFFF) {
      ERR("Unexpected value encountered\n");
      goto L_EXIT;
    }
  }
  big_ptr.reset();

  // Start Read-Only memory transaction
  if (dmp_dv_mem_sync_start(mem, 1, 0)) {
    ERR("dmp_dv_mem_sync_start() failed for reading: %s", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }

  buf = (uint32_t*)arr;
  state[0] = s0;
  for (int i = 0; i < n; ++i) {
    uint32_t gold = xorshift32(state);
    if (buf[i] != gold) {
      ERR("%u != %u at %d\n", buf[i], gold, i);
      goto L_EXIT;
    }
  }

  if (dmp_dv_mem_get_total_size() != (int64_t)dmp_dv_mem_get_size(mem)) {
    ERR("dmp_dv_mem_get_total_size() returned different value %lld than expected %lld\n",
        (long long)dmp_dv_mem_get_total_size(), (long long)dmp_dv_mem_get_size(mem));
    goto L_EXIT;
  }

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

  LOG("EXIT%s: test_mem(%zu): %d FDs\n", result ? "(FAILED)" : "", size, n_fd);
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

  for (int i = 0; i < 1000; ++i) {
    res = test_mem(n_kb << 10);
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
