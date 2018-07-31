/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
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
  LOG("ENTER: test_api(%zu)\n", size);
  LOG("dmp_dv_get_version_string(): %s\n", dmp_dv_get_version_string());

  dmp_dv_context *ctx = dmp_dv_context_create(NULL);
  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    return -1;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  dmp_dv_mem *mem = dmp_dv_mem_alloc(ctx, size);
  if (!mem) {
    ERR("dmp_dv_mem_alloc() failed: %s\n", dmp_dv_get_last_error_message());
    dmp_dv_context_release(ctx);
    return -1;
  }
  LOG("Successfully allocated %zu bytes of memory (%zu requested)\n", dmp_dv_mem_get_size(mem), size);

  uint8_t *arr = dmp_dv_mem_map(mem);
  if (!arr) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  LOG("Successfully mapped to user space %zu bytes of memory, address is %zu\n", size, (size_t)arr);
  if (dmp_dv_mem_sync_start(mem, 0, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for writing: %s\n", dmp_dv_get_last_error_message());
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  memset(arr, 0, size);
  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }

  // Start Read-Write memory transaction
  if (dmp_dv_mem_sync_start(mem, 1, 1)) {
    ERR("dmp_dv_mem_sync_start() failed for writing: %s\n", dmp_dv_get_last_error_message());
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  const int n = size >> 2;
  volatile uint32_t *buf = (uint32_t*)arr;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const uint32_t s0 = (uint32_t)(ts.tv_sec * 1000000 + ts.tv_nsec / 1000);  // in microseconds
  uint32_t state[1];
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
      dmp_dv_mem_release(mem);
      dmp_dv_context_release(ctx);
      return -1;
    }
  }
  if (dmp_dv_mem_sync_end(mem)) {
    ERR("dmp_dv_mem_sync_end() failed: %s\n", dmp_dv_get_last_error_message());
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  LOG("Successfully filled the buffer and synced with physical memory\n");

  dmp_dv_mem_unmap(mem);
  LOG("Unmapped the buffer\n");

  // Map at the previous address
  void *ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!ptr) {
    ERR("mmap() failed at the previous address\n");
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  memset(ptr, 0, 4096);

  const size_t sz_big = 1 * 1024 * 1024;
  std::unique_ptr<uint8_t> big_ptr(new uint8_t[sz_big]);
  uint8_t *big = big_ptr.get();
  if (!big) {
    ERR("malloc() failed for 64Mb\n");
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  memset(big, 0xFF, sz_big);

  arr = dmp_dv_mem_map(mem);
  if (!arr) {
    ERR("dmp_dv_mem_map() failed: %s\n", dmp_dv_get_last_error_message());
    munmap(ptr, 4096);
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  LOG("Successfully mapped to user space %zu bytes of memory, address is %zu\n", size, (size_t)arr);
  munmap(ptr, 4096);

  size_t a_start = (size_t)big;
  size_t a_end = a_start + sz_big;
  size_t a_map = (size_t)arr;
  if (!((a_map + size <= a_start) || (a_map >= a_end))) {
    ERR("Memory allocator returned bad address: [%zu, %zu] map=%zu\n", a_start, a_end, a_map);
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
    return -1;
  }
  const uint32_t *big32 = (uint32_t*)big;
  for (int i = 0; i < (int)(sz_big >> 2); ++i) {
    if (big32[i] != 0xFFFFFFFF) {
      ERR("Unexpected value encountered\n");
      return -1;
    }
  }
  big_ptr.reset();

  // Start Read-Only memory transaction
  if (dmp_dv_mem_sync_start(mem, 1, 0)) {
    ERR("dmp_dv_mem_sync_start() failed for reading: %s", dmp_dv_get_last_error_message());
    dmp_dv_mem_release(mem);
    dmp_dv_context_release(ctx);
  }

  buf = (uint32_t*)arr;
  state[0] = s0;
  for (int i = 0; i < n; ++i) {
    uint32_t gold = xorshift32(state);
    if (buf[i] != gold) {
      ERR("%u != %u at %d\n", buf[i], gold, i);
      dmp_dv_mem_release(mem);
      dmp_dv_context_release(ctx);
      return -1;
    }
  }

  dmp_dv_mem_release(mem);
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

  LOG("EXIT: test_api(%zu): %d FDs\n", size, n_fd);
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
