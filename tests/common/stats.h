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
 * @brief Execution statistics helper functions.
 */
#pragma once
#ifndef _STATS_H_
#define _STATS_H_

#ifndef _WIN32
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>
#else
#include <windows.h>
#include <psapi.h>
#include <stdint.h>
#endif


/// @brief Measures memory consumption.
static inline int get_exec_stats(long *max_mem_kb, double *utime, double *stime) {
#ifndef _WIN32

  struct rusage res;
  memset(&res, 0, sizeof(res));
  int n = getrusage(RUSAGE_SELF, &res);
  *max_mem_kb = res.ru_maxrss;
  *utime = (double)res.ru_utime.tv_sec + 0.000001 * res.ru_utime.tv_usec;
  *stime = (double)res.ru_stime.tv_sec + 0.000001 * res.ru_stime.tv_usec;
  return n;

#else

  PROCESS_MEMORY_COUNTERS res;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &res, sizeof(res))) {
    return -1;
  }
  *max_mem_kb = (long)(res.PeakWorkingSetSize >> 10);

  uint64_t tc, te, tk, tu;
  if (!GetProcessTimes(GetCurrentProcess(), (FILETIME*)&tc, (FILETIME*)&te, (FILETIME*)&tk, (FILETIME*)&tu)) {
    return -1;
  }
  *utime = 1.0e-7 * tu;
  *stime = 1.0e-7 * tk;

  return 0;

#endif
}


/// @brief Returns monotonic time in seconds related to unspecified point.
static inline double get_clock_sec() {
#ifndef _WIN32

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * ts.tv_nsec;

#else

  int64_t ts, fr;
  QueryPerformanceCounter((LARGE_INTEGER*)&ts);
  QueryPerformanceFrequency((LARGE_INTEGER*)&fr);
  return (double)ts / fr;

#endif
}


#ifdef __cplusplus
#ifndef _WIN32
/// @brief Class for delta time measurements.
class TimeIntervalEx {
 public:
  /// @brief Constructs object for delta time measurements.
  /// @param clock_id Clock ID from clock_gettime():
  ///                 CLOCK_MONOTONIC - to measure real time difference,
  ///                 CLOCK_PROCESS_CPUTIME_ID - to measure process CPU time difference,
  ///                 CLOCK_THREAD_CPUTIME_ID - to measure current thread CPU time difference.
  TimeIntervalEx(int clock_id = CLOCK_MONOTONIC) {
    clock_id_ = clock_id;
    reset();
  }

  void reset() {
    clock_gettime(clock_id_, &ts0_);
  }

  double get_ms() {
    struct timespec ts1;
    clock_gettime(clock_id_, &ts1);
    auto dt_sec = ts1.tv_sec - ts0_.tv_sec;
    auto dt_nsec = ts1.tv_nsec - ts0_.tv_nsec;
    if (dt_nsec < 0) {
      dt_sec -= 1;
      dt_nsec += 1000000000;
    }
    return 1.0e3 * dt_sec + 1.0e-6 * dt_nsec;
  }

  int get_us() {
    struct timespec ts1;
    clock_gettime(clock_id_, &ts1);
    auto dt_sec = ts1.tv_sec - ts0_.tv_sec;
    auto dt_nsec = ts1.tv_nsec - ts0_.tv_nsec;
    if (dt_nsec < 0) {
      dt_sec -= 1;
      dt_nsec += 1000000000;
    }
    return (int)dt_sec * 1000000 + dt_nsec / 1000;
  }

 private:
  struct timespec ts0_;
  int clock_id_;
};

/// @brief Helper class for delta time measurements of real time.
class TimeIntervalReal : public TimeIntervalEx {
 public:
  TimeIntervalReal() : TimeIntervalEx(CLOCK_MONOTONIC) {
  }
};

/// @brief Helper class for delta time measurements of process CPU time.
/// @details If the process was using 8 cores for 1 second, the time will be 8 seconds.
class TimeIntervalProcess : public TimeIntervalEx {
 public:
  TimeIntervalProcess() : TimeIntervalEx(CLOCK_PROCESS_CPUTIME_ID) {
  }
};

/// @brief Helper class for delta time measurements of thread CPU time.
class TimeIntervalThread : public TimeIntervalEx {
 public:
  TimeIntervalThread() : TimeIntervalEx(CLOCK_THREAD_CPUTIME_ID) {
  }
};

/// @brief Select the default clock.
typedef TimeIntervalReal TimeInterval;

#endif
#endif

#endif  // _STATS_H_
