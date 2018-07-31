/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief Base object with reference counting and common methods.
 */
#pragma once

#include "common.h"


/// @brief Base object with reference counting and common methods.
class CDMPDVBase {
 public:
  /// @brief Constructor, sets reference counter to 1.
  CDMPDVBase() {
    n_ref_ = 1;
  }

  /// @brief Copy constructor (disabled).
  CDMPDVBase(const CDMPDVBase& src) = delete;

  /// @brief Move constructor (disabled).
  CDMPDVBase(CDMPDVBase&& src) = delete;

  /// @brief Copy assignment (disabled).
  CDMPDVBase& operator=(const CDMPDVBase& src) = delete;

  /// @brief Move assignment (disabled).
  CDMPDVBase& operator=(const CDMPDVBase&& src) = delete;

  /// @brief Destructor.
  virtual ~CDMPDVBase() {
    // Empty by design
  }

  /// @brief Releases all held resources.
  virtual void Cleanup() = 0;

  /// @brief Decrements reference counter, when reference counter reaches zero, destroys an object.
  /// @return Reference counter value after decrement.
  virtual int Release() {
    int n = __sync_sub_and_fetch(&n_ref_, 1);
    if (n > 0) {
      return n;
    }
    if (n < 0) {
      char s[256];
      fill_debug_info(s, sizeof(s));
      fprintf(stderr, "WARNING: negative reference counter detected on %s\n", s);
      fflush(stderr);
      return n;
    }
    delete this;
    return n;
  }

  /// @brief Increments reference counter.
  /// @return Reference counter value after increment.
  virtual int Retain() {
    return __sync_add_and_fetch(&n_ref_, 1);
  }

  /// @brief Fills debug information for an object.
  virtual void fill_debug_info(char *info, int length) = 0;

 protected:
  /// @brief Reference counter.
  int n_ref_;
};
