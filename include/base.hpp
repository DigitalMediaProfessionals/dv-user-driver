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

  /// @brief Decrements reference counter, when reference counter reaches zero, destroys an object.
  /// @return Reference counter value after decrement.
  int Release() {
    int n = __sync_sub_and_fetch(&n_ref_, 1);
    if (n > 0) {
      return n;
    }
    if (n < 0) {
      fprintf(stderr, "WARNING: CDMPDVBase::Release(): Negative reference counter detected (addr=%zu), this should not happen\n",
              (size_t)this);
      fflush(stderr);
      return n;
    }
    delete this;
    return n;
  }

  /// @brief Increments reference counter.
  /// @return Reference counter value after increment.
  inline int Retain() {
    return __sync_add_and_fetch(&n_ref_, 1);
  }

 protected:
  /// @brief Reference counter.
  int n_ref_;
};
