/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief dv_cmdlist implementation.
 */
#pragma once

#include "common.hpp"
#include "context.hpp"


/// @brief Implementation of dv_cmdlist.
class CDVCmdList {
 public:
  CDVCmdList() {
    ctx_ = NULL;
  }

  virtual ~CDVCmdList() {
    Cleanup();
  }

  bool Initialize(CDVContext *ctx) {
    Cleanup();
    if (!ctx) {
      SET_ERR("Invalid argument: ctx is NULL");
      return NULL;
    }
    ctx_ = ctx;

    return true;
  }

  void Cleanup() {
    ctx_ = NULL;
  }

  int End() {
    // TODO: implement.
    return 0;
  }

  int Exec() {
    // TODO: implement.
    return 0;
  }

  int AddRaw(dv_cmdraw *cmd) {
    // TODO: implement.
    return 0;
  }

  static inline int32_t get_cmdraw_max_version() {
    return 0;
  }

 private:
  CDVContext *ctx_;
};
