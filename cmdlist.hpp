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

#include "mem.hpp"
#include "dv_cmdraw_v0.h"


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
    if (cmd->size < 8) {
      SET_ERR("Invalid argument: cmd->size %d is too small", (int)cmd->size);
      return -1;
    }
    switch (cmd->version) {
      case 0:
        return AddRaw_v0((dv_cmdraw_v0*)cmd);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return -1;
    }
    SET_ERR("Control should not reach line %d of file %s", __LINE__, __FILE__);
    return -1;
  }

  static inline int32_t get_cmdraw_max_version() {
    return 0;
  }

 protected:
  int AddRaw_v0(dv_cmdraw_v0 *cmd) {
    if (cmd->size != sizeof(dv_cmdraw_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d", (int)cmd->size, (int)cmd->version);
      return -1;
    }

    // TODO: implement.

    return 0;
  }

 private:
  CDVContext *ctx_;
};
