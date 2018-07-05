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

    if (!cmd->input_buf.mem) {
      SET_ERR("Invalid argument: cmd->input_buf.mem is NULL");
      return -1;
    }

    if (!cmd->output_buf.mem) {
      SET_ERR("Invalid argument: cmd->output_buf.mem is NULL");
      return -1;
    }

    if (!cmd->topo) {
      SET_ERR("Invalid argument: cmd->topo is 0");
      return -1;
    }

    for (int topo = cmd->topo, i = 0; topo; topo >>= 1, ++i) {
      if ((!cmd->run[i].conv_enable) && (!cmd->run[i].pool_enable) && (!cmd->run[i].actfunc)) {
        SET_ERR("Invalid argument: cmd->run[%d] specify no operation", i);
        return -1;
      }
      if ((cmd->run[i].conv_enable == 1) && (!cmd->run[i].weight_buf.mem)) {
        SET_ERR("Invalid argument: cmd->run[%d].weight_buf.mem is NULL", i);
        return -1;
      }
    }

    // TODO: add cmd to the queue.

    return 0;
  }

 private:
  CDVContext *ctx_;
};
