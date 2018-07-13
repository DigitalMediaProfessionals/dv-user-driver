/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief dmp_dv_cmdlist implementation.
 */
#pragma once

#include <vector>

#include "mem.hpp"
#include "dmp_dv_cmdraw_v0.h"


/// @brief Implementation of dmp_dv_cmdlist.
class CDMPDVCmdList {
 public:
  CDMPDVCmdList() {
    ctx_ = NULL;
  }

  virtual ~CDMPDVCmdList() {
    Cleanup();
  }

  bool Initialize(CDMPDVContext *ctx) {
    Cleanup();
    if (!ctx) {
      SET_ERR("Invalid argument: ctx is NULL");
      return NULL;
    }
    ctx_ = ctx;

    // TODO: increase reference counter on context.

    return true;
  }

  void Cleanup() {
    commands_.clear();
    ctx_ = NULL;
  }

  int End() {
    // Check list content
    const int n = (int)commands_.size();
    for (int i = 0; i < n; ++i) {
      switch (commands_[i].type) {
        case kCommandTypeRaw_v0:
          break;
        default:
          SET_ERR("Invalid command type %d detected at command list at position %d", commands_[i].type, i);
          return -1;
      }
    }

    // Allocate and fill memory chunk suitable for sharing with kernel module replacing dmp_dv_mem pointers with ION file descriptors
    // TODO: implement.

    // Pass this chunk to kernel module
    // TODO: implement.

    return 0;
  }

  int Exec() {
    // Issue ioctl on the kernel module requesting this list execution
    // TODO: implement.
    return 0;
  }

  int AddRaw(dmp_dv_cmdraw *cmd) {
    if (cmd->size < 8) {
      SET_ERR("Invalid argument: cmd->size %d is too small", (int)cmd->size);
      return -1;
    }
    switch (cmd->version) {
      case 0:
        return AddRaw_v0((dmp_dv_cmdraw_v0*)cmd);

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
  enum CommandType {
    kCommandTypeSTART = 0,
    kCommandTypeRaw_v0,
    kCommandTypeEND
  };

  struct Command {
    union {
      CommandType type;  // command type
      uint64_t rsvd;     // padding to 64-bit size
    };
    union {
      dmp_dv_cmdraw_v0 raw_v0;  // command content
    };
  };

  int AddRaw_v0(dmp_dv_cmdraw_v0 *cmd) {
    if (cmd->size != sizeof(dmp_dv_cmdraw_v0)) {
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

    // TODO: increase reference counters on provided mem pointers.

    const int n = (int)commands_.size();
    commands_.resize(n + 1);
    Command *command = &commands_[n];
    memset(command, 0, sizeof(Command));
    command->type = kCommandTypeRaw_v0;
    memcpy(&command->raw_v0, cmd, sizeof(dmp_dv_cmdraw_v0));

    return 0;
  }

 private:
  /// @brief Reference to device context.
  CDMPDVContext *ctx_;

  /// @brief List of commands this list contains.
  std::vector<Command> commands_;
};