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
#include "../../dv-kernel-driver/uapi/dmp_dv_cmdraw_v0.h"


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
    dmp_dv_kcmdraw_v0 *raw_commands = (dmp_dv_kcmdraw_v0*)malloc(sizeof(dmp_dv_kcmdraw_v0) * n);
    if (!raw_commands) {
      SET_ERR("Failed to allocate %zu bytes of memory", sizeof(dmp_dv_kcmdraw_v0) * n);
      return -1;
    }

    for (int i = 0; i < n; ++i) {
      raw_commands[i].size = sizeof(dmp_dv_kcmdraw_v0);
      raw_commands[i].version = 0;

      raw_commands[i].input_buf.fd = CDMPDVMem::get_fd(commands_[i].raw_v0.input_buf.mem);
      raw_commands[i].input_buf.rsvd = 0;
      raw_commands[i].input_buf.offs = commands_[i].raw_v0.input_buf.offs;

      raw_commands[i].output_buf.fd = CDMPDVMem::get_fd(commands_[i].raw_v0.output_buf.mem);
      raw_commands[i].output_buf.rsvd = 0;
      raw_commands[i].output_buf.offs = commands_[i].raw_v0.output_buf.offs;

      raw_commands[i].eltwise_buf.fd = CDMPDVMem::get_fd(commands_[i].raw_v0.eltwise_buf.mem);
      raw_commands[i].eltwise_buf.rsvd = 0;
      raw_commands[i].eltwise_buf.offs = commands_[i].raw_v0.eltwise_buf.offs;

      raw_commands[i].topo = commands_[i].raw_v0.topo;
      raw_commands[i].w = commands_[i].raw_v0.w;
      raw_commands[i].h = commands_[i].raw_v0.h;
      raw_commands[i].z = commands_[i].raw_v0.z;
      raw_commands[i].c = commands_[i].raw_v0.c;
      raw_commands[i].input_circular_offset = commands_[i].raw_v0.input_circular_offset;
      raw_commands[i].output_mode = commands_[i].raw_v0.output_mode;

      const int n_run = 32;
      for (int i_run = 0; i_run < n_run; ++i_run) {
        raw_commands[i].run[i_run].weight_buf.fd = CDMPDVMem::get_fd(commands_[i].raw_v0.run[i_run].weight_buf.mem);
        raw_commands[i].run[i_run].weight_buf.rsvd = 0;
        raw_commands[i].run[i_run].weight_buf.offs = commands_[i].raw_v0.run[i_run].weight_buf.offs;

        raw_commands[i].run[i_run].conv_pad = commands_[i].raw_v0.run[i_run].conv_pad;
        raw_commands[i].run[i_run].pool_pad = commands_[i].raw_v0.run[i_run].pool_pad;
        raw_commands[i].run[i_run].m = commands_[i].raw_v0.run[i_run].m;
        raw_commands[i].run[i_run].conv_enable = commands_[i].raw_v0.run[i_run].conv_enable;
        raw_commands[i].run[i_run].p = commands_[i].raw_v0.run[i_run].p;
        raw_commands[i].run[i_run].pz = commands_[i].raw_v0.run[i_run].pz;
        raw_commands[i].run[i_run].conv_stride = commands_[i].raw_v0.run[i_run].conv_stride;
        raw_commands[i].run[i_run].conv_dilation = commands_[i].raw_v0.run[i_run].conv_dilation;
        raw_commands[i].run[i_run].weight_fmt = commands_[i].raw_v0.run[i_run].weight_fmt;
        raw_commands[i].run[i_run].pool_enable = commands_[i].raw_v0.run[i_run].pool_enable;
        raw_commands[i].run[i_run].pool_avg_param = commands_[i].raw_v0.run[i_run].pool_avg_param;
        raw_commands[i].run[i_run].pool_size = commands_[i].raw_v0.run[i_run].pool_size;
        raw_commands[i].run[i_run].pool_stride = commands_[i].raw_v0.run[i_run].pool_stride;
        raw_commands[i].run[i_run].actfunc = commands_[i].raw_v0.run[i_run].actfunc;
        raw_commands[i].run[i_run].actfunc_param = commands_[i].raw_v0.run[i_run].actfunc_param;
        raw_commands[i].run[i_run].rectifi_en = commands_[i].raw_v0.run[i_run].rectifi_en;
        raw_commands[i].run[i_run].lrn = commands_[i].raw_v0.run[i_run].lrn;
        raw_commands[i].run[i_run].rsvd = commands_[i].raw_v0.run[i_run].rsvd;
      }
    }

    // Pass this chunk to kernel module
    dmp_dv_kcmd dv_cmd;
    dv_cmd.cmd_num = n;
    dv_cmd.cmd_pointer = (__u64)raw_commands;
    int res = ioctl(ctx_->get_fd_conv(), DMP_DV_IOC_APPEND_CMD, &dv_cmd);
    if (res < 0) {
      SET_IOCTL_ERR("/dev/dv_conv", "DMP_DV_IOC_APPEND_CMD");
      res = -1;
    }
    else {
      res = 0;
    }

    // Free temporary buffer
    free(raw_commands);

    return res;
  }

  int Exec() {
    // Issue ioctl on the kernel module requesting this list execution
    int res = ioctl(ctx_->get_fd_conv(), DMP_DV_IOC_RUN, NULL);
    if (res < 0) {
      SET_IOCTL_ERR("/dev/dv_conv", "DMP_DV_IOC_RUN");
      return -1;
    }

    // TODO: adjust when support for multiple lists will be implemented in kernel module.

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
