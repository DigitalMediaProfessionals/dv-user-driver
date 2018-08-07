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

#include "cmdlist.hpp"
#include "dmp_dv_cmdraw_v0.h"
#include "../../dv-kernel-driver/uapi/dmp_dv_cmdraw_v0.h"


/// @brief Implementation of dmp_dv_cmdlist.
class CDMPDVCmdListConv : public CDMPDVCmdList {
 public:
  /// @brief Constructor.
  CDMPDVCmdListConv() : CDMPDVCmdList() {
    // Empty by design
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdListConv() {
    ReleaseResources();
  }

  /// @brief Adds raw structure describing the command for convolutional module.
  virtual int AddRaw(dmp_dv_cmdraw *cmd) {
    int res = CheckAddRaw(cmd, DMP_DV_CONV);
    if (res) {
      return res;
    }
    switch (cmd->version) {
      case 0:
        return AddRaw_v0((dmp_dv_cmdraw_conv_v0*)cmd);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_ERR("Control should not reach line %d of file %s", __LINE__, __FILE__);
    return -1;
  }

  /// @brief Commits command list for convolutional accelerator.
  virtual int Commit() {
    const int n = (int)commands_.size();

    // Allocate and fill memory chunk suitable for sharing with kernel module replacing dmp_dv_mem pointers with ION file descriptors
    dmp_dv_kcmdraw_v0 *raw_commands = (dmp_dv_kcmdraw_v0*)malloc(sizeof(dmp_dv_kcmdraw_v0) * n);
    if (!raw_commands) {
      SET_ERR("Failed to allocate %zu bytes of memory", sizeof(dmp_dv_kcmdraw_v0) * n);
      return -1;
    }

    for (int i = 0; i < n; ++i) {
      dmp_dv_cmdraw_conv_v0& cmd = commands_[i];
      raw_commands[i].size = sizeof(dmp_dv_kcmdraw_v0);
      raw_commands[i].version = 0;

      raw_commands[i].input_buf.fd = CDMPDVMem::get_fd(cmd.input_buf.mem);
      raw_commands[i].input_buf.rsvd = 0;
      raw_commands[i].input_buf.offs = cmd.input_buf.offs;

      raw_commands[i].output_buf.fd = CDMPDVMem::get_fd(cmd.output_buf.mem);
      raw_commands[i].output_buf.rsvd = 0;
      raw_commands[i].output_buf.offs = cmd.output_buf.offs;

      raw_commands[i].eltwise_buf.fd = CDMPDVMem::get_fd(cmd.eltwise_buf.mem);
      raw_commands[i].eltwise_buf.rsvd = 0;
      raw_commands[i].eltwise_buf.offs = cmd.eltwise_buf.offs;

      raw_commands[i].topo = cmd.topo;
      raw_commands[i].w = cmd.w;
      raw_commands[i].h = cmd.h;
      raw_commands[i].z = cmd.z;
      raw_commands[i].c = cmd.c;
      raw_commands[i].input_circular_offset = cmd.input_circular_offset;
      raw_commands[i].output_mode = cmd.output_mode;

      const int n_run = 32;
      for (int i_run = 0; i_run < n_run; ++i_run) {
        raw_commands[i].run[i_run].weight_buf.fd = CDMPDVMem::get_fd(cmd.run[i_run].weight_buf.mem);
        raw_commands[i].run[i_run].weight_buf.rsvd = 0;
        raw_commands[i].run[i_run].weight_buf.offs = cmd.run[i_run].weight_buf.offs;
        raw_commands[i].run[i_run].conv_pad = cmd.run[i_run].conv_pad;
        raw_commands[i].run[i_run].pool_pad = cmd.run[i_run].pool_pad;
        raw_commands[i].run[i_run].m = cmd.run[i_run].m;
        raw_commands[i].run[i_run].conv_enable = cmd.run[i_run].conv_enable;
        raw_commands[i].run[i_run].p = cmd.run[i_run].p;
        raw_commands[i].run[i_run].pz = cmd.run[i_run].pz;
        raw_commands[i].run[i_run].conv_stride = cmd.run[i_run].conv_stride;
        raw_commands[i].run[i_run].conv_dilation = cmd.run[i_run].conv_dilation;
        raw_commands[i].run[i_run].weight_fmt = cmd.run[i_run].weight_fmt;
        raw_commands[i].run[i_run].pool_enable = cmd.run[i_run].pool_enable;
        raw_commands[i].run[i_run].pool_avg_param = cmd.run[i_run].pool_avg_param;
        raw_commands[i].run[i_run].pool_size = cmd.run[i_run].pool_size;
        raw_commands[i].run[i_run].pool_stride = cmd.run[i_run].pool_stride;
        raw_commands[i].run[i_run].actfunc = cmd.run[i_run].actfunc;
        raw_commands[i].run[i_run].actfunc_param = cmd.run[i_run].actfunc_param;
        raw_commands[i].run[i_run].rectifi_en = cmd.run[i_run].rectifi_en;
        raw_commands[i].run[i_run].lrn = cmd.run[i_run].lrn;
        raw_commands[i].run[i_run].rsvd = cmd.run[i_run].rsvd;
      }
    }

    // Pass this chunk to kernel module
    int res = IssueCommitIOCTL(n, raw_commands);

    // Free temporary buffer
    free(raw_commands);

    return res;
  }

 protected:
  /// @brief Releases held resources.
  virtual void Cleanup() {
    ReleaseResources();
    CDMPDVCmdList::Cleanup();
  }

  /// @brief Opens convolutional accelerator.
  virtual bool OpenDevice() {
    return OpenDeviceAsFile("/dev/dv_conv");
  }

  /// @brief Adds raw structure describing the concolutional layer in version 0 format.
  int AddRaw_v0(dmp_dv_cmdraw_conv_v0 *cmd) {
    if (cmd->size != sizeof(dmp_dv_cmdraw_conv_v0)) {
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
      if (cmd->run[i].conv_enable == 1) {
        const int kx = cmd->run[i].p & 0xFF;
        const int ky = cmd->run[i].p >> 8;
        const int max_kernel_size = ctx_->get_max_kernel_size();
        if ((kx < 1) || (kx > max_kernel_size) || (ky < 1) || (ky > max_kernel_size)) {
          SET_ERR("Unsupported convolutional kernel size %dx%d, only sizes from 1 to %d are supported",
                  kx, ky, max_kernel_size);
          return -1;
        }
        const int sx = cmd->run[i].conv_stride & 0xFF;
        const int sy = cmd->run[i].conv_stride >> 8;
        if ((sx < 1) || (sy < 1)) {
          SET_ERR("Stride of convolution must be greater than 0, got %dx%d", sx, sy);
          return -1;
        }
      }
      // TODO: add more checks.
    }

    // Increase reference counters on the provided memory pointers
    dmp_dv_mem_retain(cmd->input_buf.mem);
    dmp_dv_mem_retain(cmd->output_buf.mem);
    for (int topo = cmd->topo, i = 0; topo; topo >>= 1, ++i) {
      dmp_dv_mem_retain(cmd->run[i].weight_buf.mem);
    }

    // Append provided command to the command list
    int n0 = (int)commands_.size();
    commands_.resize(n0 + 1);
    int n1 = (int)commands_.size();
    if ((n0 < 0) || (n1 <= 0) || (n1 != n0 + 1)) {  // sanity check
      SET_ERR("Memory allocation error at line %d of file %s", __LINE__, __FILE__);
      return -1;
    }
    memcpy(&commands_[n0], cmd, sizeof(*cmd));

    return 0;
  }

 private:
  /// @brief Releases held resources.
  void ReleaseResources() {
    // Decrease reference counters on the used memory pointers
    const int n_commands = commands_.size();
    for (int i = n_commands - 1; i >= 0; --i) {
      dmp_dv_cmdraw_conv_v0& cmd = commands_[i];
      int n_runs = 0;
      for (int topo = cmd.topo; topo; topo >>= 1) {
        ++n_runs;
      }
      for (int i_run = n_runs - 1; i_run >= 0; --i_run) {
        dmp_dv_mem_release(cmd.run[i_run].weight_buf.mem);
      }
      dmp_dv_mem_release(cmd.output_buf.mem);
      dmp_dv_mem_release(cmd.input_buf.mem);
    }
    commands_.clear();
  }

 private:
  /// @brief Added commands.
  std::vector<dmp_dv_cmdraw_conv_v0> commands_;
};
