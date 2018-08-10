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
#include <tuple>

#include "dmp_dv.h"
#include "common.h"
#include "context.hpp"
#include "mem.hpp"
#include "dmp_dv_cmdraw_v0.h"


/// @brief Forward reference for CDMPDVCmdListDeviceHelper.
class CDMPDVCmdListDeviceHelper;


/// @brief Typedef for creator for CDMPDVCmdListDeviceHelper.
typedef CDMPDVCmdListDeviceHelper* (*f_device_helper_creator)(CDMPDVContext *ctx);


/// @brief Helper object work working with command list specific to the device type (CONV or FC).
class CDMPDVCmdListDeviceHelper : public CDMPDVBase {
 public:
  /// @brief Constructor.
  CDMPDVCmdListDeviceHelper(CDMPDVContext *ctx) : CDMPDVBase() {
    ctx->Retain();
    ctx_ = ctx;
    commited_ = false;
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdListDeviceHelper() {
    ctx_->Release();
  }

  /// @brief Checks provided command for validness.
  /// @param cmd Command to check.
  /// @param input_bufs Must be extended with pairs of <buffer, size in bytes>
  ///                   used to read data during this command.
  /// @param output_bufs Must be extended with pairs of <buffer, size in bytes>
  ///                    used to write data during this command.
  /// @return 0 on success, non-zero on error.
  /// @details Validness and bounds of returned memory handles will be checked later outside of this function,
  ///          as well as reference counters on the returned memory handles will be incremented later
  ///          outside of this function.
  ///          It is safe to extend input_bufs/output_bufs and return the error.
  virtual int CheckRaw(dmp_dv_cmdraw *cmd,
                       std::vector<std::pair<dmp_dv_buf, uint64_t> >& input_bufs,
                       std::vector<std::pair<dmp_dv_buf, uint64_t> >& output_bufs) = 0;

  /// @brief Fills command in the format suitable for later execution on the device.
  /// @param kcmd Buffer to hold kernel command, can be NULL to get only size.
  /// @param cmd Command to execute (user-space format).
  /// @param size On enter must contain available space in bytes in the kcmd buffer, on exit will contain used or required space.
  /// @return 0 on success, non-zero on error.
  virtual int FillKCommand(uint8_t *kcmd, dmp_dv_cmdraw *cmd, uint32_t& size) = 0;

  /// @brief Commits command list, e.g. issues ioctl to kernel module.
  /// @param kcmdlist Command list to commit.
  /// @param size Size in bytes of the command list.
  /// @param n_commands Number of commands contained in the command list.
  /// @return 0 on sucess, non-zero on error.
  virtual int KCommit(uint8_t *kcmdlist, uint32_t size, uint32_t n_commands) = 0;

  /// @brief Schedules commited command list for execution.
  /// @return >= 0 - execution id on sucess, < 0 on error.
  virtual int64_t Exec() = 0;

  /// @brief Waits for scheduled command to be completed.
  /// @param exec_id Execution id returned by Exec() call.
  /// @return 0 on success, non-zero on error.
  virtual int Wait(int64_t exec_id) = 0;

  /// @brief Instantiates object of a given type.
  static int Instantiate(CDMPDVContext *ctx, uint8_t device_type, CDMPDVCmdListDeviceHelper **creator) {
    if (device_type >= DMP_DV_DEV_COUNT) {
      SET_ERR("Invalid argument: device_type is out of bounds: got %d while bounds are [%d, %d]",
              (int)device_type, 0, DMP_DV_DEV_COUNT - 1);
      return EINVAL;
    }
    if (!creators_[device_type]) {
      SET_ERR("Invalid argument: device_type %d is not supported", (int)device_type);
      return EINVAL;
    }
    CDMPDVCmdListDeviceHelper *obj = (*creators_[device_type])(ctx);
    if (!obj) {
      return ENOMEM;
    }
    *creator = obj;
    return 0;
  }

 protected:
  /// @brief Sets the list to be in commited state.
  inline void set_commited() {
    commited_ = true;
  }

  /// @brief Checks if a list in a commited state.
  inline bool is_commited() const {
    return commited_;
  }

  /// @brief Pointer to the context.
  CDMPDVContext *ctx_;

 private:
  /// @brief If the list is in commited state.
  bool commited_;

  /// @brief Creators for the specific device types.
  static f_device_helper_creator creators_[DMP_DV_DEV_COUNT];
};


/// @brief Helper object work working with command lists backed by DV kernel module.
class CDMPDVCmdListKHelper : public CDMPDVCmdListDeviceHelper {
 public:
  /// @brief Constructor.
  CDMPDVCmdListKHelper(CDMPDVContext *ctx) : CDMPDVCmdListDeviceHelper(ctx) {
    fd_acc_ = -1;
    fnme_acc_ = "";
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdListKHelper() {
    ReleaseResources();
  }

 protected:
  /// @brief Issues ioctl to kernel module to commit the command list.
  virtual int KCommit(uint8_t *kcmdlist, uint32_t size, uint32_t n_commands) {
    if (is_commited()) {
      SET_ERR("Command list is already in commited state");
      return EALREADY;
    }

    if (fd_acc_ == -1) {
      fd_acc_ = open(fnme_acc_, O_RDONLY | O_CLOEXEC);
      if (fd_acc_ == -1) {
        SET_ERR("open() failed for %s: %s", fnme_acc_, strerror(errno));
        return false;
      }
    }

    dmp_dv_kcmd dv_cmd;
    dv_cmd.cmd_num = n_commands;
    dv_cmd.cmd_pointer = (__u64)kcmdlist;

    int res = ioctl(fd_acc_, DMP_DV_IOC_APPEND_CMD, &dv_cmd);
    if (res < 0) {
      SET_IOCTL_ERR(fnme_acc_, "DMP_DV_IOC_APPEND_CMD");
      res = -1;
    }
    else {
      res = 0;
      set_commited();
    }
    return res;
  }

  /// @brief Schedules commited command list for execution.
  virtual int64_t Exec() {
    // Issue ioctl on the kernel module requesting this list execution
    int64_t exec_id = -1;
    int res = ioctl(fd_acc_, DMP_DV_IOC_RUN, &exec_id);
    if (res < 0) {
      SET_IOCTL_ERR(fnme_acc_, "DMP_DV_IOC_RUN");
      return -1;
    }
    if (exec_id < 0) {
      SET_ERR("ioctl(%s) on %s hasn't returned unexpected/hasn't updated exec_id=%lld",
              "DMP_DV_IOC_RUN", fnme_acc_, (long long)exec_id);
      return -1;
    }
    return exec_id;
  }

  /// @brief Waits for scheduled command to be completed.
  virtual int Wait(int64_t exec_id) {
    for (;;) {
      int res = ioctl(fd_acc_, DMP_DV_IOC_WAIT, &exec_id);
      if (!res) {
        break;
      }
      switch (res) {
        case -EBUSY:       // timeout of 2 seconds reached
        case ERESTARTSYS:  // signal has interrupted the wait
          continue;  // repeat ioctl

        default:
          SET_IOCTL_ERR(fnme_acc_, "DMP_DV_IOC_WAIT");
          return res;
      }
    }
    return 0;
  }

  /// @brief File handle for the accelerator ioctl.
  int fd_acc_;

  /// @brief File name for the accelerator ioctl.
  const char *fnme_acc_;

 private:
  /// @brief Releases held resources.
  void ReleaseResources() {
    if (fd_acc_ != -1) {
      close(fd_acc_);
      fd_acc_ = -1;
    }
  }
};


/// @brief Helper object work working with command list for CONV accelerator.
class CDMPDVCmdListConvHelper : public CDMPDVCmdListKHelper {
 public:
  /// @brief Constructor.
  CDMPDVCmdListConvHelper(CDMPDVContext *ctx) : CDMPDVCmdListKHelper(ctx) {
    fnme_acc_ = "/dev/dv_conv";
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdListConvHelper() {
    // Empty by design
  }

  /// @brief Creates object of this type.
  static CDMPDVCmdListDeviceHelper* Create(CDMPDVContext *ctx) {
    return new CDMPDVCmdListConvHelper(ctx);
  }

 private:
  /// @brief Checks provided command for validness.
  virtual int CheckRaw(dmp_dv_cmdraw *cmd,
                       std::vector<std::pair<dmp_dv_buf, uint64_t> >& input_bufs,
                       std::vector<std::pair<dmp_dv_buf, uint64_t> >& output_bufs) {
    switch (cmd->version) {
      case 0:
        return CheckRaw_v0((dmp_dv_cmdraw_conv_v0*)cmd, input_bufs, output_bufs);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Fills command in the format suitable for later execution on the device.
  virtual int FillKCommand(uint8_t *kcmd, dmp_dv_cmdraw *cmd, uint32_t& size) {
    switch (cmd->version) {
      case 0:
        return FillKCommand_v0((dmp_dv_kcmdraw_conv_v0*)kcmd, (dmp_dv_cmdraw_conv_v0*)cmd, size);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Checks command of version 0 for validness.
  int CheckRaw_v0(dmp_dv_cmdraw_conv_v0 *cmd,
                  std::vector<std::pair<dmp_dv_buf, uint64_t> >& input_bufs,
                  std::vector<std::pair<dmp_dv_buf, uint64_t> >& output_bufs) {
    if (cmd->header.size != sizeof(dmp_dv_cmdraw_conv_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
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

    int res = 0;
    struct conv_data_size conv_size;
    init_conv_input_size_v0_4(cmd->w, cmd->h, cmd->z, cmd->c, &conv_size);
    dmp_dv_kcmdraw_conv_v0_run run;
    memset(&run, 0, sizeof(run));
    uint64_t output_size = 0;
    for (int topo = cmd->topo, i = 0; topo; topo >>= 1, ++i) {
      // Check for validness
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
        const int ky = (cmd->run[i].p >> 8) & 0xFF;
        const int max_kernel_size = ctx_->get_max_kernel_size();
        if ((kx < 1) || (kx > max_kernel_size) || (ky < 1) || (ky > max_kernel_size)) {
          SET_ERR("Unsupported convolutional kernel size %dx%d, only sizes from 1 to %d are supported",
                  kx, ky, max_kernel_size);
          return -1;
        }
        const int stride_x = cmd->run[i].conv_stride & 0xFF;
        const int stride_y = (cmd->run[i].conv_stride >> 8) & 0xFF;
        if ((stride_x < 1) || (stride_y < 1)) {
          SET_ERR("Stride of convolution must be greater than 0, got %dx%d", stride_x, stride_y);
          return -1;
        }
      }

      // Calculate input/output/weights dimensions
      const uint64_t input_size = (uint64_t)cmd->w * cmd->h * cmd->c * cmd->z;
      input_bufs.push_back(std::make_pair(cmd->input_buf, input_size));

      run.actfunc = cmd->run[i].actfunc;
      run.actfunc_param = cmd->run[i].actfunc_param;
      run.conv_dilation = cmd->run[i].conv_dilation;
      run.conv_enable = cmd->run[i].conv_enable;
      run.conv_pad = cmd->run[i].conv_pad;
      run.conv_stride = cmd->run[i].conv_stride;
      run.lrn = cmd->run[i].lrn;
      run.m = cmd->run[i].m;
      run.p = cmd->run[i].p;
      run.pool_avg_param = cmd->run[i].pool_avg_param;
      run.pool_enable = cmd->run[i].pool_enable;
      run.pool_pad = cmd->run[i].pool_pad;
      run.pool_size = cmd->run[i].pool_size;
      run.pool_stride = cmd->run[i].pool_stride;
      run.pz = cmd->run[i].pz;
      run.rectifi_en = cmd->run[i].rectifi_en;
      run.weight_fmt = cmd->run[i].weight_fmt;

      uint64_t weights_size = 0;
      get_conv_output_size_v0(&run, &conv_size, &conv_size, (uint32_t*)&weights_size);
      if (weights_size) {
        input_bufs.push_back(std::make_pair(cmd->run[i].weight_buf, weights_size));
      }
      if (topo & 1) {  // output goes to main memory
        if (!conv_size.size) {
          SET_ERR("Invalid argument: cmd->run[%d] produces output with zero size", i);
          return -1;
        }
        output_size += conv_size.size;

        // Next input will be the first
        init_conv_input_size_v0_4(cmd->w, cmd->h, cmd->z, cmd->c, &conv_size);
      }
    }
    if (!output_size) {
      SET_LOGIC_ERR();
      return -1;
    }

    output_bufs.push_back(std::make_pair(cmd->output_buf, output_size));
    if (cmd->eltwise_buf.mem) {
      output_bufs.push_back(std::make_pair(cmd->eltwise_buf, output_size));
    }

    return res;
  }

  /// @brief Fills command of version 0 in the format suitable for later execution on the device.
  int FillKCommand_v0(dmp_dv_kcmdraw_conv_v0 *kcmd, dmp_dv_cmdraw_conv_v0 *cmd, uint32_t& size) {
    if (cmd->header.size != sizeof(dmp_dv_cmdraw_conv_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
      return -1;
    }

    size_t req_size = sizeof(*kcmd) - sizeof(kcmd->run);

    if (size >= req_size) {
      kcmd->size = sizeof(dmp_dv_kcmdraw_conv_v0);
      kcmd->version = 0;

      kcmd->input_buf.fd = CDMPDVMem::get_fd(cmd->input_buf.mem);
      kcmd->input_buf.rsvd = 0;
      kcmd->input_buf.offs = cmd->input_buf.offs;

      kcmd->output_buf.fd = CDMPDVMem::get_fd(cmd->output_buf.mem);
      kcmd->output_buf.rsvd = 0;
      kcmd->output_buf.offs = cmd->output_buf.offs;

      kcmd->eltwise_buf.fd = CDMPDVMem::get_fd(cmd->eltwise_buf.mem);
      kcmd->eltwise_buf.rsvd = 0;
      kcmd->eltwise_buf.offs = cmd->eltwise_buf.offs;

      kcmd->topo = cmd->topo;
      kcmd->w = cmd->w;
      kcmd->h = cmd->h;
      kcmd->z = cmd->z;
      kcmd->c = cmd->c;
      kcmd->input_circular_offset = cmd->input_circular_offset;
      kcmd->output_mode = cmd->output_mode;
    }

    const int n_run = 32;  // TODO: use adaptive count here after debugging full count.
    for (int i_run = 0; i_run < n_run; ++i_run) {
      req_size += sizeof(kcmd->run[i_run]);
      if (size >= req_size) {
        kcmd->run[i_run].weight_buf.fd = CDMPDVMem::get_fd(cmd->run[i_run].weight_buf.mem);
        kcmd->run[i_run].weight_buf.rsvd = 0;
        kcmd->run[i_run].weight_buf.offs = cmd->run[i_run].weight_buf.offs;
        kcmd->run[i_run].conv_pad = cmd->run[i_run].conv_pad;
        kcmd->run[i_run].pool_pad = cmd->run[i_run].pool_pad;
        kcmd->run[i_run].m = cmd->run[i_run].m;
        kcmd->run[i_run].conv_enable = cmd->run[i_run].conv_enable;
        kcmd->run[i_run].p = cmd->run[i_run].p;
        kcmd->run[i_run].pz = cmd->run[i_run].pz;
        kcmd->run[i_run].conv_stride = cmd->run[i_run].conv_stride;
        kcmd->run[i_run].conv_dilation = cmd->run[i_run].conv_dilation;
        kcmd->run[i_run].weight_fmt = cmd->run[i_run].weight_fmt;
        kcmd->run[i_run].pool_enable = cmd->run[i_run].pool_enable;
        kcmd->run[i_run].pool_avg_param = cmd->run[i_run].pool_avg_param;
        kcmd->run[i_run].pool_size = cmd->run[i_run].pool_size;
        kcmd->run[i_run].pool_stride = cmd->run[i_run].pool_stride;
        kcmd->run[i_run].actfunc = cmd->run[i_run].actfunc;
        kcmd->run[i_run].actfunc_param = cmd->run[i_run].actfunc_param;
        kcmd->run[i_run].rectifi_en = cmd->run[i_run].rectifi_en;
        kcmd->run[i_run].lrn = cmd->run[i_run].lrn;
        kcmd->run[i_run].rsvd = cmd->run[i_run].rsvd;
      }
    }

    size = req_size;
    return 0;
  }
};


/// @brief Helper object work working with command list for FC accelerator.
class CDMPDVCmdListFCHelper : public CDMPDVCmdListKHelper {
 public:
  /// @brief Constructor.
  CDMPDVCmdListFCHelper(CDMPDVContext *ctx) : CDMPDVCmdListKHelper(ctx) {
    fnme_acc_ = "/dev/dv_fc";
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdListFCHelper() {
    // Empty by design
  }

  /// @brief Creates object of this type.
  static CDMPDVCmdListDeviceHelper* Create(CDMPDVContext *ctx) {
    return new CDMPDVCmdListFCHelper(ctx);
  }

 private:
  /// @brief Checks provided command for validness.
  virtual int CheckRaw(dmp_dv_cmdraw *cmd,
                       std::vector<std::pair<dmp_dv_buf, uint64_t> >& input_bufs,
                       std::vector<std::pair<dmp_dv_buf, uint64_t> >& output_bufs) {
    switch (cmd->version) {
      case 0:
        return CheckRaw_v0((dmp_dv_cmdraw_fc_v0*)cmd, input_bufs, output_bufs);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Fills command in the format suitable for later execution on the device.
  virtual int FillKCommand(uint8_t *kcmd, dmp_dv_cmdraw *cmd, uint32_t& size) {
    switch (cmd->version) {
      case 0:
        return FillKCommand_v0((dmp_dv_kcmdraw_fc_v0*)kcmd, (dmp_dv_cmdraw_fc_v0*)cmd, size);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Checks command of version 0 for validness.
  int CheckRaw_v0(dmp_dv_cmdraw_fc_v0 *cmd,
                  std::vector<std::pair<dmp_dv_buf, uint64_t> >& input_bufs,
                  std::vector<std::pair<dmp_dv_buf, uint64_t> >& output_bufs) {
    if (cmd->header.size != sizeof(dmp_dv_cmdraw_fc_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
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

    if (!cmd->weight_buf.mem) {
      SET_ERR("Invalid argument: cmd->weight_buf.mem is NULL");
      return -1;
    }

    input_bufs.push_back(std::make_pair(cmd->input_buf, cmd->input_size * 2));

    size_t weights_size = 0;
    uint16_t quant_map[256];
    int res = dmp_dv_pack_fc_weights(
        cmd->input_size, 1, 1, cmd->output_size,
        cmd->weight_fmt == 1 ? quant_map : NULL,
        NULL, NULL, NULL, &weights_size);
    if (res) {
      return res;
    }
    input_bufs.push_back(std::make_pair(cmd->weight_buf, weights_size));

    output_bufs.push_back(std::make_pair(cmd->output_buf, cmd->output_size * 2));

    return 0;
  }

  /// @brief Fills command of version 0 in the format suitable for later execution on the device.
  int FillKCommand_v0(dmp_dv_kcmdraw_fc_v0 *kcmd, dmp_dv_cmdraw_fc_v0 *cmd, uint32_t& size) {
    if (cmd->header.size != sizeof(dmp_dv_cmdraw_fc_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
      return -1;
    }

    size_t req_size = sizeof(*kcmd);

    if (size >= req_size) {
      kcmd->size = sizeof(dmp_dv_kcmdraw_fc_v0);
      kcmd->version = 0;

      kcmd->input_buf.fd = CDMPDVMem::get_fd(cmd->input_buf.mem);
      kcmd->input_buf.rsvd = 0;
      kcmd->input_buf.offs = cmd->input_buf.offs;

      kcmd->output_buf.fd = CDMPDVMem::get_fd(cmd->output_buf.mem);
      kcmd->output_buf.rsvd = 0;
      kcmd->output_buf.offs = cmd->output_buf.offs;

      kcmd->weight_buf.fd = CDMPDVMem::get_fd(cmd->weight_buf.mem);
      kcmd->weight_buf.rsvd = 0;
      kcmd->weight_buf.offs = cmd->weight_buf.offs;

      kcmd->input_size = cmd->input_size;
      kcmd->output_size = cmd->output_size;
      kcmd->weight_fmt = cmd->weight_fmt;
      kcmd->actfunc = cmd->actfunc;
      kcmd->actfunc_param = cmd->actfunc_param;
      kcmd->rsvd[0] = 0;
      kcmd->rsvd[1] = 0;
      kcmd->rsvd[2] = 0;
    }

    size = req_size;
    return 0;
  }
};


/// @brief Implementation of dmp_dv_cmdlist.
class CDMPDVCmdList : public CDMPDVBase {
 public:
  /// @brief Constructor.
  CDMPDVCmdList() : CDMPDVBase() {
    ctx_ = NULL;
    commited_ = false;
    memset(device_helpers_, 0, sizeof(device_helpers_));
    single_device_ = NULL;
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdList() {
    ReleaseResources();
  }

  /// @brief Initializes command list by assigning context to it.
  bool Initialize(CDMPDVContext *ctx) {
    Cleanup();
    if (!ctx) {
      SET_ERR("Invalid argument: ctx is NULL");
      return false;
    }

    ctx->Retain();
    ctx_ = ctx;

    return true;
  }

  /// @brief Adds raw structure describing the command.
  int AddRaw(dmp_dv_cmdraw *cmd) {
    if (commited_) {
      SET_ERR("Command list is already in commited state");
      return -1;
    }
    if (cmd->size < 8) {
      SET_ERR("Invalid argument: cmd->size %d is too small", (int)cmd->size);
      return EINVAL;
    }
    if (cmd->device_type >= DMP_DV_DEV_COUNT) {
      SET_ERR("Invalid argument: device_type is out of bounds: got %d while bounds are [%d, %d]",
              (int)cmd->device_type, 0, DMP_DV_DEV_COUNT - 1);
      return EINVAL;
    }
    int res;
    if (!device_helpers_[cmd->device_type]) {
      CDMPDVCmdListDeviceHelper *helper = NULL;
      res = CDMPDVCmdListDeviceHelper::Instantiate(ctx_, cmd->device_type, &helper);
      if (res) {
        return res;
      }
      if (!helper) {
        SET_LOGIC_ERR();
        return -1;
      }
      device_helpers_[cmd->device_type] = helper;
    }

    Command command;
    command.cmd.resize(cmd->size);
    memcpy(command.cmd.data(), cmd, cmd->size);
    command.device_helper = device_helpers_[cmd->device_type];
    res = device_helpers_[cmd->device_type]->CheckRaw(cmd, command.input_bufs, command.output_bufs);
    if (res) {
      return res;
    }

    // Validate buffers
    for (auto it = command.input_bufs.begin(); it != command.input_bufs.end(); ++it) {
      res = ValidateBuffer(it->first, it->second);
      if (res) {
        return res;
      }
    }
    for (auto it = command.output_bufs.begin(); it != command.output_bufs.end(); ++it) {
      res = ValidateBuffer(it->first, it->second);
      if (res) {
        return res;
      }
    }

    // Increase reference counters
    for (auto it = command.input_bufs.begin(); it != command.input_bufs.end(); ++it) {
      dmp_dv_mem_retain(it->first.mem);
    }
    for (auto it = command.output_bufs.begin(); it != command.output_bufs.end(); ++it) {
      dmp_dv_mem_retain(it->first.mem);
    }

    // Add command to the command list
    commands_.push_back(std::move(command));

    return 0;
  }

  /// @brief Commits command list, filling hardware-specific structures and passing them to kernel module.
  int Commit() {
    if (commited_) {
      SET_ERR("Command list is already in commited state");
      return EALREADY;
    }
    int n_devs = 0;
    for (int i = 0; i < DMP_DV_DEV_COUNT; ++i) {
      n_devs += device_helpers_[i] ? 1 : 0;
    }
    if (n_devs < 1) {
      SET_ERR("Command list is empty");
      return ENODATA;
    }
    if (n_devs == 1) {
      for (int i = 0; i < DMP_DV_DEV_COUNT; ++i) {
        if (device_helpers_[i]) {
          single_device_ = device_helpers_[i];
          return CommitSingleDevice();
        }
      }
      SET_LOGIC_ERR();
      return -1;
    }

    SET_ERR("Having different device types in the single command list is not yet implemented");
    return -1;
  }

  /// @brief Schedules commited command list for execution.
  /// @return >= 0 - execution id on sucess, < 0 on error.
  int64_t Exec() {
    if (single_device_) {
      return single_device_->Exec();
    }
    SET_ERR("Having different device types in the single command list is not yet implemented");
    return -1;
  }

  /// @brief Waits for the specific execution id to be completed.
  /// @return 0 on success, non-zero on error.
  int Wait(int64_t exec_id) {
    if (single_device_) {
      return single_device_->Wait(exec_id);
    }
    SET_ERR("Having different device types in the single command list is not yet implemented");
    return -1;
  }

 protected:
  /// @brief Releases held resources.
  virtual void Cleanup() {
    ReleaseResources();
  }

 private:
  /// @brief Releases held resources.
  void ReleaseResources() {
    // Decrease reference counters on used memory pointers
    for (auto cmd_it = commands_.rbegin(); cmd_it != commands_.rend(); ++cmd_it) {
      for (auto it = cmd_it->output_bufs.rbegin(); it != cmd_it->output_bufs.rend(); ++it) {
        dmp_dv_mem_release(it->first.mem);
      }
      for (auto it = cmd_it->input_bufs.rbegin(); it != cmd_it->input_bufs.rend(); ++it) {
        dmp_dv_mem_release(it->first.mem);
      }
    }
    commands_.clear();

    // Release device helpers
    for (int i = DMP_DV_DEV_COUNT - 1; i >= 0; --i) {
      if (device_helpers_[i]) {
        device_helpers_[i]->Release();
        device_helpers_[i] = NULL;
      }
    }

    // Release the context
    if (ctx_) {
      ctx_->Release();
      ctx_ = NULL;
    }

    // Reset other vars
    commited_ = false;
  }

  /// @brief Validates buffer.
  int ValidateBuffer(dmp_dv_buf& buf, uint64_t size) {
    if (!size) {
      SET_LOGIC_ERR();
      return -1;
    }
    if (!buf.mem) {
      SET_ERR("Memory handle in buffer is NULL");
      return EINVAL;
    }
    uint64_t n = dmp_dv_mem_get_size(buf.mem);
    if ((buf.offs >= n) || (n - buf.offs < size)) {
      SET_ERR("Insufficient space detected in the provided buffer: "
              "buffer size is %llu, offset is %llu, required bytes %llu",
              (unsigned long long)n, (unsigned long long)buf.offs, (unsigned long long)size);
      return EINVAL;
    }
    return 0;
  }

  /// @brief Commits command list in case of single device.
  int CommitSingleDevice() {  // TODO: when non-raw commands will be implemented, convert all commands to raw format before this function call.
    int res;
    size_t total_size = 0;
    for (auto cmd_it = commands_.begin(); cmd_it != commands_.end(); ++cmd_it) {
      uint32_t size = 0;
      res = cmd_it->device_helper->FillKCommand(NULL, (dmp_dv_cmdraw*)cmd_it->cmd.data(), size);
      if (res) {
        return res;
      }
      total_size += size;
    }

    // Allocate temporary buffer for the kernel command
    uint8_t *kcommand = (uint8_t*)malloc(total_size);
    if (!kcommand) {
      SET_ERR("Could not allocate %zu bytes of memory", total_size);
      return ENOMEM;
    }

    // Fill buffer for the kernel command
    size_t offs = 0;
    for (auto cmd_it = commands_.begin(); cmd_it != commands_.end(); ++cmd_it) {
      uint32_t size = total_size - offs;
      res = cmd_it->device_helper->FillKCommand(
          kcommand + offs, (dmp_dv_cmdraw*)cmd_it->cmd.data(), size);
      if (res) {
        return res;
      }
      offs += size;
      if (offs > total_size) {
        SET_LOGIC_ERR();
        free(kcommand);
        return -1;
      }
    }

    // Pass command to kernel module
    res = single_device_->KCommit(kcommand, total_size, commands_.size());

    // Free temporary buffer
    free(kcommand);

    if (!res) {
      commited_ = true;
    }
    return res;
  }

  /// @brief Command in command list.
  struct Command {
    std::vector<uint64_t> cmd;  // raw command
    CDMPDVCmdListDeviceHelper *device_helper;  // pointer to device helper for convenience
    std::vector<std::pair<dmp_dv_buf, uint64_t> > input_bufs, output_bufs;  // buffers used during this command
  };

  /// @brief Reference to device context.
  CDMPDVContext *ctx_;

  /// @brief If Commit() was called.
  bool commited_;

  /// @brief Helpers for working with commands for specific device types (CONV or FC).
  CDMPDVCmdListDeviceHelper *device_helpers_[DMP_DV_DEV_COUNT];

  /// @brief Commands.
  std::vector<Command> commands_;

  /// @brief When the command list comntains the single device, this variable is assigned to it.
  CDMPDVCmdListDeviceHelper *single_device_;
};
