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


/// @brief Helper object work working with command list specific to the device type (CONV or FC).
class CDMPDVCmdListDeviceHelper : public CDMPDVBase {
 public:
  /// @brief Constructor.
  CDMPDVCmdListDeviceHelper(CDMPDVContext *ctx) : CDMPDVBase() {
    ctx->Retain();
    ctx_ = ctx;
    commited_ = false;
    last_exec_time_ = 0;
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
  virtual int CheckRaw(struct dmp_dv_cmdraw *cmd,
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) = 0;

  /// @brief Fills command in the format suitable for later execution on the device.
  /// @param kcmd Buffer to hold kernel command, can be NULL to get only size.
  /// @param cmd Command to execute (user-space format).
  /// @param size On enter must contain available space in bytes in the kcmd buffer, on exit will contain used or required space.
  /// @return 0 on success, non-zero on error.
  virtual int FillKCommand(uint8_t *kcmd, struct dmp_dv_cmdraw *cmd, uint32_t& size) = 0;

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

  /// @brief Get last execution time.
  /// @return last execution in microseconds(us).
  virtual uint64_t GetLastExecTime() = 0;

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

  /// @brief Record last execution time.
  uint64_t last_exec_time_;

 private:
  /// @brief If the list is in commited state.
  bool commited_;

  /// @brief Creators for the specific device types.
  static CDMPDVCmdListDeviceHelper* (*creators_[DMP_DV_DEV_COUNT])(CDMPDVContext *ctx);
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
      SET_IOCTL_ERR(res, fnme_acc_, "DMP_DV_IOC_APPEND_CMD");
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
      SET_IOCTL_ERR(res, fnme_acc_, "DMP_DV_IOC_RUN");
      return -1;
    }
    if (exec_id < 0) {
      SET_ERR("ioctl(%s) on %s succeded returning invalid exec_id=%lld",
              "DMP_DV_IOC_RUN", fnme_acc_, (long long)exec_id);
      return -1;
    }
    return exec_id;
  }

  /// @brief Waits for scheduled command to be completed.
  virtual int Wait(int64_t exec_id) {
    if (exec_id < 0) {
      SET_ERR("Invalid argument: exec_id = %lld", (long long)exec_id);
      return EINVAL;
    }
    dmp_dv_kwait dv_wait;
    dv_wait.cmd_id = exec_id;
    for (;;) {
      int res = ioctl(fd_acc_, DMP_DV_IOC_WAIT, &dv_wait);
      if (!res) {
        last_exec_time_ = dv_wait.cmd_exec_time;
        break;
      }
      switch (errno) {
        case EBUSY:        // timeout of 2 seconds reached
        case ERESTARTSYS:  // signal has interrupted the wait
          continue;  // repeat ioctl

        default:
          SET_IOCTL_ERR(res, fnme_acc_, "DMP_DV_IOC_WAIT");
          return res;
      }
    }
    return 0;
  }

  virtual uint64_t GetLastExecTime() {
    return last_exec_time_;
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


/// @brief Command in command list.
struct DMPDVCommand {
  std::vector<uint8_t> cmd;  // raw command
  CDMPDVCmdListDeviceHelper *device_helper;  // pointer to device helper for convenience
  std::vector<std::pair<struct dmp_dv_buf, uint64_t> > input_bufs, output_bufs;  // buffers used during this command
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
  int AddRaw(struct dmp_dv_cmdraw *cmd) {
    if (commited_) {
      SET_ERR("Command list is already in commited state");
      return -1;
    }
    if (!cmd) {
      SET_ERR("Invalid argument: cmd is NULL");
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
    int device_type = cmd->device_type;
    if ((device_type == DMP_DV_DEV_FC) && (!ctx_->DeviceExists(DMP_DV_DEV_FC))) {
      device_type = DMP_DV_DEV_CONV;  // switch to CONV to support legacy FC configuration
    }
    int res;
    if (!device_helpers_[device_type]) {
      CDMPDVCmdListDeviceHelper *helper = NULL;
      res = CDMPDVCmdListDeviceHelper::Instantiate(ctx_, device_type, &helper);
      if (res) {
        return res;
      }
      if (!helper) {
        SET_LOGIC_ERR();
        return -1;
      }
      device_helpers_[device_type] = helper;
    }

    DMPDVCommand command;
    command.cmd.resize(cmd->size);
    memcpy(command.cmd.data(), cmd, cmd->size);
    command.device_helper = device_helpers_[device_type];
    res = device_helpers_[device_type]->CheckRaw(cmd, command.input_bufs, command.output_bufs);
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
    if (!commited_) {
      SET_ERR("Command list is not in commited state");
      return -EINVAL;
    }
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

  uint64_t GetLastExecTime() {
    if (single_device_) {
      return single_device_->GetLastExecTime();
    }
    SET_ERR("Having different device types in the single command list is not yet implemented");
    return 0;
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
  int ValidateBuffer(struct dmp_dv_buf& buf, uint64_t size) {
    if (!size) {
      SET_LOGIC_ERR();
      return -1;
    }
    if (!buf.mem) {
      SET_ERR("Memory handle in buffer is NULL");
      return EINVAL;
    }
    if (buf.offs & 15) {
      SET_ERR("Offset in buffer must be 16-bytes aligned, got %llu",
              (unsigned long long)buf.offs);
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
  int CommitSingleDevice() {
    if (!commands_.size()) {
      SET_ERR("Command list is empty");
      return EINVAL;
    }
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
    if (!total_size) {
      SET_ERR("Calculated memory size for command list raw representation is 0");
      return EINVAL;
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

  /// @brief Reference to device context.
  CDMPDVContext *ctx_;

  /// @brief If Commit() was called.
  bool commited_;

  /// @brief Helpers for working with commands for specific device types (CONV or FC).
  CDMPDVCmdListDeviceHelper *device_helpers_[DMP_DV_DEV_COUNT];

  /// @brief Commands.
  std::vector<DMPDVCommand> commands_;

  /// @brief When the command list comntains the single device, this variable is assigned to it.
  CDMPDVCmdListDeviceHelper *single_device_;
};
