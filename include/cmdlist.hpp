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

#include "mem.hpp"


/// @brief Implementation of dmp_dv_cmdlist.
class CDMPDVCmdList : public CDMPDVBase {
 public:
  /// @brief Constructor.
  CDMPDVCmdList() : CDMPDVBase() {
    ctx_ = NULL;
    fd_acc_ = -1;
    commited_ = false;
    fnme_acc_ = "";
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

    if (fd_acc_ != -1) {
      SET_ERR("Implementation error (inconsistency detected: device file should not be open at this point)");
      return -1;
    }

    if (!OpenDevice()) {
      return false;
    }

    ctx->Retain();
    ctx_ = ctx;

    return true;
  }

  /// @brief Adds raw structure describing the command.
  virtual int AddRaw(dmp_dv_cmdraw *cmd) = 0;

  /// @brief Commits command list, filling hardware-specific structures and passing them to kernel module.
  virtual int Commit() = 0;

  /// @brief Executes command list.
  /// @return Id of this execution >= 0 on success, < 0 on error.
  /// @details The result is undefined if command list was not in commited state.
  int64_t Exec() {
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

  /// @brief Waits for the specific execution id to be completed.
  /// @return 0 on success, non-zero on error.
  int Wait(int64_t exec_id) {
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

 protected:
  /// @brief Releases held resources.
  virtual void Cleanup() {
    ReleaseResources();
  }

  /// @brief Opens device file and assigns fnme_acc_ and fd_acc_ accordingly.
  virtual bool OpenDevice() = 0;

  /// @brief Helper functions for opening device as file.
  bool OpenDeviceAsFile(const char *fnme) {
    fnme_acc_ = fnme;
    fd_acc_ = open(fnme_acc_, O_RDONLY | O_CLOEXEC);
    if (fd_acc_ == -1) {
      SET_ERR("open() failed for %s: %s", fnme_acc_, strerror(errno));
      return false;
    }
    return true;
  }

  /// @brief Helper function doing some basic checks on the added command.
  int CheckAddRaw(dmp_dv_cmdraw *cmd, dmp_dv_device_type device_type) {
    if (commited_) {
      SET_ERR("Command list is already in commited state");
      return -1;
    }
    if (cmd->size < 8) {
      SET_ERR("Invalid argument: cmd->size %d is too small", (int)cmd->size);
      return EINVAL;
    }
    if (cmd->device_type != device_type) {
      SET_ERR("Invalid argument: cmd->device_type: got %d while expecting %d",
              (int)cmd->device_type, (int)device_type);
      return EINVAL;
    }
    return 0;
  }

  /// @brief Helper function to pass commands to kernel module.
  int IssueCommitIOCTL(int cmd_num, void *cmd_pointer) {
    dmp_dv_kcmd dv_cmd;
    dv_cmd.cmd_num = cmd_num;
    dv_cmd.cmd_pointer = (__u64)cmd_pointer;

    int res = ioctl(fd_acc_, DMP_DV_IOC_APPEND_CMD, &dv_cmd);
    if (res < 0) {
      SET_IOCTL_ERR(fnme_acc_, "DMP_DV_IOC_APPEND_CMD");
      res = -1;
    }
    else {
      res = 0;
      commited_ = true;
    }
    return res;
  }

 protected:
  /// @brief Reference to device context.
  CDMPDVContext *ctx_;

  /// @brief File handle for CONV or FC accelerator.
  int fd_acc_;

  /// @brief Filename for CONV or FC accelerator.
  const char *fnme_acc_;

  /// @brief If Commit() was called.
  bool commited_;

 private:
  /// @brief Releases held resources.
  void ReleaseResources() {
    // Close opened files
    if (fd_acc_ != -1) {
      close(fd_acc_);
      fd_acc_ = -1;
    }

    // Release the context
    if (ctx_) {
      ctx_->Release();
      ctx_ = NULL;
    }

    // Reset other vars
    commited_ = false;
    fnme_acc_ = "";
  }
};
