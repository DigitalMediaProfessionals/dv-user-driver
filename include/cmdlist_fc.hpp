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
class CDMPDVCmdListFC : public CDMPDVCmdList {
 public:
  /// @brief Constructor.
  CDMPDVCmdListFC() : CDMPDVCmdList() {
    // Empty by design
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdListFC() {
    ReleaseResources();
  }

  /// @brief Adds raw structure describing the command for convolutional module.
  virtual int AddRaw(dmp_dv_cmdraw *cmd) {
    int res = CheckAddRaw(cmd, DMP_DV_FC);
    if (res) {
      return res;
    }
    switch (cmd->version) {
      case 0:
        return AddRaw_v0((dmp_dv_cmdraw_fc_v0*)cmd);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_ERR("Control should not reach line %d of file %s", __LINE__, __FILE__);
    return -1;
  }

  /// @brief Commits command list for convolutional accelerator.
  virtual int Commit() {
    SET_ERR("Not implemented at line %d of file %s", __LINE__, __FILE__);  // TODO: implement.
    return -1;
  }

 protected:
  /// @brief Releases held resources.
  virtual void Cleanup() {
    ReleaseResources();
    CDMPDVCmdList::Cleanup();
  }

  /// @brief Opens convolutional accelerator.
  virtual bool OpenDevice() {
    return OpenDeviceAsFile("/dev/dv_fc");
  }

  /// @brief Adds raw structure describing the fully connected layer in version 0 format.
  int AddRaw_v0(dmp_dv_cmdraw_fc_v0 *cmd) {
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

    // TODO: add more checks.

    // Increase reference counters on the provided memory pointers
    dmp_dv_mem_retain(cmd->input_buf.mem);
    dmp_dv_mem_retain(cmd->output_buf.mem);
    dmp_dv_mem_retain(cmd->weight_buf.mem);

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
      dmp_dv_cmdraw_fc_v0& cmd = commands_[i];
      dmp_dv_mem_release(cmd.weight_buf.mem);
      dmp_dv_mem_release(cmd.output_buf.mem);
      dmp_dv_mem_release(cmd.input_buf.mem);
    }
    commands_.clear();
  }

 private:
  /// @brief Added commands.
  std::vector<dmp_dv_cmdraw_fc_v0> commands_;
};
