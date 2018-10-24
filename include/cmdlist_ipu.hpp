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
 * @brief Helper object work working with command list for IPU implementation.
 */
#pragma once

#include "cmdlist.hpp"


/// @brief Helper object work working with command list for IPU
class CDMPDVCmdListIPUHelper : public CDMPDVCmdListKHelper {
  /// @brief Constructor.
  CDMPDVCmdListIPUHelper(CDMPDVContext *ctx) : CDMPDVCmdListKHelper(ctx) {
    fnme_acc_ = "/dev/dv_ipu";
  }
  
  /// @brief Destructor.
  virtual ~CDMPDVCmdListIPUHelper() {
    // Empty by design
  }

  /// @brief Creates object of this type.
  static CDMPDVCmdListDeviceHelper* Create(CDMPDVContext *ctx) {
    return new CDMPDVCmdListIPUHelper(ctx);
  }

 private:
  /// @brief Checks provided command for validness.
  virtual int CheckRaw(dmp_dv_cmdraw *cmd,
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
    switch (cmd->version) {
      case 0:
        return CheckRaw_v0((dmp_dv_cmdraw_ipu_v0*)cmd, input_bufs, output_bufs);

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
        return FillKCommand_v0((dmp_dv_kcmdraw_ipu_v0*)kcmd, (dmp_dv_cmdraw_ipu_v0*)cmd, size);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Checks command of version 0 for validness.
  int CheckRaw_v0(struct dmp_dv_cmdraw_ipu_v0 *cmd,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
	  //TODO: implement
	  assert(0);
  }

  /// @brief Fills command of version 0 in the format suitable for later execution on the device.
  int FillKCommand_v0(struct dmp_dv_kcmdraw_ipu_v0 *kcmd, struct dmp_dv_cmdraw_ipu_v0 *cmd, uint32_t& size) {
	  //TODO: implement
	  assert(0);
  }
}
