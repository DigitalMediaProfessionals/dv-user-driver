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
/// @file
/// @brief Helper object work working with command list for FC accelerator implementation.
/// @details FC accelerator is deprecated and this file exists for backward compatibility.
#pragma once

#include "cmdlist.hpp"


/// @brief Helper object work working with command list for FC accelerator.
/// @details FC accelerator is deprecated and this class exists for backward compatibility.
class CDMPDVCmdListFCHelper : public CDMPDVCmdListKHelper {
 public:
  /// @brief Constructor.
  CDMPDVCmdListFCHelper(CDMPDVContext *ctx) : CDMPDVCmdListKHelper(ctx) {
    fnme_acc_ = DMP_DV_DEV_PATH_FC;
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
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
    switch (cmd->version) {
      case 0:
        return CheckRaw_v0((struct dmp_dv_cmdraw_fc_v0*)cmd, input_bufs, output_bufs);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Fills command in the format suitable for later execution on the device.
  virtual int FillKCommand(uint8_t *kcmd, struct dmp_dv_cmdraw *cmd, uint32_t& size) {
    switch (cmd->version) {
      case 0:
        return FillKCommand_v0((struct dmp_dv_kcmdraw_fc_v0*)kcmd, (struct dmp_dv_cmdraw_fc_v0*)cmd, size);

      default:
        SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Checks command of version 0 for validness.
  int CheckRaw_v0(struct dmp_dv_cmdraw_fc_v0 *cmd,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
    if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_fc_v0)) {
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

    if ((!cmd->input_size) || ((int)cmd->input_size > ctx_->get_max_fc_vector_size())) {
      SET_ERR("Unsupported input vector size %d, only sizes up to %d are supported",
              (int)cmd->input_size, ctx_->get_max_fc_vector_size());
      return -1;
    }

    if ((!cmd->output_size) || ((int)cmd->output_size > ctx_->get_max_fc_vector_size())) {
      SET_ERR("Unsupported output vector size %d, only sizes from 1 to %d are supported",
              (int)cmd->input_size, ctx_->get_max_fc_vector_size());
      return -1;
    }

    input_bufs.push_back(std::make_pair(cmd->input_buf, cmd->input_size * 2));

    size_t weights_size = 0;
    uint16_t quant_map[256];
    int res = dmp_dv_pack_fc_weights(
        cmd->input_size, 1, 1,
        cmd->output_size, 1, 1,
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
  int FillKCommand_v0(struct dmp_dv_kcmdraw_fc_v0 *kcmd, struct dmp_dv_cmdraw_fc_v0 *cmd, uint32_t& size) {
    if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_fc_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
      return -1;
    }

    uint32_t req_size = sizeof(*kcmd);

    if (size >= req_size) {
      kcmd->header.size = sizeof(struct dmp_dv_kcmdraw_fc_v0);
      kcmd->header.version = 0;

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
