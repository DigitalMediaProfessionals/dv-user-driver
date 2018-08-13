/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
*/
/*
 * @brief Helper object work working with command list for FC accelerator implementation.
 */
#pragma once

#include "cmdlist.hpp"


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

    if ((int)cmd->input_size > ctx_->get_max_fc_vector_size()) {
      SET_ERR("Unsupported input vector size %d, only sizes from 1 to %d are supported",
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
