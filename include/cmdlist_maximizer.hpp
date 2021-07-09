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
/// @brief Helper object work working with command list for MAXIMIZER implementation.
#pragma once

#include "cmdlist.hpp"

#ifdef __x86_64__
#include "half.h"
typedef half_float::half __fp16;
#endif

/// @brief Helper object work working with command list for MAXIMIZER
class CDMPDVCmdListMaximizerHelper : public CDMPDVCmdListKHelper {
  public:
    /// @brief Constructor.
    CDMPDVCmdListMaximizerHelper(CDMPDVContext *ctx) : CDMPDVCmdListKHelper(ctx) {
      fnme_acc_ = "/dev/dv_maximizer";
    }

    /// @brief Destructor.
    virtual ~CDMPDVCmdListMaximizerHelper() {
      // Empty by design
    }

    /// @brief Creates object of this type.
    static CDMPDVCmdListDeviceHelper* Create(CDMPDVContext *ctx) {
      return new CDMPDVCmdListMaximizerHelper(ctx);
    }

  private:
    /// @brief Checks provided command for validness.
    virtual int CheckRaw(dmp_dv_cmdraw *cmd,
                         std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                         std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
      switch (cmd->version) {
        case 0:
          return CheckRaw_v0((dmp_dv_cmdraw_maximizer_v0*)cmd, input_bufs, output_bufs);

        default:
          SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
          return ENOTSUP;
      }
      SET_LOGIC_ERR();
      return -1;
    }

    /// @brief Checks provided command for validness.
    int CheckRaw_v0(struct dmp_dv_cmdraw_maximizer_v0 *cmd,
        std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
        std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
      if (cmd->header.size != sizeof(*cmd)) {
        SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
                (int)cmd->header.size, (int)cmd->header.version);
        return -1;
      }

      if (cmd->width == 0) {
        SET_ERR("Invalid argument: cmd->width is 0");
        return -1;
      }
      if (cmd->height == 0) {
        SET_ERR("Invalid argument: cmd->height is 0");
        return -1;
      }
      uint32_t npixel = static_cast<uint32_t>(cmd->width) * cmd->height;
      if (npixel & 0xff000000) {
        SET_ERR("Invalid argument: the number of pixel is %u but must be smaller than %u", npixel, 1 << 24);
        return -1;
      }
      if (cmd->nclass < 2) {
        SET_ERR("Invalid argument: cmd->nclass is %d but must be larger than 1", cmd->nclass);
        return -1;
      }

      // register buffers
      input_bufs.push_back(std::make_pair(cmd->input_buf, cmd->width * cmd->height * cmd->nclass * sizeof(__fp16)));
      output_bufs.push_back(std::make_pair(cmd->output_buf, cmd->width * cmd->height));

      return 0;
    }

    /// @brief Fills command in the format suitable for later execution on the device.
    virtual int FillKCommand(uint8_t *kcmd, dmp_dv_cmdraw *cmd, uint32_t& size) {
      switch (cmd->version) {
        case 0:
          return FillKCommand_v0((dmp_dv_kcmdraw_maximizer_v0*)kcmd,
                                 (dmp_dv_cmdraw_maximizer_v0*)cmd, size);

        default:
          SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
          return ENOTSUP;
      }
      SET_LOGIC_ERR();
      return -1;
    }

    /// @brief Fills command of version 0 in the format suitable for later execution on the device.
    int FillKCommand_v0(struct dmp_dv_kcmdraw_maximizer_v0 *kcmd,
                        struct dmp_dv_cmdraw_maximizer_v0 *cmd, uint32_t& size) {
      if (cmd->header.size != sizeof(*cmd)) {
        SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
                (int)cmd->header.size, (int)cmd->header.version);
        return -1;
      }

      size_t req_size = sizeof(*kcmd);
      if (size >= req_size) {
        kcmd->header.version  = 0;
        kcmd->header.size     = sizeof(*kcmd);
        kcmd->input_buf.fd    = CDMPDVMem::get_fd(cmd->input_buf.mem);
        kcmd->input_buf.rsvd  = 0;
        kcmd->input_buf.offs  = cmd->input_buf.offs;
        kcmd->output_buf.fd   = CDMPDVMem::get_fd(cmd->output_buf.mem);
        kcmd->output_buf.rsvd = 0;
        kcmd->output_buf.offs = cmd->output_buf.offs;

        kcmd->width  = cmd->width;
        kcmd->height = cmd->height;
        kcmd->nclass = cmd->nclass;
      }
      size = req_size;
      return 0;
    }
};
