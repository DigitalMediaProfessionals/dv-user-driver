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

#include <assert.h>
#include "cmdlist.hpp"

/// @brief Helper object work working with command list for IPU
class CDMPDVCmdListIPUHelper : public CDMPDVCmdListKHelper {
  public:
    static const int32_t  STRIDE_WR_MIN;
    static const int32_t  STRIDE_WR_MAX;
    static const int32_t  STRIDE_RD_MIN;
    static const int32_t  STRIDE_RD_MAX;
    static const uint32_t RECT_WIDTH_MAX;
    static const uint32_t RECT_HEIGHT_MAX;
    static const uint32_t TEX_WIDTH_MAX;
    static const uint32_t TEX_HEIGHT_MAX;

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
      if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_ipu_v0)) {
        SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
            (int)cmd->header.size, (int)cmd->header.version);
        return -1;
      }

      // check wr
      if (!cmd->wr.mem) {
        SET_ERR("Invalid argument: cmd->wr.mem is NULL");
        return -1;
      }
      if (cmd->fmt_wr != DMP_DV_RGBA8888 &&
          cmd->fmt_wr != DMP_DV_RGB888 &&
          cmd->fmt_wr != DMP_DV_RGBFP16) {
        SET_ERR("Invalid argument: cmd->fmt_wr must be DMP_DV_RGBA8888, DMP_DV_RGB888 or DMP_DV_RGBFP16");
        return -1;
      }
      if (cmd->stride_wr < STRIDE_WR_MIN) {
        SET_ERR("Invalid argument: cmd->stride_wr must be higher than %d", STRIDE_WR_MIN + 1);
        return -1;
      }
      if (STRIDE_WR_MAX < cmd->stride_wr) {
        SET_ERR("Invalid argument: cmd->stride_wr must be smaller than %d", STRIDE_WR_MIN + 1);
        return -1;
      }
      if (cmd->rect_width == 0) {
        SET_ERR("Invalid argument: cmd->rect_width is 0");
        return -1;
      }
      if (cmd->rect_height == 0) {
        SET_ERR("Invalid argument: cmd->rect_height is 0");
        return -1;
      }
      if (cmd->rect_width >= RECT_WIDTH_MAX) {
        SET_ERR("Invalid argument: cmd->rect_width is higher than %u", RECT_WIDTH_MAX);
        return -1;
      }
      if (cmd->rect_height >= RECT_HEIGHT_MAX) {
        SET_ERR("Invalid argument: cmd->rect_height is higher than %u", RECT_HEIGHT_MAX);
        return -1;
      }

      if (!cmd->use_tex && !cmd->use_rd) {
        SET_ERR("Invalid argument: at least one of cmd->use_tex and cmd->use_rd must be non-zero");
        return -1;
      }
      // check tex
      if (cmd->use_tex) {
        if (!cmd->tex.mem) {
          SET_ERR("Invalid argument: cmd->tex.mem is NULL");
          return -1;
        }
        if (cmd->tex_width == 0) {
          SET_ERR("Invalid argument: cmd->tex_width is 0");
          return -1;
        }
        if (cmd->tex_height == 0) {
          SET_ERR("Invalid argument: cmd->tex_height is 0");
          return -1;
        }
        if (cmd->tex_width > TEX_WIDTH_MAX) {
          SET_ERR("Invalid argument: cmd->tex_width is higher than %u", TEX_WIDTH_MAX);
          return -1;
        }
        if (cmd->tex_height > TEX_HEIGHT_MAX) {
          SET_ERR("Invalid argument: cmd->tex_height is higher than %u", TEX_HEIGHT_MAX);
          return -1;
        }

        // format related check
        if (cmd->fmt_tex == DMP_DV_RGBA8888) {
          int ret = _SwizzleCheck(3, cmd);
          if (ret != 0) {
            return ret;
          }
        } else if (cmd->fmt_tex == DMP_DV_RGB888) {
          int ret = _SwizzleCheck(2, cmd);
          if (ret != 0) {
            return ret;
          }
        } else if (cmd->fmt_tex == DMP_DV_LUT) {
          int ret = _SwizzleCheck(3, cmd);
          if (ret != 0) {
            return ret;
          }
        } else {
          SET_ERR("Invalid argument: cmd->fmt_wr must be DMP_DV_RGBA8888, DMP_DV_RGB888 or DMP_DV_RGBLUT");
          return -1;
        }

        // conversion check
        if (cmd->cnv_type != DMP_DV_CNV_FP16_SUB &&
            cmd->cnv_type != DMP_DV_CNV_FP16_DIV_255) {
          SET_ERR("Invalid argument: cmd->cnv_type must be DMP_DV_CNV_FP16_DIV_255 or DMP_DV_CNV_FP16_SUB");
          return -1;
        }
      }
      // check rd
      if (cmd->use_rd) {
        if (!cmd->rd.mem) {
          SET_ERR("Invalid argument: cmd->rd.mem is NULL");
          return -1;
        }
        if (cmd->fmt_rd != DMP_DV_RGBA8888 &&
            cmd->fmt_rd != DMP_DV_RGB888) {
          SET_ERR("Invalid argument: cmd->fmt_rd must be DMP_DV_RGBA8888 or DMP_DV_RGB888");
          return -1;
        }
        if (cmd->stride_rd == 0) {
          SET_ERR("Invalid argument: cmd->stride_rd must be non-zero");
          return -1;
        }
        if (cmd->stride_rd < STRIDE_RD_MIN) {
          SET_ERR("Invalid argument: cmd->stride_rd must be higher than %d", STRIDE_RD_MIN + 1);
          return -1;
        }
        if (STRIDE_RD_MAX < cmd->stride_rd) {
          SET_ERR("Invalid argument: cmd->stride_rd must be smaller than %d", STRIDE_RD_MIN + 1);
          return -1;
        }
      }

      // register buffers
      uint64_t size = cmd->rect_width * cmd->rect_height * _GetPixelSize(cmd->fmt_wr);
      output_bufs.push_back(std::make_pair(cmd->wr, size));
      if (cmd->use_rd) {
        size = cmd->rect_width * cmd->rect_height * _GetPixelSize(cmd->fmt_rd);
        input_bufs.push_back(std::make_pair(cmd->rd, size));
      }
      if (cmd->use_tex) {
        size = cmd->tex_width * cmd->tex_height * _GetPixelSize(cmd->fmt_tex);
        input_bufs.push_back(std::make_pair(cmd->tex, size));
      }

      return 0;
    }

    /// @brief Fills command of version 0 in the format suitable for later execution on the device.
    int FillKCommand_v0(struct dmp_dv_kcmdraw_ipu_v0 *kcmd, struct dmp_dv_cmdraw_ipu_v0 *cmd, uint32_t& size) {
      if (cmd->header.size != sizeof(*kcmd)) {
        SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
            (int)cmd->header.size, (int)cmd->header.version);
        return -1;
      }

      size_t req_size = sizeof(*kcmd);

      if (size >= req_size) {
        kcmd->header.size = sizeof(struct dmp_dv_kcmdraw_ipu_v0);
        kcmd->header.version = 0;

        kcmd->tex.fd   = CDMPDVMem::get_fd(cmd->tex.mem);
        kcmd->tex.rsvd = 0;
        kcmd->tex.offs = cmd->tex.offs;
        kcmd->rd.fd    = CDMPDVMem::get_fd(cmd->rd.mem);
        kcmd->rd.rsvd  = 0;
        kcmd->rd.offs  = cmd->rd.offs;
        kcmd->wr.fd    = CDMPDVMem::get_fd(cmd->wr.mem);
        kcmd->wr.rsvd  = 0;
        kcmd->wr.offs  = cmd->wr.offs;

        kcmd->fmt_tex      = cmd->fmt_tex;
        kcmd->fmt_rd       = cmd->fmt_rd;
        kcmd->fmt_wr       = cmd->fmt_wr;
        kcmd->tex_width    = cmd->tex_width;
        kcmd->tex_height   = cmd->tex_height;
        kcmd->rect_width   = cmd->rect_width;
        kcmd->rect_height  = cmd->rect_height;
        kcmd->scale_width  = _f2fp24(1.0f/static_cast<float>(cmd->tex_width));
        kcmd->scale_height = _f2fp24(1.0f/static_cast<float>(cmd->tex_height));
        kcmd->stride_rd    = cmd->stride_rd;
        kcmd->stride_wr    = cmd->stride_wr;
        for(unsigned i = 0; i < sizeof(cmd->lut)/sizeof(cmd->lut[0]); i++) {
          kcmd->lut[i] = cmd->lut[i];
        }
        kcmd->ncolor_lut      = cmd->ncolor_lut;
        kcmd->alpha           = cmd->alpha;
        kcmd->transpose       = cmd->transpose;
        kcmd->use_const_alpha = cmd->use_const_alpha;
        kcmd->use_tex         = cmd->use_tex;
        kcmd->use_rd          = cmd->use_rd;
        kcmd->blf             = cmd->blf;
        kcmd->ridx            = cmd->ridx;
        kcmd->gidx            = cmd->gidx;
        kcmd->bidx            = cmd->bidx;
        kcmd->aidx            = cmd->aidx;
        kcmd->cnv_type        = cmd->cnv_type;
        kcmd->cnv_param[0]    = cmd->cnv_param[0];
        kcmd->cnv_param[1]    = cmd->cnv_param[1];
        kcmd->cnv_param[2]    = cmd->cnv_param[2];
      }
      size = req_size;
      return 0;
    }

    /// @brief auxiliary function for CheckRaw_v0
    static int _SwizzleCheck (uint8_t max_idx, const struct dmp_dv_cmdraw_ipu_v0 * cmd) {
      int8_t indices [] = {cmd->ridx, cmd->gidx, cmd->bidx, cmd->aidx};
      char index_names[][16] = {"cmd->ridx", "cmd->gidx", "cmd->bidx", "cmd->aidx"};
      int _range[4] = {};  // store which cmd->*idx has the index
      assert(max_idx <= sizeof(_range)/sizeof(_range[0]));
      for(uint8_t i = 0; i <= max_idx; i++) {
        if (indices[i] < 0 || max_idx < indices[i]) { 
          SET_ERR("Invalid argument: %s is %d", index_names[i], indices[i]);
          return -1;
        }
        if (_range[indices[i]]) {
          SET_ERR("Invalid argument: %s and %s has the same value '%d'", index_names[_range[indices[i]] - 1], index_names[i], indices[i]);
          return -1;
        } else {
          _range[indices[i]] = i + 1;
        }
      }
      return 0;
    };

    /// @brief auxiliary function for CheckRaw_v0
    static int _GetPixelSize(int img_fmt) {
      switch (img_fmt) {
        case DMP_DV_RGB888:
          return 3;
        case DMP_DV_RGBA8888:
          return 4;
        case DMP_DV_RGBFP16:
          return 6;
        case DMP_DV_LUT:
          return 1;
        default:
          return -1;
      }
    }

    static uint32_t _f2fp24(float g)
    {
      static const uint32_t expW = 7;
      static const uint32_t manW = 16;
      union { unsigned int uu; float ff; } jj;
      jj.ff = g;
      uint32_t uf = jj.uu;
      uint32_t u = 0;
      int neg = uf & 0x80000000;
      uint32_t expf = (uf >> 23) & 0xff;
      int32_t exp = expf - 127 + (1 << (expW - 1)) - 1;

      if (!expf || (exp <= 0)) {
        u=0;
      } else if (expf == 255 || exp >= ((1 << expW) - 1)) {
        exp = (1 << expW) - 1;
        u = (exp<<manW) | (neg ? (1 << (expW + manW)) : 0);
      } else {
        u = (exp<<manW) | (neg ? (1 << (expW + manW)) : 0);
        u |= (uf & ((1 << 23) - 1)) >> (23 - manW);
        if (manW < 22) {
          int half = uf & (1 << (22 - manW));
          if (half) {
            u += 1;
          }
        }
      }
      return u;
    }
};

const int32_t  CDMPDVCmdListIPUHelper::STRIDE_WR_MIN   = -32768;
const int32_t  CDMPDVCmdListIPUHelper::STRIDE_WR_MAX   = 32767;
const int32_t  CDMPDVCmdListIPUHelper::STRIDE_RD_MIN   = -32768;
const int32_t  CDMPDVCmdListIPUHelper::STRIDE_RD_MAX   = 32767;
const uint32_t CDMPDVCmdListIPUHelper::RECT_WIDTH_MAX  = 4095;
const uint32_t CDMPDVCmdListIPUHelper::RECT_HEIGHT_MAX = 4095;
const uint32_t CDMPDVCmdListIPUHelper::TEX_WIDTH_MAX   = 4095;
const uint32_t CDMPDVCmdListIPUHelper::TEX_HEIGHT_MAX  = 4095;
