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
 * @brief Helper object work working with command list for CONV accelerator implementation.
 */
#pragma once

#include "cmdlist.hpp"


/// @brief Helper object work working with command list for CONV accelerator.
class CDMPDVCmdListConvHelper : public CDMPDVCmdListKHelper {
 public:
  /// @brief Constructor.
  CDMPDVCmdListConvHelper(CDMPDVContext *ctx) : CDMPDVCmdListKHelper(ctx) {
    fnme_acc_ = DMP_DV_DEV_PATH_CONV;
  }

  /// @brief Destructor.
  virtual ~CDMPDVCmdListConvHelper() {
    for (auto it = helper_bufs_.rbegin(); it != helper_bufs_.rend(); ++it) {
      dmp_dv_mem_release(*it);
    }
  }

  /// @brief Creates object of this type.
  static CDMPDVCmdListDeviceHelper* Create(CDMPDVContext *ctx) {
    return new CDMPDVCmdListConvHelper(ctx);
  }

 private:
  /// @brief Checks provided command for validness.
  virtual int CheckRaw(dmp_dv_cmdraw *cmd,
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                       std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
    switch (cmd->device_type) {
      case DMP_DV_DEV_CONV:
        switch (cmd->version) {
          case 0:
            return CheckRaw_v0((dmp_dv_cmdraw_conv_v0*)cmd, input_bufs, output_bufs);

          case 1:
            return CheckRaw_v1((dmp_dv_cmdraw_conv_v1*)cmd, input_bufs, output_bufs);

          default:
            SET_ERR("Invalid argument: cmd->version %d is not supported with device_type %d",
                    (int)cmd->version, cmd->device_type);
            return ENOTSUP;
        }
        break;
      case DMP_DV_DEV_FC:
        switch (cmd->version) {
          case 0:
            return CheckRawFC_v0((dmp_dv_cmdraw_fc_v0*)cmd, input_bufs, output_bufs);

          default:
            SET_ERR("Invalid argument: cmd->version %d is not supported with device_type %d on device_type %d",
                    (int)cmd->version, cmd->device_type, DMP_DV_DEV_CONV);
            return ENOTSUP;
        }
        break;
      default:
        SET_ERR("Invalid argument: handling of cmd->device_type %d is not supported on device_type %d",
                cmd->device_type, DMP_DV_DEV_CONV);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Fills command in the format suitable for later execution on the device.
  virtual int FillKCommand(uint8_t *kcmd, dmp_dv_cmdraw *cmd, uint32_t& size) {
    switch (cmd->device_type) {
      case DMP_DV_DEV_CONV:
        switch (cmd->version) {
          case 0:
            return FillKCommand_v0((dmp_dv_kcmdraw_conv_v0*)kcmd, (dmp_dv_cmdraw_conv_v0*)cmd, size);

          case 1:
            return FillKCommand_v1((dmp_dv_kcmdraw_conv_v1*)kcmd, (dmp_dv_cmdraw_conv_v1*)cmd, size);

          default:
            SET_ERR("Invalid argument: cmd->version %d is not supported", (int)cmd->version);
            return ENOTSUP;
        }
        break;
      case DMP_DV_DEV_FC:
        switch (cmd->version) {
          case 0:
            return FillKCommandFC_v0((dmp_dv_kcmdraw_conv_v0*)kcmd, (dmp_dv_cmdraw_fc_v0*)cmd, size);

          default:
            SET_ERR("Invalid argument: cmd->version %d is not supported with device_type %d on device_type %d",
                    (int)cmd->version, cmd->device_type, DMP_DV_DEV_CONV);
            return ENOTSUP;
        }
        break;
      default:
        SET_ERR("Invalid argument: handling of cmd->device_type %d is not supported on device_type %d",
                cmd->device_type, DMP_DV_DEV_CONV);
        return ENOTSUP;
    }
    SET_LOGIC_ERR();
    return -1;
  }

  /// @brief Checks command of version 0 for validness.
  int CheckRaw_v0(struct dmp_dv_cmdraw_conv_v0 *cmd,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {

    if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_conv_v0)) {
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
    if (!cmd->z) {
      SET_ERR("Invalid argument: cmd->z is 0");
      return -1;
    }

    const int max_kernel_size = ctx_->get_max_kernel_size();

    const uint64_t input_size = (uint64_t)cmd->w * cmd->h * cmd->c * cmd->z;
    input_bufs.push_back(std::make_pair(cmd->input_buf, input_size));

    struct conv_data_size conv_size;
    init_conv_input_size_v0_4(cmd->w, cmd->h, cmd->z, cmd->c, &conv_size);
    struct dmp_dv_kcmdraw_conv_v0 kcmd;
    memset(&kcmd, 0, sizeof(kcmd));
    kcmd.header.size = sizeof(kcmd);
    kcmd.header.version = 0;
    kcmd.topo = cmd->topo;
    kcmd.w = cmd->w;
    kcmd.h = cmd->h;
    kcmd.c = cmd->c;
    kcmd.z = cmd->z;
    kcmd.input_circular_offset = cmd->input_circular_offset;
    kcmd.output_mode = cmd->output_mode;
    uint64_t output_size = 0;
    bool valid_multi_run = true;

    for (uint32_t topo = cmd->topo, i_run = 0; topo; topo >>= 1, ++i_run) {
      if ((cmd->run[i_run].conv_enable) && (!cmd->run[i_run].pz)) {
        SET_ERR("Invalid argument: cmd->run[%d]->pz is 0", i_run);
        return -1;
      }
      if (cmd->run[i_run].conv_enable) {
        switch (cmd->run[i_run].pool_enable) {
          case 0:  // no pooling
            break;
          case 1:  // max pooling
            break;
          default:
            SET_ERR("cmd->run[%d] pooling of type %d cannot be combined with convolution", i_run, cmd->run[i_run].pool_enable);
            return -1;
        }
      }
      const int is_deconv = (cmd->run[i_run].conv_enable & 4) ? 1 : 0;
      const int kx = cmd->run[i_run].p & 0xFF;
      const int ky = (cmd->run[i_run].p & 0xFF00) ? (cmd->run[i_run].p & 0xFF00) >> 8 : kx;
      const int pad[4] = {(int)(cmd->run[i_run].conv_pad & 0x7F), (int)((cmd->run[i_run].conv_pad >> 8) & 0xFF),
                          (int)((cmd->run[i_run].conv_pad >> 16) & 0x7F), (int)((cmd->run[i_run].conv_pad >> 24) & 0xFF)};
      const int stride[2] = {(int)(cmd->run[i_run].conv_stride & 0xFF), (int)((cmd->run[i_run].conv_stride >> 8) & 0xFF)};
      const int pool_kx = cmd->run[i_run].pool_size & 0xFF;
      const int pool_ky = (cmd->run[i_run].pool_size >> 8) & 0xFF;
      const int pool_pad[4] = {(int)(cmd->run[i_run].pool_pad & 0x7F), (int)((cmd->run[i_run].pool_pad >> 8) & 0xFF),
                               (int)((cmd->run[i_run].pool_pad >> 16) & 0x7F), (int)((cmd->run[i_run].pool_pad >> 24) & 0xFF)};
      const int pool_stride[2] = {(int)(cmd->run[i_run].pool_stride & 0xFF), (int)((cmd->run[i_run].pool_stride >> 8) & 0xFF)};
      const int m = cmd->run[i_run].m;
      const int w = conv_size.w, h = conv_size.h, c = conv_size.c;
      const int dil[2] = {std::max((int)(cmd->run[i_run].conv_dilation & 0xFF), 1),
                          std::max((int)((cmd->run[i_run].conv_dilation >> 8) & 0xFF), 1)};

      if ((!cmd->run[i_run].conv_enable) && (!cmd->run[i_run].pool_enable) &&
          (!cmd->run[i_run].actfunc) && (!(cmd->run[i_run].lrn & 1))) {
        SET_ERR("Invalid argument: cmd->run[%d] specify no operation", i_run);
        return -1;
      }
      if ((cmd->run[i_run].conv_enable == 1) && (!cmd->run[i_run].weight_buf.mem)) {
        SET_ERR("Invalid argument: cmd->run[%d].weight_buf.mem is NULL", i_run);
        return -1;
      }
      if (cmd->run[i_run].conv_enable == 1) {
        if ((kx < 1) || (kx > max_kernel_size) || (ky < 1) || (ky > max_kernel_size)) {
          SET_ERR("Unsupported convolutional kernel size %dx%d, only sizes from 1 to %d are supported",
                  kx, ky, max_kernel_size);
          return -1;
        }
        if ((stride[0] < 1) || (stride[1] < 1)) {
          SET_ERR("Stride of convolution must be greater than 0, got %dx%d", stride[0], stride[1]);
          return -1;
        }
        if ((kx > pad[0] + w + pad[1]) || (ky > pad[2] + h + pad[3])) {
          SET_ERR("Input (%d, %d) with padding L=%d, R=%d, T=%d, B=%d is too small for convolution of size (%d, %d)",
                  w, h, pad[0], pad[1], pad[2], pad[3], kx, ky);
          return -1;
        }
      }
      if ((cmd->run[i_run].conv_enable == 3) && (cmd->run[i_run].m != c)) {
        SET_ERR("Depthwise convolution only supports one-to-one mapping, got c=%d m=%d",
                c, cmd->run[i_run].m);
        return -1;
      }
      switch (cmd->run[i_run].pool_enable) {
        case 0:
          break;
        case 1:
        case 2:
        {
          if ((cmd->run[i_run].pool_enable == 1) &&
              ((pool_kx < 2) || (pool_kx > 3) || (pool_ky < 1) || (pool_ky > 3))) {
            SET_ERR("Unsupported max pooling size %dx%d, only 2x1, 2x2, 2x3, 3x1, 3x2, 3x3 are supported",
                    pool_kx, pool_ky);
            return -1;
          }
          if ((cmd->run[i_run].pool_enable == 2) &&
              ((pool_kx < 1) || (pool_kx > max_kernel_size) || (pool_ky < 1) || (pool_ky > max_kernel_size))) {
            SET_ERR("Unsupported average pooling size %dx%d, only sizes from 1 to %d are supported",
                    pool_kx, pool_ky, max_kernel_size);
            return -1;
          }
          if ((pool_stride[0] < 1) || (pool_stride[1] < 1)) {
            SET_ERR("Stride of pooling must be greater than 0, got %dx%d", pool_stride[0], pool_stride[1]);
            return -1;
          }
          if ((pool_kx > pool_pad[0] + w + pool_pad[1]) || (pool_ky > pool_pad[2] + h + pool_pad[3])) {
            SET_ERR("Input (%d, %d) with padding L=%d, R=%d, T=%d, B=%d is too small for pooling of size (%d, %d)",
                    (int)cmd->w, (int)cmd->h, pool_pad[0], pool_pad[1], pool_pad[2], pool_pad[3], pool_kx, pool_ky);
            return -1;
          }
          if ((pool_kx != pool_ky) && (ctx_->get_svn_version() < 93)) {
            SET_ERR("Non-square pooling support requires /sys/class/dmp_dv/dv_conv/svn_version to be at least 93, got %d",
                    ctx_->get_svn_version());
            return -1;
          }
          break;
        }
        case 4:
          // Upsampling is always 2x2
          break;
        default:
          SET_ERR("Unsupported cmd->run[%d].pool_enable=%d", i_run, cmd->run[i_run].pool_enable);
          return -1;
      }
      if (cmd->run[i_run].lrn & 1) {
        if (c & 15) {
          SET_ERR("Unsupported number of channels for LRN layer, must be multiple of 16, got %d", c);
          return -1;
        }
        if ((cmd->run[i_run].conv_enable) || (cmd->run[i_run].pool_enable) || (cmd->topo != 1)) {
          SET_ERR("LRN must be a standalone layer");
          return -1;
        }
      }

      if ((is_deconv) && (ctx_->get_svn_version() < 93)) {
        SET_ERR("Deconvolution support requires /sys/class/dmp_dv/dv_conv/svn_version to be at least 93, got %d",
                ctx_->get_svn_version());
        return -1;
      }

      if ((dil[0] > 1) || (dil[1] > 1)) {
        if (cmd->topo != 1) {
          SET_ERR("Dilated convolution must be the only run, but topo=%u", cmd->topo);
          return -1;
        }
        if (cmd->run[i_run].pool_enable) {
          SET_ERR("Dilated convolution cannot be combined with pooling");
          return -1;
        }
        if ((!(kx & 1)) || (!(ky & 1))) {
          SET_ERR("Only odd kernel sizes are supported for dilated convolutions, got %dx%d",
                  kx, ky);
          return -1;
        }
        const int kxfull = (kx - 1) * dil[0] + 1,
                  kyfull = (ky - 1) * dil[1] + 1;
        if ((w + pad[0] + pad[1] < kxfull) || (h + pad[2] + pad[3] < kyfull)) {
          SET_ERR("Input size %dx%d pad_lrtb=%dx%dx%dx%d is too small for convolution of size %dx%d dilated by %dx%d",
                  w, h, pad[0], pad[1], pad[2], pad[3], kx, ky, dil[0], dil[1]);
          return -1;
        }
        const int min_svn_version = ctx_->is_zia_c2() ? 83 : 93;
        if ((ctx_->get_svn_version() < min_svn_version) && ((w < pad[0]) || (w < pad[1]) || (h < pad[2]) || (h < pad[3]))) {
          SET_ERR("Input size %dx%d pad_lrtb=%dx%dx%dx%d is too small for convolution of size %dx%d dilated by %dx%d "
                  "for /sys/class/dmp_dv/dv_conv/svn_version less than %d, got %d",
                  w, h, pad[0], pad[1], pad[2], pad[3], kx, ky, dil[0], dil[1], min_svn_version, ctx_->get_svn_version());
          return -1;
        }
        if ((is_deconv) && ((stride[0] != 1) || (stride[1] != 1))) {
          SET_ERR("Deconvolution with dilation only supports stride 1");
          return -1;
        }
        const int ox = get_conv_out_width(w, kxfull, pad[0], pad[1], stride[0], is_deconv),
                  oy = get_conv_out_width(h, kyfull, pad[2], pad[3], stride[1], is_deconv);
        if ((ox * stride[0] != w) || (oy * stride[1] != h)) {
          SET_ERR("Dilated convolution only supports \"same\" padding: (%d, %d, %d) => (%d, %d, %d); "
                  "w=%d, h=%d, kx=%d, ky=%d, pad_lrtb=(%d, %d, %d, %d), dil=(%d, %d)",
                  h, w, c, oy, ox, m, w, h, kx, ky, pad[0], pad[1], pad[2], pad[3], dil[0], dil[1]);
          return -1;
        }
        if (cmd->run[i_run].actfunc == 4) {
          SET_ERR("Dilated convolution and PReLU activation cannot be used together");
          return -1;
        }
      }
      if ((cmd->run[i_run].actfunc == 4) && (cmd->run[i_run].conv_enable) && (cmd->run[i_run].weight_fmt == 3)) {
        SET_ERR("Quantized weights and PReLU activation cannot be used together");
        return -1;
      }

      kcmd.run[i_run].actfunc = cmd->run[i_run].actfunc;
      kcmd.run[i_run].actfunc_param = cmd->run[i_run].actfunc_param;
      kcmd.run[i_run].conv_dilation = (uint16_t)dil[0] | ((uint16_t)dil[1] << 8);
      kcmd.run[i_run].conv_enable = cmd->run[i_run].conv_enable;
      kcmd.run[i_run].conv_pad = cmd->run[i_run].conv_pad;
      kcmd.run[i_run].conv_stride = cmd->run[i_run].conv_stride;
      kcmd.run[i_run].lrn = cmd->run[i_run].lrn;
      kcmd.run[i_run].m = cmd->run[i_run].m;
      kcmd.run[i_run].p = cmd->run[i_run].p;
      kcmd.run[i_run].pool_avg_param = cmd->run[i_run].pool_avg_param;
      kcmd.run[i_run].pool_enable = cmd->run[i_run].pool_enable;
      kcmd.run[i_run].pool_pad = cmd->run[i_run].pool_pad;
      kcmd.run[i_run].pool_size = cmd->run[i_run].pool_size;
      kcmd.run[i_run].pool_stride = cmd->run[i_run].pool_stride;
      kcmd.run[i_run].pz = cmd->run[i_run].pz;
      kcmd.run[i_run].rectifi_en = cmd->run[i_run].rectifi_en;
      kcmd.run[i_run].weight_fmt = cmd->run[i_run].weight_fmt;

      uint32_t weights_size = 0;
      get_conv_output_size_v0(&kcmd.run[i_run], &conv_size, &conv_size, &weights_size);
      if (weights_size) {
        input_bufs.push_back(std::make_pair(cmd->run[i_run].weight_buf, (uint64_t)weights_size));
      }

      int tiles = 1;
      int u_b_in, u_b_out;
      if (kcmd.run[i_run].lrn & 1) {
        tiles = calc_num_tiles_lrn(w, h, c, ctx_->get_ub_size() >> 10, &u_b_in, &u_b_out);
      }
      else if (!is_conv_2d_v0(&kcmd.run[i_run])) {
        if (kcmd.run[i_run].pool_enable) {
          tiles = calc_num_tiles_pool(w, h, c, &u_b_in, &u_b_out);
        }
      }
      else {
        tiles = calc_num_tiles_conv(
            w, h, c, m, kx, ky,
            pad[0], pad[1], pad[2], pad[3],
            stride[0], stride[1], dil[0], dil[1],
            ctx_->get_ub_size() >> 10, is_deconv, &u_b_in, &u_b_out);
      }
      if (tiles < 1) {
        SET_ERR("cmd->run[%d] requires at least %d bytes of unified buffer: w=%d h=%d c=%d m=%d p=0x%04x dil=0x%04x",
                i_run, u_b_in + u_b_out, w, h, c, m, kcmd.run[i_run].p, kcmd.run[i_run].conv_dilation);
        return -1;
      }
      /*if ((tiles > 1) && (kcmd.run[i_run].conv_enable) && (kcmd.run[i_run].pool_enable)) {
        SET_ERR("cmd->run[%d] requires %d (> 1) tiles and cannot contain convolution together with pooling",
                i_run, tiles);
        return -1;
      }*/  // there is no such restriction in HW since 20190729

      if ((kcmd.z > 1) || (kcmd.run[i_run].pz > 1) ||
          (dil[0] > 1) || (dil[1] > 1)) {
        // TODO: add more checks: no maxpool_with_argmax, no unpool_with_argmax.
        valid_multi_run = false;
      }

      if (topo & 1) {  // output goes to main memory
        if (!conv_size.size) {
          SET_ERR("Invalid argument: cmd->run[%d] produces output with zero size", i_run);
          return -1;
        }
        output_size += conv_size.size;

        // Next input will be the first
        init_conv_input_size_v0_4(cmd->w, cmd->h, cmd->z, cmd->c, &conv_size);
      }
      else {  // output goes to unified buffer
        if (tiles != 1) {
          SET_ERR("cmd->run[%d] wants tiles to be %d while only %d is supported for output in the Unified Buffer",
                  i_run, tiles, 1);
          return -1;
        }
      }
    }
    if (!output_size) {
      SET_LOGIC_ERR();
      return -1;
    }
    if (kcmd.topo != 1) {
      if (!valid_multi_run) {
        SET_ERR("Command cannot be executed with multiple runs (input is W=%d H=%d C=%d Z=%d)",
                (int)cmd->w, (int)cmd->h, (int)cmd->c, (int)cmd->z);
        return -1;
      }
      int ubuf_used = ubuf_get_single_tile_usage(&kcmd, ctx_->get_ub_size());
      if (ubuf_used > ctx_->get_ub_size()) {
        SET_ERR("Unified buffer should be at least %d bytes to process the input W=%d H=%d C=%d",
                ubuf_used, (int)cmd->w, (int)cmd->h, (int)cmd->c);
        return -1;
      }
    }

    // Success
    output_bufs.push_back(std::make_pair(cmd->output_buf, output_size));
    if (cmd->eltwise_buf.mem) {
      output_bufs.push_back(std::make_pair(cmd->eltwise_buf, output_size));
    }
    return 0;
  }

  /// @brief Checks command of version 0 for validness.
  int CheckRaw_v1(struct dmp_dv_cmdraw_conv_v1 *cmd,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& input_bufs,
                  std::vector<std::pair<struct dmp_dv_buf, uint64_t> >& output_bufs) {
    if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_conv_v1)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
      return -1;
    }

    if (cmd->u8tofp16_table.mem) {
      input_bufs.push_back(std::make_pair(cmd->u8tofp16_table, 6 * 256));
    }

    return CheckRaw_v0(&cmd->conv_cmd, input_bufs, output_bufs);
  }

  /// @brief Fills command of version 0 in the format suitable for later execution on the device.
  int FillKCommand_v0(struct dmp_dv_kcmdraw_conv_v0 *kcmd, struct dmp_dv_cmdraw_conv_v0 *cmd, uint32_t& size) {
    if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_conv_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
      return -1;
    }

    int n_run = 0;
    for (uint32_t topo = cmd->topo; topo; topo >>= 1) {
      ++n_run;
    }
    if (!n_run) {
      SET_ERR("CONV command should have at least one run");
      return -1;
    }

    uint32_t req_size = sizeof(*kcmd) - sizeof(kcmd->run) + n_run * sizeof(kcmd->run[0]);

    if ((kcmd) && (size < req_size)) {
      SET_ERR("Not enough buffer size for the CONV kernel command: %u < %u",
              size, req_size);
      return -1;
    }

    if (kcmd) {
      kcmd->header.size = req_size;
      kcmd->header.version = 0;

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

      for (int i_run = 0; i_run < n_run; ++i_run) {
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
        {
          uint16_t dil_x = cmd->run[i_run].conv_dilation & 0xFF,
                   dil_y = (cmd->run[i_run].conv_dilation >> 8) & 0xFF;
          dil_x = dil_x < 1 ? 1 : dil_x;
          dil_y = dil_y < 1 ? 1 : dil_y;
          kcmd->run[i_run].conv_dilation = dil_x | (dil_y << 8);
        }
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

  int FillKCommand_v1(struct dmp_dv_kcmdraw_conv_v1 *kcmd, struct dmp_dv_cmdraw_conv_v1 *cmd, uint32_t& size) {
    if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_conv_v1)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
      return -1;
    }

    // Reuse v0 implementation
    uint8_t *kcmd_v0 = (uint8_t *)kcmd;
    if (kcmd)
      kcmd_v0 += sizeof(struct dmp_dv_kbuf) + 8;
    int ret = FillKCommand_v0((dmp_dv_kcmdraw_conv_v0*)kcmd_v0, &cmd->conv_cmd, size);
    size += sizeof(struct dmp_dv_kbuf) + 8;

    // Fill v1 data
    if (kcmd) {
      kcmd->header.size = size;
      kcmd->header.version = 1;
      kcmd->u8tofp16_table.fd = CDMPDVMem::get_fd(cmd->u8tofp16_table.mem);
      kcmd->u8tofp16_table.rsvd = 0;
      kcmd->u8tofp16_table.offs = cmd->u8tofp16_table.offs;
      kcmd->to_bgr = cmd->to_bgr;
    }

    return ret;
  }

  /// @brief Checks command of version 0 for validness for legacy FC configuration.
  int CheckRawFC_v0(struct dmp_dv_cmdraw_fc_v0 *cmd,
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
  int FillKCommandFC_v0(struct dmp_dv_kcmdraw_conv_v0 *kcmd, struct dmp_dv_cmdraw_fc_v0 *cmd, uint32_t& size) {
    if (cmd->header.size != sizeof(struct dmp_dv_cmdraw_fc_v0)) {
      SET_ERR("Invalid argument: cmd->size %d is incorrect for version %d",
              (int)cmd->header.size, (int)cmd->header.version);
      return -1;
    }

    uint32_t req_size = sizeof(*kcmd) - sizeof(kcmd->run) + sizeof(kcmd->run[0]);
    if (size < req_size) {
      size = req_size;
      return 0;
    }

    // Convert command to convolutional configuration
    struct dmp_dv_cmdraw_conv_v0 conv;
    memset(&conv, 0, sizeof(conv));
    conv.header.size = sizeof(conv);
    conv.header.device_type = DMP_DV_DEV_CONV;
    conv.topo = 1;
    conv.w = 1;
    conv.h = 1;
    conv.c = cmd->input_size;
    conv.z = 1;
    conv.run[0].m = cmd->output_size;
    conv.run[0].conv_enable = 1;
    conv.run[0].p = 0x0101;
    conv.run[0].pz = 1;
    conv.run[0].conv_stride = 0x0101;
    conv.run[0].weight_fmt = cmd->weight_fmt ? 3 : 1;
    uint16_t actfunc = 0;
    switch (cmd->actfunc) {
      case 0:  // None
        actfunc = 0;
        break;
      case 1:  // ReLU
        actfunc = 2;
        break;
      case 2:  // Tanh
        actfunc = 1;
        break;
      case 3:  // Leaky ReLU
        actfunc = 2;
        break;
      case 4:  // Sigmoid
        actfunc = 3;
        break;
      default:
        SET_ERR("Unsupported actfunc %d for device_type %d on device_type %d",
                (int)cmd->actfunc, DMP_DV_DEV_FC, DMP_DV_DEV_CONV);
        return ENOTSUP;
    }
    conv.run[0].actfunc = actfunc;
    conv.run[0].actfunc_param = cmd->actfunc_param;

    conv.input_buf = cmd->input_buf;
    conv.output_buf = cmd->output_buf;

    // Allocate buffer for weights and repack them

    // Get old weights buffer
    CDMPDVMem *weights_fc = (CDMPDVMem*)cmd->weight_buf.mem;
    if (!weights_fc) {
      SET_LOGIC_ERR();
      return -1;
    }
    bool unmap_after = false;
    if (!weights_fc->get_ptr()) {
      if (!weights_fc->Map()) {
        return -1;
      }
      unmap_after = true;
    }
    bool sync_end_after = false;
    if (!weights_fc->get_sync_flags()) {
      sync_end_after = true;
      if (weights_fc->SyncStart(1, 0)) {
        if (unmap_after) {
          weights_fc->Unmap();
        }
        return -1;
      }
    }

    uint16_t *quant_map;
    uint16_t *bias;
    uint8_t *weights;
    uint32_t bias_offs = (uint32_t)cmd->input_size * cmd->output_size;
    if (cmd->weight_fmt) {
      quant_map = (uint16_t*)(weights_fc->get_ptr() + cmd->weight_buf.offs);
      weights = ((uint8_t*)quant_map) + 512;
    }
    else {
      quant_map = NULL;
      weights = weights_fc->get_ptr() + cmd->weight_buf.offs;
      bias_offs <<= 1;
    }
    if (bias_offs & 15) {
      bias_offs += 16 - (bias_offs & 15);
    }
    bias = (uint16_t*)(weights + bias_offs);

    size_t packed_weights_size = 0;
    int res = dmp_dv_pack_conv_weights(
                conv.c, 1, 1, conv.run[0].m,
                quant_map, NULL, NULL, NULL,
                NULL, &packed_weights_size);
    if (res) {
      return res;
    }

    dmp_dv_mem mem = dmp_dv_mem_alloc((dmp_dv_context)ctx_, packed_weights_size);
    uint8_t *ptr = NULL;
    if (!mem) {
      res = ENOMEM;
      goto L_EXIT;
    }
    ptr = dmp_dv_mem_map(mem);
    if ((!ptr) || (dmp_dv_mem_sync_start(mem, 0, 1))) {
      res = -1;
      goto L_EXIT;
    }

    res = dmp_dv_pack_conv_weights(
        conv.c, 1, 1, conv.run[0].m,
        quant_map, weights, bias, NULL,
        ptr, &packed_weights_size);

    dmp_dv_mem_unmap(mem);

    conv.run[0].weight_buf.mem = mem;

    L_EXIT:

    if (unmap_after) {
      weights_fc->Unmap();
    }
    else if (sync_end_after) {
      weights_fc->SyncEnd();
    }

    if (!res) {
      res = FillKCommand_v0(kcmd, &conv, size);
    }
    if (res) {
      dmp_dv_mem_release(mem);
      return res;
    }

    helper_bufs_.push_back(mem);
    return 0;
  }

  /// @brief Helper buffers.
  std::vector<dmp_dv_mem> helper_bufs_;
};
