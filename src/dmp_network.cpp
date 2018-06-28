/*
 *  Copyright 2018 Digital Media Professionals Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

//#define DUMP_LAYER_OUTPUT

#include "dmp_network.h"
#include "ieeehalfprecision.h"

void CDMP_Network::verbose(bool en) { _verbose = en; }

fpga_layer &CDMP_Network::get_layer(unsigned int i) {
  if (i >= layers.size()) {
    static fpga_layer err_layer;
    std::cerr << "Requested layer id out of bound" << std::endl;
    return err_layer;
  }
  return layers[i];
}

top_conv_conf &CDMP_Network::get_conv_layer(unsigned int i) {
  if (i >= conv_layers.size()) {
    static top_conv_conf err_layer;
    std::cerr << "Requested convolution layer id out of bound" << std::endl;
    return err_layer;
  }
  return conv_layers[i];
}

top_fc_conf &CDMP_Network::get_ip_layer(unsigned int i) {
  if (i >= fc_layers.size()) {
    static top_fc_conf err_layer;
    std::cerr << "Requested inner product layer id out of bound" << std::endl;
    return err_layer;
  }
  return fc_layers[i];
}

static void load_IQ_table(top_fc_conf &fc_layer) {
  // Load IQ table
  unsigned short *addr =
      reinterpret_cast<unsigned short *>(fc_layer.sw.weight_addr);

  for (int i = 0; i < 256; i++) {
    *(volatile unsigned int *)(dmp::modules::get_iomap_fc() + 0x004C) =
        (2 << 28) | (1 << 24) | (i << 16) | addr[i];  // matrix vertical
  }
}

bool CDMP_Network::load_weights(const std::string &filename) {
  std::ifstream dfs(filename.c_str());
  if (!dfs.is_open()) {
    std::cerr << "Failed to open conv weights file." << std::endl;
    return false;
  };

  int bytes_read = 0;
  if (_verbose) std::cout << "Loading CONV/IP  Weights." << std::endl;

  dfs.read((char *)weight_buffer_addr, memory_size_request[0]);
  bytes_read += memory_size_request[0];

  if (_verbose) {
    std::cout << bytes_read << " bytes read."
              << " [DONE]" << std::endl;
  }

  dfs.close();

  if (fc_layers.size()) load_IQ_table(fc_layers[0]);

  return true;
}

static int get_mem_req_vec_total(std::vector<std::pair<int, int>> &vec) {
  int tot = 0;
  for (auto i = vec.begin(); i != vec.end(); i++) {
    tot += i->first;
  }
  return tot;
}

static void print_mem_req_vec(std::vector<std::pair<int, int>> &vec) {
  std::cout << "---------------\n";
  std::cout << "Memory Requests\n";
  std::cout << "---------------\n";
  std::cout << std::dec;
  for (auto i = vec.begin(); i != vec.end(); i++) {
    std::cout << "MEM  : " << i->second << "\n";
    std::cout << "SIZE : " << i->first << "\n";
  }
  std::cout << "TOTAL REQ : " << get_mem_req_vec_total(vec) << "\n";
}

static int get_mem_free_vec_total(std::vector<std::pair<int, int>> &vec) {
  int tot = 0;
  for (auto i = vec.begin(); i != vec.end(); i++) {
    tot += i->first;
  }
  return tot;
}

static void print_mem_free_vec(std::vector<std::pair<int, int>> &vec) {
  std::cout << "-----------\n";
  std::cout << "Free Memory\n";
  std::cout << "-----------\n";
  for (auto i = vec.begin(); i != vec.end(); i++) {
    std::cout << std::hex;
    std::cout << "ADDR : " << i->second << "\n";
    std::cout << std::dec;
    std::cout << "SIZE : " << i->first << "\n";
  }
  std::cout << "FREE TOTAL : " << get_mem_free_vec_total(vec) << "\n";
}

static int memory_alloc(
    int fd, bool _verbose, std::vector<unsigned long> &memory_size_request,
    std::vector<unsigned long> &reserved_memory_addresses_fpga,
    std::vector<unsigned long> &reserved_memory_addresses_cpu) {
  int N = 2;

  std::vector<std::pair<int, int>> mem_req_vec;
  for (int i = 0; i < N; i++) {
    std::pair<int, int> p;
    p.first = memory_size_request[i];
    p.second = i;
    mem_req_vec.push_back(p);
  }

  static std::vector<std::pair<int, int>> mem_free_vec;
  static bool mem_free_initialized = false;
  if (!mem_free_initialized) {
    unsigned int memSec[16];
    for (int k = 0; k < 16; k++) memSec[k] = 0;
    ioctl(fd, CNV_MEMSEC, memSec);

    int j = 0;
    while (memSec[2 * j] != 0) {
      std::pair<int, int> p;
      p.first = memSec[2 * j + 1];
      p.second = memSec[2 * j];
      mem_free_vec.push_back(p);
      j++;
    }
    mem_free_initialized = true;
  }

  std::sort(mem_req_vec.begin(), mem_req_vec.end());
  std::reverse(mem_req_vec.begin(), mem_req_vec.end());

  if (_verbose) {
    print_mem_req_vec(mem_req_vec);
    print_mem_free_vec(mem_free_vec);
  }

  int tot_req = get_mem_req_vec_total(mem_req_vec);
  int tot_free_0 = get_mem_free_vec_total(mem_free_vec);

  int errors = 0;

  for (auto i = mem_req_vec.begin(); i != mem_req_vec.end(); i++) {
    int mem_size = i->first;
    int mem_idx = i->second;

    // int mem_size_pages = (mem_size / page_size) + ((mem_size % page_size) ==
    // 0 ? 0 : 1);
    // mem_size = mem_size_pages * page_size;

    std::sort(mem_free_vec.begin(), mem_free_vec.end());
    std::reverse(mem_free_vec.begin(), mem_free_vec.end());

    if (mem_free_vec[0].first >= mem_size) {
      reserved_memory_addresses_fpga[mem_idx] = mem_free_vec[0].second;
      mem_free_vec[0].first = mem_free_vec[0].first - mem_size;
      mem_free_vec[0].second = mem_free_vec[0].second + mem_size;
    } else {
      std::cerr << "*** Memory Allocation Failed! ***\n";
      std::cerr << "Size Required : " << mem_size << "\n";
      std::cerr << "Size Available: " << mem_free_vec[0].first << "\n";
      errors++;
    }
  }

  int tot_free_1 = get_mem_req_vec_total(mem_free_vec);
  if (!(tot_free_0 - tot_req == tot_free_1)) {
    std::cerr << "\n### FAIL ###\n";
    errors++;
  }

  if (_verbose) {
    if (tot_free_0 - tot_req == tot_free_1) {
      std::cout << "\n### PASS ###\n";
    } else {
      std::cerr << "\n### FAIL ###\n";
      errors++;
    }
  }

  unsigned long iomap_ddr = dmp::modules::get_iomap_ddr();

  for (int i = 0; i < N; i++) {
    reserved_memory_addresses_cpu[i] =
        reserved_memory_addresses_fpga[i] - SYS_DDR_BASE_PA + iomap_ddr;
    if (_verbose)
      std::cout << std::hex << i << "\t" << reserved_memory_addresses_fpga[i]
                << "\t-> " << reserved_memory_addresses_cpu[i] << "\n";
  }

  return errors;
}

bool CDMP_Network::reserve_memory(bool set_addr) {
  if (_verbose) {
    std::cout << "Reserving memory" << std::endl;
  }

  int N = 2;
  reserved_memory_addresses_fpga.resize(N);
  reserved_memory_addresses_cpu.resize(N);

  int fdC = dmp::modules::get_fdC();
  int error = memory_alloc(fdC, _verbose, memory_size_request,
                           reserved_memory_addresses_fpga,
                           reserved_memory_addresses_cpu);

  if (_verbose) {
    std::cout << "Reserving memory complete" << std::endl;
  }

  if (set_addr) {
    SetLayerAddresses();
  }

  return (error == 0);
}

static void remap(unsigned short *src, unsigned short *dst, int x_size,
                  int y_size, int channel_size) {
  for (int y = 0; y < y_size; y++) {
    for (int x = 0; x < x_size; x++) {
      for (int i = 0; i < channel_size; i += 8) {
        int copy_size = (channel_size - i > 8 ? 8 : channel_size - i);
        memcpy(dst + (y * x_size + x) * channel_size + i,
               src + i * (x_size * y_size) + (x * y_size + y) * copy_size,
               copy_size * sizeof(unsigned short));
      }
    }
  }
}

static void remap_hw(unsigned short *src, unsigned short *dst, int x_size,
                     int y_size, int channel_size) {
  for (int y = 0; y < y_size; y++) {
    for (int x = 0; x < x_size; x++) {
      for (int i = 0; i < channel_size; i += 8) {
        int copy_size = (channel_size - i > 8 ? 8 : channel_size - i);
        memcpy(dst + i * (x_size * y_size) + (x * y_size + y) * copy_size,
               src + (y * x_size + x) * channel_size + i,
               copy_size * sizeof(unsigned short));
      }
    }
  }
}

static void run_conv(top_conv_conf *conf, int t_sleep) {
  unsigned int iomap_cnv = dmp::modules::get_iomap_cnv();
  unsigned int fdC = dmp::modules::get_fdC();

  // std::cout<<"\t\t("<<iomap_cnv<<","<<fdC<<")"<<std::endl;

  // Copy Configuration Struct to RISC-V RAM
  *(volatile unsigned int *)(iomap_cnv + 0x0080) =
      0x2000;  // Conf. struct starts at 0x2000
  // int conf_size = sizeof(configuration)/4 + ((sizeof(configuration) & 0x3) ==
  // 0 ? 0 : 1);
  int conf_size_bytes = hw_conf_size(conf);
  int conf_size =
      conf_size_bytes / 4 + ((conf_size_bytes & 0x3) != 0);  // words
  // std::cout << "conf_size = " << conf_size << "\n";

  unsigned int *conf_ptr = (unsigned int *)conf;

  for (int i = 0; i < conf_size; i++) {
    *(volatile unsigned int *)(iomap_cnv + 0x0084) = conf_ptr[i];
  }

  // Run the HW ACC
  *(volatile unsigned int *)(iomap_cnv + 0x0040) = 0x00000001;
  if (t_sleep > 0) {
    usleep(t_sleep);
  }
  dmp::modules::cnvWaitInt(fdC);
  *(volatile unsigned int *)(iomap_cnv + 0x0420) = 0x00000000;
}

static void run_ip(top_fc_conf *fc) {
  unsigned int iomap_fc = dmp::modules::get_iomap_fc();
  int fdF = dmp::modules::get_fdF();

  // Calculate Fully Connected layers
  *(volatile unsigned int *)(iomap_fc + 0x0044) =
      0x00000223;  // Raw vector, quant weights, raw bias as separate input
  *(volatile unsigned int *)(iomap_fc + 0x0048) =
      fc->hw.actfunc;  // w*v, tanh activation function
  *(volatile unsigned int *)(iomap_fc + 0x0050) =
      fc->hw.input_size;  // matrix horizontal size
  *(volatile unsigned int *)(iomap_fc + 0x0054) =
      fc->hw.output_size;  // matrix vertical
  *(volatile unsigned int *)(iomap_fc + 0x0058) =
      fc->hw.output_base_addr;  // output write address
  *(volatile unsigned int *)(iomap_fc + 0x0074) =
      fc->hw.input_base_addr;  // input read address
  *(volatile unsigned int *)(iomap_fc + 0x007C) = fc->hw.stride;  // stride
  *(volatile unsigned int *)(iomap_fc + 0x0080) =
      fc->hw.weight_addr;  // weight address
  *(volatile unsigned int *)(iomap_fc + 0x0060) =
      fc->hw.bias_addr;  // bias address
  *(volatile unsigned int *)(iomap_fc + 0x0064) =
      fc->hw.bias_size;  // bias size (in bytes)

  //// Run the HW ACC
  *(volatile unsigned int *)(iomap_fc + 0x0040) = 0x00000002;  // Start
  dmp::modules::cnvWaitInt(fdF);
  //*(volatile unsigned int*)(iomap_fc + 0x0020) = 0x00000000;
}

static void run_softmax(fpga_layer &layer, int softmax_axis) {
  void *src_buffer = layer.addr_cpu_input;
  void *dst_buffer = layer.addr_cpu_output;

  if (layer.is_input_hw_layout) {
    unsigned short *src = reinterpret_cast<unsigned short *>(src_buffer);
    unsigned short *dst = reinterpret_cast<unsigned short *>(dst_buffer);
    int x_size = layer.input_dim[0];
    int y_size = layer.input_dim[1];
    int channel_size = layer.input_dim[2];
    dst += x_size * y_size * channel_size;
    remap(src, dst, x_size, y_size, channel_size);
    src_buffer = dst;
  }

  int tensor_size = 1, group_size, axis_size, remain_size;
  int axis_stride = 1, group_stride;
  for (int i = 0; i < layer.input_dim_size; i++) {
    tensor_size *= layer.input_dim[i];
    if (i > softmax_axis) axis_stride *= layer.input_dim[i];
  }
  axis_size = layer.input_dim[softmax_axis];
  remain_size = axis_stride;
  group_size = tensor_size / axis_size / remain_size;
  group_stride = axis_size * axis_stride;
  halfp2singles(dst_buffer, src_buffer, tensor_size);

  for (int j = 0; j < group_size; j++) {
    for (int k = 0; k < remain_size; k++) {
      float *f = reinterpret_cast<float *>(dst_buffer) + (j * group_stride + k);
      float fmax = f[0];
      for (int i = 1; i < axis_size; i++)
        fmax = std::max(fmax, f[i * axis_stride]);
      for (int i = 0; i < axis_size; i++) f[i * axis_stride] -= fmax;
      float e_sum = 0;
      for (int i = 0; i < axis_size; i++) {
        float d = std::exp(f[i * axis_stride]);
        f[i * axis_stride] = d;
        e_sum += d;
      }
      if (std::fabs(e_sum) < 1e-6f) {
        std::cout << "very low sum of last layer / softmax. To chech\n";
      }

      float inv_e_sum = 1.0f / e_sum;
      for (int i = 0; i < axis_size; i++) {
        f[i * axis_stride] *= inv_e_sum;  // SoftMax
      }
    }
  }
}

static void run_flatten(fpga_layer &layer) {
  if (!layer.is_input_hw_layout) return;
  unsigned short *src =
      reinterpret_cast<unsigned short *>(layer.addr_cpu_input);
  unsigned short *dst =
      reinterpret_cast<unsigned short *>(layer.addr_cpu_output);
  int x_size = layer.input_dim[0];
  int y_size = layer.input_dim[1];
  int channel_size = layer.input_dim[2];
  remap(src, dst, x_size, y_size, channel_size);
}

static void run_copy_concat(fpga_layer &layer, int input_layer_num,
                            fpga_layer **input_layers) {
  const int chunk_size = 8;
  unsigned short *dst =
      reinterpret_cast<unsigned short *>(layer.addr_cpu_output);
  const int x_size = layer.output_dim[0];
  const int y_size = layer.output_dim[1];
  const int dst_channel_size = layer.output_dim[2];
  const int chunk_stride = x_size * y_size * chunk_size;
  int dst_copied_size = 0;
  for (int i = 0; i < input_layer_num; i++) {
    unsigned short *src =
        reinterpret_cast<unsigned short *>(input_layers[i]->addr_cpu_output);
    const int src_channel_size = input_layers[i]->output_dim[2];
    int src_copied_size = 0;
    while (src_copied_size < src_channel_size) {
      int dst_copy_size = chunk_size - (dst_copied_size % chunk_size);
      if (dst_copy_size > dst_channel_size - dst_copied_size)
        dst_copy_size = dst_channel_size - dst_copied_size;
      int src_copy_size = chunk_size - (src_copied_size % chunk_size);
      if (src_copy_size > src_channel_size - src_copied_size)
        src_copy_size = src_channel_size - src_copied_size;
      int copy_size =
          (dst_copy_size < src_copy_size ? dst_copy_size : src_copy_size);
      unsigned short *dst_copy = dst +
                                 (dst_copied_size / chunk_size) * chunk_stride +
                                 (dst_copied_size % chunk_size);
      unsigned short *src_copy = src +
                                 (src_copied_size / chunk_size) * chunk_stride +
                                 (src_copied_size % chunk_size);
      const int dst_stride =
          (dst_channel_size / chunk_size > dst_copied_size / chunk_size
               ? chunk_size
               : dst_channel_size % chunk_size);
      const int src_stride =
          (src_channel_size / chunk_size > src_copied_size / chunk_size
               ? chunk_size
               : src_channel_size % chunk_size);
      for (int j = 0; j < x_size * y_size; j++) {
        memcpy(dst_copy, src_copy, sizeof(unsigned short) * copy_size);
        dst_copy += dst_stride;
        src_copy += src_stride;
      }
      dst_copied_size += copy_size;
      src_copied_size += copy_size;
    }
  }
}

#ifdef DUMP_LAYER_OUTPUT
bool dump_output = true;
#endif
void CDMP_Network::run_network(int t_sleep) {
  TimeInterval dt;

  for (unsigned int i = 0; i < num_layers; i++) {
    // printf("Running layer %d...\n", i);
    switch (layers[i].type) {
      case LT_CONV:
        dt.reset();
        // std::cout<<"---Layer: "<<i<<std::endl;
        run_conv(reinterpret_cast<top_conv_conf *>(layers[i].hw_conf), t_sleep);
        conv_layers[i].sw.output.performance = dt.get_us();
        break;
      case LT_FC:
        dt.reset();
        if (fc_layers.size() > 1) {
          load_IQ_table(fc_layers[i]);
        }
        run_ip(reinterpret_cast<top_fc_conf *>(layers[i].hw_conf));
        fc_layers[i].sw.performance = dt.get_us();
        break;
      case LT_SOFTMAX:
        run_softmax(layers[i], layers[i].softmax_axis);
        break;
      case LT_FLATTEN:
        run_flatten(layers[i]);
        break;
      case LT_COPY_CONCAT:
        run_copy_concat(layers[i], layers[i].input_layer_num,
                        layers[i].input_layers);
        break;
      case LT_CUSTOM:
        (*layers[i].custom_proc_ptr)(layers[i], layers[i].custom_param);
        break;
      case LT_INPUT:
      case LT_CONCAT:
        // Don't do anything
        break;
    }
#ifdef DUMP_LAYER_OUTPUT
    if (dump_output) {
      char output_file_name[64];
      sprintf(output_file_name, "layer%02d.bin", i);
      FILE *fout = fopen(output_file_name, "w");
      fwrite(layers[i].addr_cpu_output, 1, layers[i].output_size, fout);
      fclose(fout);
      if (i == 0) {
        fout = fopen("layer_input.bin", "w");
        fwrite(layers[i].addr_cpu_input, 1,
               layers[i].input_dim[0] * layers[i].input_dim[1] * 6, fout);
        fclose(fout);
      }
    }
#endif
  }
#ifdef DUMP_LAYER_OUTPUT
  dump_output = false;
#endif
}

void *CDMP_Network::get_network_input_addr_cpu() {
  return layers[0].addr_cpu_input;
}

void CDMP_Network::get_final_output(std::vector<float> &out, unsigned int i) {
  unsigned int output_size = output_layers[i]->output_size;
  if (output_layers[i]->is_f32_output)
    output_size /= 4;
  else
    output_size /= 2;

  if (out.size() != output_size) out.resize(output_size);

  if (output_layers[i]->is_f32_output)
    memcpy((void *)(&out.front()), output_layers[i]->addr_cpu_output,
           output_size * 4);
  else
    halfp2singles((void *)(&out.front()), output_layers[i]->addr_cpu_output,
                  output_size);
}

int CDMP_Network::get_convolution_performance(int layerID) {
  if (layerID == -1) {
    int perf = 0;
    for (unsigned int i = 0; i < conv_layers.size(); i++)
      perf += conv_layers[i].sw.output.performance;
    return perf;
  }
  if (conv_layers.size()) return get_conv_layer(layerID).sw.output.performance;

  return 0;
}

int CDMP_Network::get_innerproduct_performance(int layerID) {
  if (layerID == -1) {
    int perf = 0;
    for (unsigned int i = 0; i < fc_layers.size(); i++)
      perf += fc_layers[i].sw.performance;
    return perf;
  }
  if (fc_layers.size()) return get_ip_layer(layerID).sw.performance;

  return 0;
}

void get_layer_input(fpga_layer &layer, std::vector<float> &layer_input) {
  unsigned int input_size = 1;
  for (int i = 0; i < layer.input_dim_size; i++)
    input_size *= layer.input_dim[i];

  if (layer_input.size() != input_size) layer_input.resize(input_size);

  if (layer.is_input_hw_layout) {
    unsigned short *src =
        reinterpret_cast<unsigned short *>(layer.addr_cpu_input);
    unsigned short *dst =
        reinterpret_cast<unsigned short *>(&layer_input.front());
    int x_size = layer.input_dim[0];
    int y_size = layer.input_dim[1];
    int channel_size = layer.input_dim[2];
    dst += input_size;
    remap(src, dst, x_size, y_size, channel_size);
    halfp2singles((void *)(&layer_input.front()), dst, input_size);
  } else
    halfp2singles((void *)(&layer_input.front()), layer.addr_cpu_input,
                  input_size);
}

void put_layer_output(fpga_layer &layer, std::vector<float> &layer_output,
                      bool is_output_hw_layout) {
  unsigned int output_size = 1;
  for (int i = 0; i < layer.output_dim_size; i++)
    output_size *= layer.output_dim[i];

  if (!is_output_hw_layout) {
    memcpy(layer.addr_cpu_output, (void *)(&layer_output.front()),
           layer.output_size);
  } else {
    unsigned short *src =
        reinterpret_cast<unsigned short *>(&layer_output.front());
    unsigned short *dst =
        reinterpret_cast<unsigned short *>(layer.addr_cpu_output);
    int x_size = layer.output_dim[0];
    int y_size = layer.output_dim[1];
    int channel_size = layer.output_dim[2];
    dst += output_size;
    singles2halfp(dst + output_size, src, output_size);
    remap_hw(dst + output_size, dst, x_size, y_size, channel_size);
  }
}

void CDMP_Network::SetLayerAddresses() {
  for (unsigned int i = 0; i < num_conv_layers; i++) {
    conv_layers[i].hw.input.input_base_addr +=
        reserved_memory_addresses_fpga[1];
    conv_layers[i].hw.output.output_base_addr +=
        reserved_memory_addresses_fpga[1];
    if (conv_layers[i].hw.output.eltwise_base_addr != 0xDEADBEEF) {
      conv_layers[i].hw.output.eltwise_base_addr +=
          reserved_memory_addresses_fpga[1];
    }
    int run_num = conv_conf_num_runs(&conv_layers[i]);
    for (int j = 0; j < run_num; j++) {
      conv_layers[i].hw.run[j].weight_base_addr +=
          reserved_memory_addresses_fpga[0];
    }
  }
  for (unsigned int i = 0; i < num_fc_layers; i++) {
    fc_layers[i].sw.weight_addr = reinterpret_cast<void *>(
        fc_layers[i].hw.param_base_addr + reserved_memory_addresses_cpu[0]);
    fc_layers[i].hw.param_base_addr += reserved_memory_addresses_fpga[0];
    fc_layers[i].hw.weight_addr += reserved_memory_addresses_fpga[0];
    fc_layers[i].hw.bias_addr += reserved_memory_addresses_fpga[0];
    fc_layers[i].hw.input_base_addr += reserved_memory_addresses_fpga[1];
    fc_layers[i].hw.output_base_addr += reserved_memory_addresses_fpga[1];
  }
  for (unsigned int i = 0; i < num_layers; i++) {
    layers[i].addr_cpu_input = reinterpret_cast<void *>(
        reserved_memory_addresses_cpu[1] + layers[i].addr_offset_input);
    layers[i].addr_cpu_output = reinterpret_cast<void *>(
        reserved_memory_addresses_cpu[1] + layers[i].addr_offset_output);
  }

  weight_buffer_addr = (void *)reserved_memory_addresses_cpu[0];
}
