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
 * @brief Tests Fully-Connected layers.
 */
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <dirent.h>

#include <stdio.h>
#include <string.h>

#include <memory>
#include <set>
#include <vector>
#include <cmath>
#include <limits>
#include <tuple>
#include <algorithm>
#include <random>
#include <chrono>

#include <openssl/sha.h>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"


/// @brief Rounds up "a" to be the multiple of "n".
static inline int roundup(int a, int n) {
  int d = a % n;
  return d ? a + (n - d) : a;
}


/// @brief File which will store logs.
static FILE *g_flog = NULL;


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout); fprintf(g_flog, __VA_ARGS__); fflush(g_flog)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr); fprintf(g_flog, __VA_ARGS__); fflush(g_flog)


/// @brief Number of file descriptors for the process.
static int g_n_fd = -1;
#define N_T 6
static const float g_max_diff_bounds[N_T] = {0.1f, 0.5f, 1.0f, 5.0f, 10.0f, 1000000.0f};
static float g_max_diff[N_T] = {0, 0, 0, 0, 0, 0};
static float g_max_diff_y[N_T] = {0, 0, 0, 0, 0, 0};
static float g_max_diff_t[N_T] = {0, 0, 0, 0, 0, 0};


/// @brief Configuration description to be tested.
typedef struct fc_config_impl {
  int c_input, h_input, w_input, c_output, h_output, w_output, activation;
  bool hash_set;
  bool failed;
  uint8_t hash[32];
  dmp_dv_mem *io_mem, *weights_mem;
  size_t io_size, io_offs, weights_offs;
  __fp16 *io_ptr;
  __fp16 *caffe_output;
  bool pure_ints;  // only integers were used - error check should be exact

  bool operator <(const struct fc_config_impl& pt) const {
    return std::make_tuple(c_input, h_input, w_input, c_output, h_output, w_output, activation) <
        std::make_tuple(pt.c_input, pt.h_input, pt.w_input, pt.c_output, pt.h_output, pt.w_output, pt.activation);
  }
} fc_config;


/// @brief Checks if error is acceptable.
/// @param y Output to check.
/// @param t Target.
/// @param conf Executed convolutional configuration.
/// @return 0 if error is acceptable, non-zero otherwise.
int check_err(float y, float t, fc_config *conf) {
  const float diff = std::abs(y - t);
  float dmax;
  if (conf->pure_ints) {
    dmax = conf->activation ? 1.0e-3 : 1.0e-6;
    return diff < dmax ? 0 : -1;
  }

  const int n_adds = conf->c_input * conf->h_input * conf->w_input;

  const float ta = std::abs(t);

  if (ta < 0.09f) {
    dmax = 0.03f;
  }
  else {
    dmax = ta * 0.33f;
  }
  if (n_adds >= 5 * 5 * 64) {
    dmax *= std::sqrt((float)n_adds / (5 * 5 * 64)) * 1.4f;
  }

  return diff < dmax ? 0 : -1;
}


/// @brief Checks if an array has a float.
bool has_float(__fp16 *x, int n) {
  for (int i = 0; i < n; ++i) {
    float vle = (float)x[i];
    if (vle != (float)(int)vle) {
      return true;
    }
  }
  return false;
}


/// @brief Tests convolutional configurations for correctness using data from folder "data".
int test_fc(const std::vector<fc_config*>& confs) {
  char prefix[256];
  LOG("ENTER: test_fc: %d commands:", (int)confs.size());
  for (auto it = confs.begin(); it != confs.end(); ++it) {
    fc_config *conf = *it;
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d/%d_act%d",
             conf->c_input, conf->h_input, conf->w_input, conf->c_output * conf->h_output * conf->w_output, conf->activation);
    LOG(" %s", prefix);
  }
  LOG("\n");

  int result = -1;
  dmp_dv_context *ctx = NULL;
  dmp_dv_cmdlist *cmdlist = NULL;
  uint8_t *weights;
  size_t weights_size;
  float failed_diff = 0, failed_diff_y = 0, failed_diff_t = 0;
  int failed_i = -1;
  float caffe_a = std::numeric_limits<float>::max(), caffe_b = std::numeric_limits<float>::lowest();
  float dv_a = std::numeric_limits<float>::max(), dv_b = std::numeric_limits<float>::lowest();
  dmp_dv_cmdraw_fc_v0 cmd;
  char fnme[512];
  uint16_t quant_map[256];
  FILE *fin;
  std::vector<__fp16> caffe_input;
  std::vector<uint8_t> caffe_weights;
  std::vector<__fp16> caffe_bias;
  int n;
  char c;
  bool fend;
  int64_t exec_id;

  ctx  = dmp_dv_context_create();
  if (!ctx) {
    ERR("dmp_dv_context_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Successfully created context: %s\n", dmp_dv_context_get_info_string(ctx));

  cmdlist = dmp_dv_cmdlist_create(ctx);
  if (!cmdlist) {
    ERR("dmp_dv_cmdlist_create() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Created command list\n");

  // Outer loop by configurations to be packed in the single command list
  for (auto it = confs.begin(); it != confs.end(); ++it) {
    fc_config *conf = *it;
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d/%d_act%d",
             conf->c_input, conf->h_input, conf->w_input, conf->c_output * conf->h_output * conf->w_output, conf->activation);

    const int input_size = conf->c_input * conf->h_input * conf->w_input;
    const int output_size = conf->c_output * conf->h_output * conf->w_output;

    // Load quantization map
    snprintf(fnme, sizeof(fnme), "%s.q.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    n = fread(quant_map, sizeof(quant_map[0]), sizeof(quant_map) / sizeof(quant_map[0]), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != 256) {
      ERR("fread() returned %d while expecting %d for %s\n", n, 256, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Load input
    snprintf(fnme, sizeof(fnme), "%s.i.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }

    caffe_input.resize(input_size);
    n = fread(caffe_input.data(), sizeof(caffe_input[0]), caffe_input.size(), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != input_size) {
      ERR("fread() returned %d while expecting %d for %s\n", n, input_size, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Find if we are using only integers or not
    conf->pure_ints = ((!has_float((__fp16*)quant_map, 256) && (!has_float(caffe_input.data(), caffe_input.size()))));

    // Load weights
    snprintf(fnme, sizeof(fnme), "%s.w.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    caffe_weights.resize(input_size * output_size);
    n = fread(caffe_weights.data(), sizeof(caffe_weights[0]), caffe_weights.size(), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != input_size * output_size) {
      ERR("fread() returned %d while expecting %d for %s\n",
          n, input_size * output_size, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Load bias
    snprintf(fnme, sizeof(fnme), "%s.b.bin", prefix);
    fin = fopen(fnme, "rb");
    if (!fin) {
      ERR("fopen() failed for %s\n", fnme);
      goto L_EXIT;
    }
    caffe_bias.resize(output_size);
    n = fread(caffe_bias.data(), sizeof(caffe_bias[0]), caffe_bias.size(), fin);
    fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
    fclose(fin);
    if (n != output_size) {
      ERR("fread() returned %d while expecting %d for %s\n", n, output_size, fnme);
      goto L_EXIT;
    }
    if (!fend) {
      ERR("File is bigger than expected: %s\n", fnme);
      goto L_EXIT;
    }

    // Load output
    if (!conf->hash_set) {
      snprintf(fnme, sizeof(fnme), "%s.o.bin", prefix);
      fin = fopen(fnme, "rb");
      if (!fin) {
        ERR("fopen() failed for %s\n", fnme);
        goto L_EXIT;
      }
      conf->caffe_output = (__fp16*)malloc(output_size * sizeof(__fp16));
      n = fread(conf->caffe_output, sizeof(__fp16), output_size, fin);
      fend = feof(fin) || ((fread(&c, 1, 1, fin) == 0) && (feof(fin)));
      fclose(fin);
      if (n != output_size) {
        ERR("fread() returned %d while expecting %d for %s\n", n, output_size, fnme);
        goto L_EXIT;
      }
      if (!fend) {
        ERR("File is bigger than expected: %s\n", fnme);
        goto L_EXIT;
      }
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.header.size = sizeof(cmd);
    cmd.header.device_type = DMP_DV_DEV_FC;
    cmd.header.version = 0;
    cmd.input_size = input_size;
    cmd.output_size = output_size;
    cmd.actfunc = conf->activation;

    conf->io_size = (roundup(input_size, 4) + roundup(output_size, 4)) * sizeof(__fp16);
    conf->io_mem = dmp_dv_mem_alloc(ctx, conf->io_offs + conf->io_size);
    if (!conf->io_mem) {
      ERR("dmp_dv_mem_alloc() failed for %zu bytes: %s\n", conf->io_offs + conf->io_size, dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    LOG("Allocated %zu (%zu(+%zu random offset) requested) bytes for input/output\n", dmp_dv_mem_get_size(conf->io_mem), conf->io_size, conf->io_offs);
    cmd.input_buf.mem = conf->io_mem;
    cmd.input_buf.offs = conf->io_offs;
    cmd.output_buf.mem = conf->io_mem;
    cmd.output_buf.offs = conf->io_offs + roundup(input_size, 4) * sizeof(__fp16);

    weights_size = 0;
    if (dmp_dv_pack_fc_weights(
            conf->c_input, conf->h_input, conf->w_input,
            conf->c_output, conf->h_output, conf->w_output,
            quant_map, NULL, NULL, NULL, &weights_size)) {
      ERR("dmp_dv_pack_fc_weights() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    conf->weights_mem = dmp_dv_mem_alloc(ctx, conf->weights_offs + weights_size);
    if (!conf->weights_mem) {
      ERR("dmp_dv_mem_alloc() failed for %zu bytes: %s\n", weights_size, dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    LOG("Allocated %zu (%zu(+%zu random offset) requested) bytes for weights\n", dmp_dv_mem_get_size(conf->weights_mem), weights_size, conf->weights_offs);
    cmd.weight_buf.mem = conf->weights_mem;
    cmd.weight_buf.offs = conf->weights_offs;
    cmd.weight_fmt = 1;

    weights = dmp_dv_mem_map(conf->weights_mem) + conf->weights_offs;
    if (!weights) {
      ERR("dmp_dv_mem_map() failed for weights: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_mem_sync_start(conf->weights_mem, 0, 1)) {
      ERR("dmp_dv_mem_sync_start() failed for weights: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    // Fill weights
    if (dmp_dv_pack_fc_weights(
          conf->c_input, conf->h_input, conf->w_input,
          conf->c_output, conf->h_output, conf->w_output,
          quant_map, caffe_weights.data(), (const uint16_t*)caffe_bias.data(), weights, &weights_size)) {
      ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_mem_sync_end(conf->weights_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for weights: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    dmp_dv_mem_unmap(conf->weights_mem);

    conf->io_ptr = (__fp16*)(dmp_dv_mem_map(conf->io_mem) + conf->io_offs);
    if (!conf->io_ptr) {
      ERR("dmp_dv_mem_map() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    if (dmp_dv_mem_sync_start(conf->io_mem, 0, 1)) {
      ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
    memset(conf->io_ptr, 0, conf->io_size);

    if ((conf->h_input == 1) && (conf->w_input == 1)) {
      memcpy(conf->io_ptr, caffe_input.data(), input_size * 2);
    }
    else {
      // Caffe's input is stored as channel, height, width
      // DV input should be stored as chunks by max of 8 channels as width, height, channel
      for (int chan_group = 0, o_offs = 0; chan_group < conf->c_input; chan_group += 8) {
        const int last_chan = std::min(chan_group + 8, conf->c_input);
        for (int i = 0; i < conf->w_input; ++i) {
          for (int j = 0; j < conf->h_input; ++j) {
            for (int k = chan_group; k < last_chan; ++k, ++o_offs) {
              const int i_offs = k * conf->h_input * conf->w_input + j * conf->w_input + i;
              const __fp16 vle = caffe_input[i_offs];
              conf->io_ptr[o_offs] = vle;
            }
          }
        }
      }
    }

    if (dmp_dv_mem_sync_end(conf->io_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    if (dmp_dv_cmdlist_add_raw(cmdlist, (dmp_dv_cmdraw*)&cmd)) {
      ERR("dmp_dv_cmdlist_add_raw() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
  }

  if (dmp_dv_cmdlist_commit(cmdlist)) {
    ERR("dmp_dv_cmdlist_commit() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Commited the command list\n");

  exec_id = dmp_dv_cmdlist_exec(cmdlist);
  if (exec_id < 0) {
    ERR("dmp_dv_cmdlist_exec() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Scheduled command list for execution\n");

  LOG("Waiting for completion\n");
  if (dmp_dv_cmdlist_wait(cmdlist, exec_id)) {
    ERR("dmp_dv_sync() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Execution has completed\n");

  for (auto it = confs.begin(); it != confs.end(); ++it) {
    fc_config *conf = *it;

    const int input_size = conf->c_input * conf->h_input * conf->w_input;
    const int output_size = conf->c_output * conf->h_output * conf->w_output;

    if (dmp_dv_mem_sync_start(conf->io_mem, 1, 0)) {
      ERR("dmp_dv_mem_sync_start() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    // Compare output with the gold one
    const int o_offs = roundup(input_size, 4);
    if (conf->hash_set) {  // check hash
      uint8_t hash[32];
      SHA256_CTX sha256;
      SHA256_Init(&sha256);
      SHA256_Update(&sha256, conf->io_ptr + o_offs, output_size * sizeof(__fp16));
      SHA256_Final(hash, &sha256);
      if (memcmp(conf->hash, hash, 32)) {
        ERR("Hash differs\n");
        goto L_EXIT;
      }
    }
    else {  // approximately compare output
      // Caffe's output is stored as channels, height, width
      // DV output is stored as chunks by max of 8 channels as width, height, channel
      float max_diff[N_T], max_diff_y[N_T], max_diff_t[N_T];
      memset(max_diff, 0, sizeof(max_diff));
      memset(max_diff_y, 0, sizeof(max_diff_y));
      memset(max_diff_t, 0, sizeof(max_diff_t));
      for (int i = 0; i < output_size; ++i) {
        const __fp16 vle = conf->caffe_output[i];
        const float y = (float)conf->io_ptr[o_offs + i], t = (float)vle;
        caffe_a = std::min(caffe_a, t);
        caffe_b = std::max(caffe_b, t);
        dv_a = std::min(dv_a, y);
        dv_b = std::max(dv_b, y);
        const float diff = std::abs(y - t);
        if (check_err(y, t, conf)) {
          conf->failed = true;
          if (diff > failed_diff) {
            failed_diff = diff;
            failed_diff_y = y;
            failed_diff_t = t;
            failed_i = i;
          }
        }
        else {
          for (int ii = 0; ii < N_T; ++ii) {
            if (std::abs(t) > g_max_diff_bounds[ii]) {
              continue;
            }
            if (diff > max_diff[ii]) {
              max_diff[ii] = diff;
              max_diff_y[ii] = y;
              max_diff_t[ii] = t;
            }
            break;
          }
        }
      }
      LOG("caffe: [%.6f, %.6f] dv: [%.6f, %.6f]\n", caffe_a, caffe_b, dv_a, dv_b);
      for (int i = 0; i < N_T; ++i) {
        LOG("t <= %.1f: max_diff=%.6f on y=%.6f and t=%.6f\n",
            g_max_diff_bounds[i], max_diff[i], max_diff_y[i], max_diff_t[i]);
      }
      for (int i = 0; i < N_T; ++i) {
        if (max_diff[i] > g_max_diff[i]) {
          g_max_diff[i] = max_diff[i];
          g_max_diff_y[i] = max_diff_y[i];
          g_max_diff_t[i] = max_diff_t[i];
        }
      }
      if (failed_diff > 0.0f) {
        ERR("FAILED: failed_diff=%.6f on y=%.6f and t=%.6f i=%d %s\n", failed_diff, failed_diff_y, failed_diff_t,
            failed_i, prefix);
        goto L_EXIT;
      }
    }
    if (!conf->failed) {  // compute hash
      const int n_bytes = output_size * sizeof(__fp16);
      if (o_offs * sizeof(__fp16) + n_bytes > conf->io_size) {
        ERR("Incorrect allocation size for input/output");
        _exit(-1);
      }
      SHA256_CTX sha256;
      SHA256_Init(&sha256);
      SHA256_Update(&sha256, conf->io_ptr + o_offs, n_bytes);
      SHA256_Final(&conf->hash[0], &sha256);
      conf->hash_set = true;
    }
    if (dmp_dv_mem_sync_end(conf->io_mem)) {
      ERR("dmp_dv_mem_sync_end() failed for input/output: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }
  }

  result = 0;
  LOG("SUCCESS: test_cmdlist\n");

  L_EXIT:
  dmp_dv_cmdlist_release(cmdlist);
  for (auto it = confs.rbegin(); it != confs.rend(); ++it) {
    fc_config *conf = *it;
    if (conf->caffe_output) {
      free(conf->caffe_output);
      conf->caffe_output = NULL;
    }
    dmp_dv_mem_release(conf->io_mem);
    conf->io_mem = NULL;
    conf->io_ptr = NULL;
    conf->io_size = 0;
    dmp_dv_mem_release(conf->weights_mem);
    conf->weights_mem = NULL;
  }
  dmp_dv_context_release(ctx);

  int n_fd = 0;
  DIR *d;
  struct dirent *dir;
  d = opendir("/proc/self/fd");
  if (!d) {
    ERR("Could not open \"/proc/self/fd\" folder\n");
    return -1;
  }
  while ((dir = readdir(d))) {
    char *fnme = dir->d_name;
    bool num = true;
    for (; *fnme; ++fnme) {
      if ((*fnme >= '0') && (*fnme <= '9')) {
        continue;
      }
      num = false;
      break;
    }
    if (num) {
      ++n_fd;
    }
  }
  closedir(d);

  if (g_n_fd == -1) {
    g_n_fd = n_fd;
  }
  if (n_fd != g_n_fd) {
    ERR("Inconsistent file descriptor count detected, memory leak is probable\n");
    result = -1;
  }

  LOG("EXIT: test_fc: %d commands, %d FDs:", (int)confs.size(), n_fd);
  for (auto it = confs.begin(); it != confs.end(); ++it) {
    fc_config *conf = *it;
    snprintf(prefix, sizeof(prefix), "data/%dx%dx%d/%d_act%d",
             conf->c_input, conf->h_input, conf->w_input, conf->c_output * conf->h_output * conf->w_output, conf->activation);
    LOG(" %s", prefix);
  }
  LOG("\n");

  return result;
}


/// @brief Entry point.
/// @details All messages will be logged to filename argv[1] if argc > 1.
int main(int argc, char **argv) {
  g_flog = fopen(argc > 1 ? argv[1] : "/dev/null", "w");
  if (!g_flog) {
    fprintf(stderr, "fopen() failed for %s\n", argv[1]);
    fflush(stderr);
    return -1;
  }

  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  std::shared_ptr<std::set<fc_config> > config_set = std::make_shared<std::set<fc_config> >();

  DIR *d, *subd;
  struct dirent *dir, *subdir;
  d = opendir("data");
  if (!d) {
    ERR("Could not open \"data\" folder\n");
    return -1;
  }
  while ((dir = readdir(d))) {
    fc_config config;
    memset(&config, 0, sizeof(config));

    char fnme[512];
    for (int i = 0; ; ) {
      char c = dir->d_name[i];
      fnme[i] = ((c >= '0') && (c <= '9')) ? c : ' ';
      ++i;
      if (!dir->d_name[i]) {
        fnme[i] = 0;
        break;
      }
    }
    if (sscanf(fnme, "%d%d%d", &config.c_input, &config.h_input, &config.w_input) != 3) {
      continue;
    }
    snprintf(fnme, sizeof(fnme), "data/%s", dir->d_name);
    subd = opendir(fnme);
    if (!subd) {
      continue;
    }

    while ((subdir = readdir(subd))) {
      for (int i = 0; ; ) {
        char c = subdir->d_name[i];
        fnme[i] = ((c >= '0') && (c <= '9')) ? c : ' ';
        ++i;
        if (!subdir->d_name[i]) {
          fnme[i] = 0;
          break;
        }
      }

      config.h_output = 1;
      config.w_output = 1;
      if (sscanf(fnme, "%d%d", &config.c_output, &config.activation) != 2) {
        continue;
      }

      char prefix[256];
      snprintf(prefix, sizeof(prefix), "data/%dx%dx%d/%d_act%d",
               config.c_input, config.h_input, config.w_input, config.c_output * config.h_output * config.w_output, config.activation);

      const size_t prev_size = (int)config_set->size();
      config_set->emplace(config);
      const size_t this_size = (int)config_set->size();

      if ((this_size != prev_size) && (!(this_size % 1000))) {
        LOG("Loaded %zu configurations\n", this_size);
      }
    }
    closedir(subd);
  }
  closedir(d);
  LOG("Loaded %zu configurations\n", config_set->size());

  // Copy configurations from the set to obtain fixed memory pointers to elements
  std::vector<fc_config> config_data;
  for (auto it = config_set->cbegin(); it != config_set->cend(); ++it) {
    config_data.push_back(*it);
  }
  config_set->clear();
  config_set.reset();  // delete the set

  const int n_configs = (int)config_data.size();
  std::vector<fc_config*> configs;
  configs.resize(n_configs);
  for (int i = 0; i < n_configs; ++i) {
    configs[i] = config_data.data() + i;
  }

  std::mt19937 mt_rand(time(NULL));
  std::uniform_int_distribution<int> rnd_offs(0, 1024);

  const int n_passes = 2;
  for (int i_pass = 0; i_pass < n_passes; ++i_pass) {
    // Randomize configrations order
    std::shuffle(configs.begin(), configs.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    // Execute configurations in different chunk sizes
    const size_t pack_sizes[2] = {1, 50};
    const int n_packs = 2;
    for (int i_pack = 0; i_pack < n_packs; ++i_pack) {
      std::vector<fc_config*> confs;
      int i_config = 0;
      for (auto it = configs.begin(); it != configs.end(); ++it, ++i_config) {
        fc_config *conf = *it;
        if (conf->failed) {
          continue;
        }
        conf->io_offs = rnd_offs(mt_rand) << 4;
        conf->weights_offs = rnd_offs(mt_rand) << 4;
        confs.push_back(conf);
        if (confs.size() < pack_sizes[i_pack]) {
          continue;
        }
        res = test_fc(confs);
        if (res) {
          ++n_err;
        }
        else {
          ++n_ok;
        }
        confs.clear();
      }
      if (confs.size()) {
        res = test_fc(confs);
        if (res) {
          ++n_err;
        }
        else {
          ++n_ok;
        }
      }
    }
  }

  for (int i = 0; i < N_T; ++i) {
    LOG("t <= %.1f: g_max_diff=%.6f on y=%.6f and t=%.6f\n",
        g_max_diff_bounds[i], g_max_diff[i], g_max_diff_y[i], g_max_diff_t[i]);
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);
  LOG("Number of configurations tested: %d\n", (int)configs.size());

  fclose(g_flog);

  return n_err;
}
