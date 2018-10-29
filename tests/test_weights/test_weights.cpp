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
 * @brief Tests for weights packing.
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

#include <openssl/sha.h>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"
#include "../common/stats.h"
#include "../../dv-kernel-driver/uapi/dimensions.h"


#define LOG(...) fprintf(stdout, __VA_ARGS__); fflush(stdout)
#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


/// @brief CPU frequency.
static double clocks_per_ms = 600000.0;  // assume CPU runs at least at 600 MHz


/* The state array must be initialized to not be all zero */
uint32_t xorshift128(uint32_t state[4]) {
    /* Algorithm "xor128" from p. 5 of Marsaglia, "Xorshift RNGs" */
    uint32_t s, t = state[3];
    t ^= t << 11;
    t ^= t >> 8;
    state[3] = state[2]; state[2] = state[1]; state[1] = s = state[0];
    t ^= s;
    t ^= s >> 19;
    state[0] = t;
    return t;
}


/// @brief Half floats used in test (uniform in [-1, 1]).
static const uint16_t valid_floats[256] = {
    0, 14249, 13806, 47192, 14461, 12825, 14256, 15260, 47742,
    14349, 14862, 14781, 11943, 48047, 44506, 10491, 12801, 44023,
    15000, 11521, 37940, 47775, 47844, 13322, 12841, 48012, 46678,
    47158, 10691, 15296, 45887, 44346, 46028, 43918, 47876, 45657,
    15294, 15265, 14684, 15337, 44426, 47338, 47941, 41546, 47891,
    15086, 13759, 47929, 15331, 47152, 47067, 14598, 46890,  9515,
    14989, 15181, 47345, 47567, 14310, 14702, 46163, 47710, 15177,
    14769, 44121, 10401, 45249, 14446, 15149, 15338, 12361, 47419,
    46509, 15317, 14530, 14534, 13729, 44317, 14663, 15354, 47400,
    44544, 48004, 46658, 46946, 15129, 44006, 14257, 10093, 47363,
    48075, 47713, 12068, 13237, 47512, 15215, 45544, 47685, 12603,
    14876, 42069, 47286, 47629, 46211, 14600, 46347, 14621, 14570,
    46489, 12440, 13645, 14558, 13349, 13619, 47359, 15318, 47981,
    44117, 47162, 13673, 44761, 47630, 47743, 15007, 47686, 47755,
    44436, 47909, 13723, 14103, 14321, 46936, 45528, 14375, 14377,
    12445, 47132, 42341, 14693, 46193, 14717, 14547, 47847, 46309,
    45088, 15270, 42764, 47601, 48063, 46709, 11819, 44506, 47612,
    14047, 47579, 10633, 14996, 13390, 47361, 14479, 14233, 47148,
    14372, 47875, 47505, 47532, 15166, 14597, 46819, 47288, 10735,
    13007, 40891, 37194, 13637, 48072, 47204, 47983, 47299, 13286,
    47590, 47761, 46093, 46572, 47246, 47480, 14362, 47181, 47687,
    12599, 15036, 47269, 46527, 13677, 48112, 11607, 13685, 47200,
    44771, 46303, 15176, 46612, 15269, 45363, 15155, 47039, 46750,
    13870, 14534, 15087, 14966, 12323, 47154, 14496, 47561, 47308,
    45809, 47602, 15096, 14784, 15024, 14515, 13411, 12563, 46854,
    48021, 13754, 45794, 47789, 13626, 47205, 14117, 14300, 45514,
    46410, 47210, 12741, 47218, 46168,  6839, 11508, 46528, 14784,
    47346, 46640, 14373, 47607, 13478, 13922, 45830, 13773, 13734,
    12359, 13764, 14442, 13234
};


int test_weights(uint32_t state[4], const char *s_gold_hash,
                 const uint16_t quant_map[256], int n_channels, int kx, int ky, int n_kernels,
                 int prelu) {
  int result = -1;
  char prefix[64];
  snprintf(prefix, sizeof(prefix), "(%d, %d, %d, %d)", n_kernels, n_channels, ky, kx);
  LOG("ENTER: test_weights: %s\n", prefix);

  std::vector<uint8_t> caffe_weights8;
  std::vector<uint16_t> caffe_weights16;
  if (quant_map) {
    caffe_weights8.resize(n_kernels * n_channels * ky * kx);
  }
  else {
    caffe_weights16.resize(n_kernels * n_channels * ky * kx);
  }

  std::vector<uint16_t> bias;
  bias.resize(n_kernels);

  std::vector<uint16_t> prelu_vals;
  prelu_vals.resize(n_kernels);

  for (int i = 0; i < n_kernels; ++i) {
    bias[i] = valid_floats[xorshift128(state) >> 24];
  }
  if (prelu) {
    for (int i = 0; i < n_kernels; ++i) {
      prelu_vals[i] = valid_floats[xorshift128(state) >> 24];
    }
  }
  const int n_caffe_weights = n_kernels * n_channels * ky * kx;
  if (quant_map) {
    for (int i = 0; i < n_caffe_weights; ++i) {
      caffe_weights8[i] = xorshift128(state) >> 24;
    }
  }
  else {
    for (int i = 0; i < n_caffe_weights; ++i) {
      caffe_weights16[i] = valid_floats[xorshift128(state) >> 24];
    }
  }

  std::vector<uint8_t> weights;
  size_t weights_size = 0, kweights_size = 0;

  if (dmp_dv_pack_conv_weights(
        n_channels, kx, ky, n_kernels,
        quant_map, NULL, NULL, prelu ? prelu_vals.data() : NULL, NULL, &weights_size)) {
    ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  LOG("Required %zu bytes for weights\n", weights_size);
  kweights_size = get_weight_size(n_channels, n_kernels, std::max(kx, ky) | 1, quant_map ? 1 : 0, 0, prelu);
  if (kweights_size != weights_size) {
    ERR("Kernel module function get_weight_size() returned %zu while user-space function dmp_dv_pack_conv_weights() returned %zu\n",
        kweights_size, weights_size);
    goto L_EXIT;
  }
  weights.resize(weights_size);
  if (weights.size() != weights_size) {
    ERR("Failed to allocated %zu bytes of memory\n", weights_size);
    goto L_EXIT;
  }

  // Fill weights
  {
    TimeIntervalThread dt;
    if (dmp_dv_pack_conv_weights(
          n_channels, kx, ky, n_kernels,
          quant_map, quant_map ? caffe_weights8.data() : (uint8_t*)caffe_weights16.data(),
          bias.data(), prelu ? prelu_vals.data() : NULL, weights.data(), &weights_size)) {
      ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
      goto L_EXIT;
    }

    const double dt_ms = dt.get_ms();
    const double n_elems = n_channels * kx * ky * n_kernels;
    // Complexity should be O(n)
    const double max_C = 25.0;  // maximum O(n) constant
    const double elems_per_ms = n_elems / dt_ms;
    const double clocks_per_elem = clocks_per_ms / elems_per_ms;
    if (clocks_per_elem > max_C) {
      int c = 1;
      double clocks = clocks_per_elem;
      for (; clocks > n_elems; clocks /= n_elems) {
        ++c;
      }
      ERR("Complexity O(n^%d) is detected with constant: %.0f\n", c, clocks);
      --result;
    }
  }

  // Compute hash
  {
    uint8_t hash[32];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, weights.data(), weights_size);
    SHA256_Final(hash, &sha256);

    static const char h_digits[16] = {
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'
    };
    char s_hash[65];
    for (int i = 0; i < 32; ++i) {
      s_hash[(i << 1) + 0] = h_digits[hash[i] >> 4];
      s_hash[(i << 1) + 1] = h_digits[hash[i] & 15];
    }
    s_hash[64] = 0;
    if (strcmp(s_hash, s_gold_hash)) {
      ERR("Packed weights hash differ: got %s while expecting %s\n", s_hash, s_gold_hash);
      --result;
    }
  }

  if (result == -1) {
    result = 0;
    LOG("SUCCESS: test_weights\n");
  }

  L_EXIT:

  LOG("EXIT: test_weights: %s\n", prefix);
  return result;
}


int main(int argc, char **argv) {
  FILE *fin = fopen("/proc/cpuinfo", "r");
  if (fin) {
    char s[256];
    s[255] = 0;
    while (!feof(fin)) {
      if (!fgets(s, sizeof(s) - 1, fin)) {
        break;
      }
      if (!strncasecmp(s, "BogoMIPS", 8)) {
        int n = strlen(s);
        while ((n >= 0) && (!(((s[n] >= '0') && (s[n] <= '9')) || (s[n] == '.')))) {
          s[n] = 0;
          --n;
        }
        while ((n >= 0) && (((s[n] >= '0') && (s[n] <= '9')) || (s[n] == '.'))) {
          --n;
        }
        ++n;
        double vle = atof(&s[n]);
        LOG("BogoMIPS = %.3f\n", vle);
        clocks_per_ms = std::max(clocks_per_ms, vle * 1000.0);
        break;
      }
    }
    fclose(fin);
  }
  LOG("clocks_per_ms = %.3f\n", clocks_per_ms);

  int n_ok = 0;
  int n_err = 0;
  int res = 0;

  #define N_CONFIGS 32
  struct config {
    const uint16_t *quant_map;
    uint32_t state[4];
    const char *s_gold_hash;
    int n_channels, kx, ky, n_kernels;
    int prelu;
  } configs[N_CONFIGS] = {
      {valid_floats, {1, 2, 3, 4}, "3781796B12E74C43C2313DC74D3EA4C25D0F3D19AB8FBAA64BB17362F9080A79", 256, 1, 1, 512, 0},
      {valid_floats, {1, 2, 3, 4}, "21CC18D3A0183D383C8E0E5EDEBC6693A4F8F9C7F625C5ED34596AA6637EA93B", 128, 3, 3, 256, 0},
      {valid_floats, {1, 2, 3, 4}, "98E85214A6BA955B4699693EFBCE1DFE2A1A549C4217E5DACD1FE5F45790CA46", 64, 5, 5, 128, 0},
      {valid_floats, {1, 2, 3, 4}, "D54DB09C1F722255470D8E546F378CB7BCFE1C240BFEF6A7EE39CE1A6FC759CF", 64, 7, 7, 128, 0},
      {valid_floats, {1, 2, 3, 4}, "C220BF05D85BB7C0B5D1CDD7EDF614796F297EFD7BE4CFB63E51D93009C37C21", 260, 1, 1, 510, 0},
      {valid_floats, {1, 2, 3, 4}, "4BAE08599903424BE6634F3EDC0804E78328DBE961014CB608EDB13B704D66ED", 70, 3, 3, 130, 0},
      {valid_floats, {1, 2, 3, 4}, "8AF1E3CAA7BA35BC009C8B79520E1F0EA33C5F9469FD2FD25DC617A505AADF69", 70, 5, 5, 130, 0},
      {valid_floats, {1, 2, 3, 4}, "934AE39C6401F631F28C746E2047C9A69890C8D36E0085ABA93B7556A9E17BB4", 70, 7, 7, 130, 0},

      {NULL, {1, 2, 3, 4}, "B9CC04250D601B1699D88E20BBC665DE1D34A440734812F403B2FDA18713C954", 256, 1, 1, 512, 0},
      {NULL, {1, 2, 3, 4}, "0D15B90F0B236F73532965A79329CC367E8FB21B48A7E29A9D49A78C122D968F", 128, 3, 3, 256, 0},
      {NULL, {1, 2, 3, 4}, "786D1ACE1DF3DA407100EA1E6944D8A3A838524FF9A38C7EF4514A9F88619B25", 64, 5, 5, 128, 0},
      {NULL, {1, 2, 3, 4}, "AD19782CEAA632C68ADD811AB651E93CCB7F6719FCAA9BB0B993DC364AB1BD52", 64, 7, 7, 128, 0},
      {NULL, {1, 2, 3, 4}, "C9115AB33F0E31FA0E4DD6C040AA851AB27D1330EDEF0E18CAA711C5FF27B3F7", 260, 1, 1, 510, 0},
      {NULL, {1, 2, 3, 4}, "6B5DDACE58F9BCB3DC39556976813CA4AAB70A63E3DD4233556439158BA94A8D", 70, 3, 3, 130, 0},
      {NULL, {1, 2, 3, 4}, "6620E41CE8B6828F4200BBE87AD03C3C8E5CF59AAB858DF03EF6D992FBE2A2B0", 70, 5, 5, 130, 0},
      {NULL, {1, 2, 3, 4}, "33F10592BB98FA01E5918BABE4B1C23C19B5895DDF99B660C8413FD46D5F0DC1", 70, 7, 7, 130, 0},

      {valid_floats, {1, 2, 3, 4}, "5C91C8EEE9D70BAB9F7F5BAEFF7B64ED6E318D180667E3BBECE45BB565FDB32F", 256, 1, 1, 512, 1},
      {valid_floats, {1, 2, 3, 4}, "98AEDB4E7308537BC5BFF055F5D979C73BD7A6EFA7D5D36F0F69A02800A2889A", 128, 3, 3, 256, 1},
      {valid_floats, {1, 2, 3, 4}, "59488B6F6BD41C89D0680F41F36CBC054E4B2F720DBB63BD7862427D2E9B788A", 64, 5, 5, 128, 1},
      {valid_floats, {1, 2, 3, 4}, "44C1DA142CBDF30BCC87C7ED3C2E9498FF2922256BF323C5BDBF89D535ABC3B2", 64, 7, 7, 128, 1},
      {valid_floats, {1, 2, 3, 4}, "1D5CDB91C50B90BA2F7A426A89C26642C5415361B9ECE17896CCD3CE39080E2F", 260, 1, 1, 510, 1},
      {valid_floats, {1, 2, 3, 4}, "CF7F7FB9F7C166ED8DF513841E34E6FA3579E19699CE6687EFED6B993E75BC71", 70, 3, 3, 130, 1},
      {valid_floats, {1, 2, 3, 4}, "723E66992EB07DC62FA70A639BDC98F2626EC1203C79699B0A7A9B413AEC4AD4", 70, 5, 5, 130, 1},
      {valid_floats, {1, 2, 3, 4}, "4DEFF737F144151C43D2D0274BD25477D9A3DC1552859D1E0368EE1EB6359800", 70, 7, 7, 130, 1},

      {NULL, {1, 2, 3, 4}, "176F2CA4A15B6C3AB0B637EA1E7EB11B4A352027C362874A3D0404B903C242F2", 256, 1, 1, 512, 1},
      {NULL, {1, 2, 3, 4}, "38F2F4826D216F4F71F265AB39770ED9C43ACEF07C80AA0ECF23ECD30287C28D", 128, 3, 3, 256, 1},
      {NULL, {1, 2, 3, 4}, "E53FC40659C422169560A1DF5460A02155D41CFDE1ADEEE34B8FAC34A27C9FF8", 64, 5, 5, 128, 1},
      {NULL, {1, 2, 3, 4}, "A19879EE2681EAA36614CF70B4F1ABE3A18626177A17EDE0C469D83DCA43249D", 64, 7, 7, 128, 1},
      {NULL, {1, 2, 3, 4}, "F3152CCA0B2D5F67001389BA9588887A922C9458A52911E1E98DE96EE4A3004E", 260, 1, 1, 510, 1},
      {NULL, {1, 2, 3, 4}, "FDED78999ED1F3C1CCD89CB63164CED4B90FF00994E8E5B852CF2C7FBC80E8A5", 70, 3, 3, 130, 1},
      {NULL, {1, 2, 3, 4}, "E77264BFB5D5EB95A91A7DE71AB591C0F10173F916F56207887C37FE672ADEA6", 70, 5, 5, 130, 1},
      {NULL, {1, 2, 3, 4}, "B1DDB1FB9FE4F57788E13E1451330D75E50359A34C541C3C3DFD2130EC812254", 70, 7, 7, 130, 1},
  };

  for (int i = 0; i < N_CONFIGS; ++i) {
    res = test_weights(configs[i].state, configs[i].s_gold_hash,
                       configs[i].quant_map, configs[i].n_channels, configs[i].kx, configs[i].ky, configs[i].n_kernels,
                       configs[i].prelu);
    if (res) {
      ++n_err;
    }
    else {
      ++n_ok;
    }
  }

  LOG("Tests succeeded: %d\n", n_ok);
  LOG("Tests failed: %d\n", n_err);
  return n_err;
}
