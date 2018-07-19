/*
*------------------------------------------------------------
* Copyright(c) 2018 by Digital Media Professionals Inc.
* All rights reserved.
*------------------------------------------------------------
* The code is licenced under Apache License, Version 2.0
*------------------------------------------------------------
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


int test_weights(uint32_t state[4], const char *s_gold_hash,
                 const uint16_t quant_map[256], int n_channels, int kx, int ky, int n_kernels) {
  int result = -1;
  char prefix[64];
  snprintf(prefix, sizeof(prefix), "(%d, %d, %d, %d)", n_kernels, n_channels, ky, kx);
  LOG("ENTER: test_weights: %s\n", prefix);

  std::vector<uint16_t> caffe_weights;
  caffe_weights.resize(n_kernels * n_channels * ky * kx);

  std::vector<uint16_t> bias;
  bias.resize(n_kernels);

  for (int i = 0; i < n_kernels; ++i) {
    bias[i] = quant_map[xorshift128(state) >> 24];
  }
  const int n_caffe_weights = n_kernels * n_channels * ky * kx;
  for (int i = 0; i < n_caffe_weights; ++i) {
    caffe_weights[i] = quant_map[xorshift128(state) >> 24];
  }

  std::vector<uint8_t> weights;
  size_t weights_size = 0;

  if (dmp_dv_pack_conv_weights(
        n_channels, kx, ky, n_kernels,
        quant_map, NULL, NULL, NULL, &weights_size)) {
    ERR("dmp_dv_pack_conv_weights() failed: %s\n", dmp_dv_get_last_error_message());
    goto L_EXIT;
  }
  weights.resize(weights_size);
  if (weights.size() != weights_size) {
    ERR("Failed to allocated %zu bytes of memory\n", weights_size);
    goto L_EXIT;
  }
  LOG("Allocated %zu bytes for weights\n", weights_size);

  // Fill weights
  {
    TimeIntervalThread dt;
    if (dmp_dv_pack_conv_weights(
          n_channels, kx, ky, n_kernels,
          quant_map, caffe_weights.data(), bias.data(), weights.data(), &weights_size)) {
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
  static const uint16_t quant_map[256] = {
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

  #define N_CONFIGS 8
  struct config {
    uint32_t state[4];
    const char *s_gold_hash;
    int n_channels, kx, ky, n_kernels;
  } configs[N_CONFIGS] = {
      {{1, 2, 3, 4}, "E502761DFD02E62B7669EBAF40EF745E5D5DE573BFBEC155536E3BABD683B808", 256, 1, 1, 512},
      {{1, 2, 3, 4}, "582027B12DC615C77891BFC12E05990646C733A59D964A611682F81A4D445649", 128, 3, 3, 256},
      {{1, 2, 3, 4}, "187F56F18425AAD6D680D3AAAB3D6666B1EC37A9A55D58FFBBA7FC5F07233BDD", 64, 5, 5, 128},
      {{1, 2, 3, 4}, "85A0CE8578581124D194C32998F4C170034BB5C49E53D7C065788C8425C4266C", 64, 7, 7, 128},
      {{1, 2, 3, 4}, "C4D25989D994CFE4FE3828C140B33277A1726A25D0C46A49DD9DBC857F5EDDE9", 260, 1, 1, 510},
      {{1, 2, 3, 4}, "57227F3D6142C1277600DF131F05F82523F840E411E99092CC9F7D75CCB75E38", 70, 3, 3, 130},
      {{1, 2, 3, 4}, "3991503316E0C268B5AB7C6B69E537E2D00963A8EF3F3D7582D50E827E7C5053", 70, 5, 5, 130},
      {{1, 2, 3, 4}, "297B1A067D2BC8FF506E89D6C8276A0BA40D060844FECEDD12230595AD7E9C29", 70, 7, 7, 130},
  };

  for (int i = 0; i < N_CONFIGS; ++i) {
    res = test_weights(configs[i].state, configs[i].s_gold_hash,
                       quant_map, configs[i].n_channels, configs[i].kx, configs[i].ky, configs[i].n_kernels);
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
