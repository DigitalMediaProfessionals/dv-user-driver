#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <memory>
#include <errno.h>

#include "dmp_dv.h"
#include "dmp_dv_cmdraw_v0.h"

using namespace std;


#define PERR(fmt, ...) fprintf(stderr, "%s:%u # " fmt "\n", \
    __FILE__, __LINE__, ##__VA_ARGS__)
#define PERRNO(fmt, ...) fprintf(stderr, "%s:%u # errno(%s) : " fmt " \n", \
    __FILE__, __LINE__, strerror(errno), ##__VA_ARGS__)

namespace {
  dmp_dv_context context    = nullptr;
  dmp_dv_mem     phys_mem   = nullptr;
  uint8_t        *phys_map  = nullptr;
  size_t         cma_size  = 0;
  const char test_config_file[] = "test_maximizer_config.csv";

  enum class RESULT {
    FAIL = 0,
    SUCCESS = 1,
    ERROR_ON_ADDRAW = 2,
  };

  class Test {
    public:
      uint16_t width;   // 0, 1, 128, 491(prime value), 65534, 65535
      uint16_t height;  // 0, 1, 128, 491(prime value), 65534, 65535
      uint16_t nclass;  // 0, 1, 8, 128, 149(prime value), 254, 255
      string valgen;  // rand, 0, -1, 1
      RESULT expected;  // 1(success), 2(ERROR_ON_ADDRAW)

      friend ostream& operator<<(ostream &os, const Test &c);

      RESULT run_test(dmp_dv_context ctx, dmp_dv_mem mem, uint8_t *map) {
        dmp_dv_cmdlist cmdlist = nullptr;
        struct dmp_dv_cmdraw_maximizer_v0 cmd;
        uint8_t *sw_output = new uint8_t[width * height];
        RESULT ret = RESULT::FAIL;
        int64_t exec_id;

        if(!sw_output) {
          goto error;
        }

        // setup IPU
        cmdlist = dmp_dv_cmdlist_create(ctx);
        if(!cmdlist) {
          goto error;
        }
        fill_cmdraw_v0(cmd, mem);
        if(dmp_dv_cmdlist_add_raw(cmdlist, reinterpret_cast<struct dmp_dv_cmdraw*>(&cmd))) {
          ret = RESULT::ERROR_ON_ADDRAW;
          goto error;
        }
        if(dmp_dv_cmdlist_commit(cmdlist)) {
          goto error;
        }

        // initialize input
        initialize_input_buf(map);
        if(dmp_dv_mem_sync_start(mem, 0, 1)) {
          goto error;
        }
        if(dmp_dv_mem_sync_end(mem)) {
          goto error;
        }

        // run HW/SW IPU
        exec_id = dmp_dv_cmdlist_exec(cmdlist);
        if(exec_id < 0) {
          goto error;
        }
        run_sw_maximizer(reinterpret_cast<__fp16*>(map), sw_output);

        if(dmp_dv_cmdlist_wait(cmdlist, exec_id)) {
          goto error;
        }
        if(dmp_dv_mem_sync_start(mem, 1, 0)) {
          goto error;
        }

        // compare result
        ret = (memcmp(static_cast<void*>(sw_output), static_cast<void*>(map), height * width) == 0) ?
              RESULT::SUCCESS : RESULT::FAIL;

error:
        if(!cmdlist) {
          dmp_dv_cmdlist_release(cmdlist);
        }
        if(!sw_output) {
          delete[] sw_output;
        }
        return ret;
      }

    private:
      void run_sw_maximizer(__fp16 *input, uint8_t *output) {
        constexpr int sub = 8;
        int block_size_16 = width * height * (nclass > sub ? sub : nclass);
        int block_size_128 = ceil(block_size_16 / 8.0);
        for(int x = 0; x < width; x++) {
          for(int y = 0; y < height ; y++) {
            int blk_i = -1;
            int max_cls = -1;
            int blockS = 0;
            __fp16 max = 0;
            __fp16 comp = 0;
            for(int cls = 0; cls < nclass; cls++) {
              if(cls % sub == 0) {
                blk_i++;
                blockS = nclass - blk_i * sub;
                if(blockS > sub) {
                  blockS = sub;
                }
              }

              comp = input[(block_size_128 << 3) * blk_i + x * height * blockS + y * blockS + (cls % sub)];
              if(cls == 0) {
                max_cls = 0;
                max = comp;
              } else {
                if(comp > max) {
                  max = comp;
                  max_cls = cls;
                }
              }
            }
            output[x * height + y] = max_cls;
          }
        }
      }

      void initialize_input_buf(uint8_t *map) {
        size_t buf_len = width * height * nclass;
        __fp16 *buf = reinterpret_cast<__fp16*>(map);
        size_t i;
        int val = 0;

        if(valgen == "0") {
          memset(buf, 0, buf_len * 2);
        } else {
          if(valgen == "rand") {
            val = 0;
          }  else if(valgen == "-1") {
            val = -1;
          } else if(valgen == "1") {
            val = 1;
          } 
          for(i = 0; i < buf_len; i++) {
            buf[i] = static_cast<__fp16>(val != 0 ? val : rand());
          }
        }
      }

      void fill_cmdraw_v0(struct dmp_dv_cmdraw_maximizer_v0 &cmd, dmp_dv_mem mem) {
        cmd.header.version = 0;
        cmd.header.size = sizeof(cmd);
        cmd.header.device_type = DMP_DV_DEV_MAXIMIZER;
        cmd.input_buf.mem = mem;
        cmd.input_buf.offs = 0;
        cmd.output_buf.mem = mem;
        cmd.output_buf.offs = 2 * width * height * nclass;
        cmd.width = width;
        cmd.height = height;
        cmd.nclass = nclass;
      }
  };

  ostream& operator<<(ostream &os, const Test &c) {
    os << "Test\n" <<
      "\t- width : " << c.width << "\n" << 
      "\t- height : " << c.height << "\n" << 
      "\t- nclass : " << c.nclass << "\n" << 
      "\t- valgen : " << c.valgen << "\n" << 
      "\t- expected : " << static_cast<int>(c.expected) << "\n";
    return os;
  }

  bool read_test_config(istream &is, Test &c) {
    char del = ',';
    int result;
    is >> c.width;
    is.ignore(numeric_limits<streamsize>::max(), del);
    is >> c.height;
    is.ignore(numeric_limits<streamsize>::max(), del);
    is >> c.nclass;
    is.ignore(numeric_limits<streamsize>::max(), del);
    is >> c.valgen;
    is.ignore(numeric_limits<streamsize>::max(), del);
    is >> result;
    c.expected = static_cast<RESULT>(result);

    return !is.fail();
  }

  size_t get_free_cma_size(void) {
    FILE *file = fopen("/proc/meminfo", "r");
    if(!file) {
      return 0;
    }

    char buf[1024];
    size_t cma = 0;
    while(fgets(buf, sizeof(buf), file)) {
      if(strstr(buf, "CmaFree:")) {
        // extract size in kB
        char *p = strchr(buf, ':');
        p++;
        while(isblank(*p)) {
          p++;
        }
        char *end = strchr(p, ' ');
        *end = 0x00;

        cma = atoll(p);
        cma *= 1024;
        break;
      }
    }
    return cma;
  }

  int _init() {
    context = dmp_dv_context_create();
    if(!context) {
      PERR("Fail to create dmp_dv_context");
      return -1;
    }
    cma_size = get_free_cma_size();
    phys_mem = dmp_dv_mem_alloc(context, cma_size);
    if(!phys_mem) {
      PERR("Fail to allocate dmp_dv_mem of %lu B", cma_size);
      return -1;
    }
    phys_map = dmp_dv_mem_map(phys_mem);
    if(!phys_map) {
      PERR("Fail to map dmp_dv_mem");
      return -1;
    }

    return 0;
  }

  void _fin() {
    if(phys_mem) {
      if(phys_map) {
        dmp_dv_mem_unmap(phys_mem);
      }
      dmp_dv_mem_release(phys_mem);
    }
    if(context) {
      dmp_dv_context_release(context);
    }
  }

  const char COLOR_WHITE[]  = "\x1b[37m";
  const char COLOR_GREEN[]  = "\x1b[32m";
  const char COLOR_YELLOW[] = "\x1b[33m";
  const char COLOR_RED[]    = "\x1b[31m";
  void log_success(Test &test) {
    cout << COLOR_GREEN << "[TEST SUCCESSED]" << COLOR_WHITE << test << endl;
  }

  void log_fail(Test &test, RESULT result) {
    cout << COLOR_RED << "[TEST FAILED]" << COLOR_WHITE << test << endl;
  }

  void log_overall_result(unsigned success, unsigned failed, unsigned not_tested) {
    cout << COLOR_YELLOW << "Overall Result\n"
      << COLOR_GREEN << "\tSuccessed Test : " << COLOR_WHITE << success << "\n"
      << COLOR_RED << "\tFailed Test : " << COLOR_WHITE << failed << "\n"
      << "\tNot Executed Test : " << not_tested << "\n";

    cout << "\n"
      << "NOTE some tests are not executed due to memory limitation\n";

    cout << endl;
  }
}


int main(int argc, char const **argv) {
  ifstream f(test_config_file);
  Test test;
  unsigned n_succ = 0;
  unsigned n_fail = 0;
  unsigned n_not_tested = 0;
  size_t buf_sz;
  int ret = 0;
  RESULT result;

  if(f.fail()) {
    PERRNO("Failed to open %s", test_config_file);
    return -1;
  }
  // discard header
  f.ignore(numeric_limits<streamsize>::max(), '\n');

  ret = _init();
  if(ret) {
    goto error;
  }

  // main loop
  while(read_test_config(f, test)) {
    buf_sz = test.width * test.height * (test.nclass * 2 + 1);
    if(buf_sz > cma_size) {
      n_not_tested++;
      continue;
    }

    result = test.run_test(context, phys_mem, phys_map);
    if(test.expected == result) {
      n_succ++;
      log_success(test);
    } else {
      n_fail++;
      ret = -1;
      log_fail(test, result);
    }
  }

  log_overall_result(n_succ, n_fail, n_not_tested);

error:
  _fin();

  return ret;
}
