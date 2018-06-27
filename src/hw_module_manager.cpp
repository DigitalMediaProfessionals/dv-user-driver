#include "hw_module_manager.h"

#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <unistd.h>   // note
#include <string.h>   // note
#include <stdlib.h>   // note
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "dmpSys.h"



namespace dmp {
	namespace modules
	{

static unsigned long iomap_cnv;
static unsigned long iomap_fc;
static unsigned long iomap_pdc;
static unsigned long iomap_ddr;
static int fdC;
static int fdF;
static int fdP;
static unsigned int fbA;


static int noWait=0;


unsigned long get_iomap_cnv(){return iomap_cnv;}	;
unsigned long get_iomap_fc (){return iomap_fc;};
unsigned long get_iomap_pdc(){return iomap_pdc;}	;
unsigned long get_iomap_ddr(){return iomap_ddr;}	;


int          get_fdC(){return fdC;};
int          get_fdF(){return fdF;};
int          get_fdP(){return fdP;};
unsigned int get_fbA(){return fbA;};





bool load_program();


int set_iomap(void)
{
  // open /dev/mem
  int fd = open(MEM_DEV, O_RDWR | O_SYNC);
  if (fd <= 0) return -1;

  // mmap registers (CNV):
  void* mm = mmap(0, CNV_REG_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd,
		  CNV_REG_BASE);
  iomap_cnv = (unsigned long)mm;
  if (iomap_cnv < 0) return -1;
  //close(fd);

  // mmap registers (FC):
  mm = mmap(0, FC_REG_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd,
            FC_REG_BASE);
  iomap_fc = (unsigned long)mm;
  if (iomap_fc < 0) return -1;
  //close(fd);

  // open /dev/mem
  //fd = open(MEM_DEV, O_RDWR | O_SYNC);
  //if (fd <= 0) return -1;

  // mmap ddr:
  mm = mmap(0, SYS_DDR_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd,
	    SYS_DDR_BASE_PA);
  iomap_ddr = (unsigned long)mm;
  if (iomap_ddr < 0) return -1;
  close(fd);

  return 0;
}

void cnvWaitInt(unsigned int fd)
{
  if (!(noWait&1)) {
    ioctl(fd,CNV_WAITINT,NULL);
  }
}

void swap_buffer()
{
  int fd = get_fdP();
  unsigned int ret=0;
  ioctl(fd,CNV_WAITPDC,&ret);
  fbA = ret- SYS_DDR_BASE_PA;
 // return ret;
}


bool initialize()
{

  // get i/o file descriptors:
  fdC = open(CNV_DEV,O_RDWR);
  if (fdC<=0) {
    std::cerr<<"CNV dev open failed"<<std::endl;
    return false;
  }


	  fdF = open(FC_DEV,O_RDWR);
  if (fdF<=0) {
    std::cerr<<"FC dev open failed"<<std::endl;
    close(fdC);
    return false;
  }
  fdP = open(PDC_DEV,O_RDWR);

  if (fdP<=0) {
    std::cerr<<"PDC dev open failed"<<std::endl;
    close(fdC);
    close(fdF);
    return false;
  }
  // initialize memory map:
  if (set_iomap()<0) {
    std::cerr<<"mmap failed"<<std::endl;
    close(fdC);
    close(fdF);
    close(fdP);
    return false;
  }


	swap_buffer();
  if(!load_program())
  {
	  return false;

  }

  return true;
}

void shutdown()
{
	close(fdC);
    close(fdF);
    close(fdP);
}

// read a 32-bit little-endian unsigned integer
//used in load_program
unsigned int read32_le(std::istream& stream)
{
    char b[4];
    stream.read((char*)b,4);

    return static_cast<unsigned int>(
        (b[0])      |
        (b[1] << 8) |
        (b[2] << 16)|
        (b[3] << 24) );
}
/*
// read a 16-bit little-endian unsigned integer
unsigned short read16_le(std::istream& stream)
{
    char b[2];
    stream.read((char*)b,2);

    return static_cast<unsigned short>(
        (b[0])      |
        (b[1] << 8) );
}
*/
bool load_program()
{
  unsigned int word;
  std::ifstream dfs("program.bin");
  if (!dfs.is_open()) {
    std::cerr<<"Failed to open RISC-V program file."<<std::endl;
    return false;
  } else {
    int n = 0;
    *(volatile unsigned int*)(iomap_cnv + 0x0080) = 0; // Program starts at address 0
    do {
      word = read32_le(dfs);
      *(volatile unsigned int*)(iomap_cnv + 0x0084) = word; // Write word and increment address
      //cout << hex << word << "\t";
      //if (n%32 == 31)
      //cout << "\n";
      n++;
    } while (word != 0);
    //cout << "\n";
    //cout << "Program size (bytes) = " << dec << 4*n << "\n";
    //cout << "End address = " << hex << (*(volatile unsigned int*)(iomap_cnv + 0x0080))*4 << "\n";
    dfs.close();
    return true;
  }
}


void get_hw_info()
{
	std::string conv_freq, fc_freq;
	unsigned int fi;
	fi 	=*(volatile unsigned long *)(iomap_cnv + ((0x109) << 2)); // read ffrequency information
	conv_freq = std::to_string(fi & 0xFF);
	fc_freq = std::to_string((fi >> 8) & 0xFF);
	int pdc_freq = (fi >> 16) & 0xFF;
	int pix_freq = (fi >> 24) & 0xFF;

	std::cout<<std::dec << "conv_freq = " << conv_freq << std::endl;
	std::cout<<std::dec << "fc_freq   = " << fc_freq << std::endl;
	std::cout<<std::dec << "pdc_freq  = " << pdc_freq << std::endl;
	std::cout<<std::dec << "pix_freq  = " << pix_freq << std::endl;

}

void get_info(unsigned int v, unsigned int *ret)
{
  unsigned int fi;
  fi = *(volatile unsigned long *)(iomap_cnv + ((0x109) << 2)); // read frequency information
  switch (v) {
  case FREQ_CONV: *ret = fi & 0xFF; break;
  case FREQ_FC: *ret = (fi >> 8) & 0xFF; break;
  case FREQ_PDC: *ret = (fi >> 16) & 0xFF; break;
  case FREQ_PIX: *ret = (fi >> 24) & 0xFF; break;
  default: *ret = -1;
  }
}

enum {
  sanityRW_addr=0x106*4,
  buttonChk_addr=0x107*4
};


void 	reset_button_state()
{

    *(volatile unsigned int*)(iomap_cnv + buttonChk_addr) = 0; // clear button sticky bits
}

unsigned int get_button_state()
{
 unsigned int button = *(volatile unsigned int*)(iomap_cnv + buttonChk_addr); // read and clear button sticky bits

 return button;
 }
	};//end of namespace modules
};//end of dmp
