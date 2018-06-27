
#ifndef DMPSYS_H_
#define DMPSYS_H_

#define MEM_DEV  "/dev/mem"
#define CNV_DEV  "/dev/DMP_drm0"
#define PDC_DEV  "/dev/DMP_drm1"
#define FC_DEV   "/dev/DMP_drm2"

#ifdef DMP_ZC706
#define	CNV_REG_BASE    0x43c00000
#define FC_REG_BASE     0x43c20000
#ifdef STATIC_MEM
#define SYS_DDR_BASE_PA 0x10000000
#else
#define SYS_DDR_BASE_PA 0x01000000
#endif
#endif

#ifdef DMP_ARRIA10
#define	CNV_REG_BASE    0xff210000
#define FC_REG_BASE     0xff200000
#ifdef STATIC_MEM
#define SYS_DDR_BASE_PA 0x10000000
#else
#define SYS_DDR_BASE_PA 0x01000000
#endif
#endif

#define	CNV_REG_SIZE 0x2000
#define FC_REG_SIZE  0x100
//#define SYS_DDR_SIZE 0x30000000
#define SYS_DDR_SIZE 0x30000000

#define CNV_IOC_MAJOR 0x82
#define CNV_WAITPDC  _IOR(CNV_IOC_MAJOR, 3, unsigned int)
#define CNV_MEMSEC   _IOR(CNV_IOC_MAJOR, 4, unsigned int)
#define CNV_WAITINT  _IO(CNV_IOC_MAJOR, 6)

#endif  // DMPSYS_H_
