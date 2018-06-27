#ifndef FC_CONF_H_
#define FC_CONF_H_
#include "hw_fc_conf.h"

struct d_fc_conf {
  char			 fc_name[128];
  char			 act_name[128];
  unsigned int 	 total_size;	// Actual size in bytes of LUT, weights and bias (in bytes);
  void *         weight_addr;
  int			 performance;	//rendering time in ms
};


struct top_fc_conf
{
	struct hw_fc_conf hw;
	struct d_fc_conf  sw;
};

#endif // FC_CONF_H_
