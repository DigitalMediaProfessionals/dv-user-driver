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
