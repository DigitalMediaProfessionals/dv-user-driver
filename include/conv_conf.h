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

#ifndef CONV_CONF_H_
#define CONV_CONF_H_

#define CONV_RUNS_MAX 32
#include "hw_conv_conf.h"
////////////////////////////////////////////////////////////////////////////
//SW  configuration structures (Optional debug info included)
////////////////////////////////////////////////////////////////////////////

/*! @brief Header for software convolution info. Currently empty
*/
struct d_conv_header { // 4 bytes
};

/*! @brief Input for software convolution info. Currently empty
*/
struct d_conv_input {
};

/*! @brief Output for software convolution info. Output information should match next layer input.
*/
struct d_conv_output {
	unsigned short w; //!<Output Width
	unsigned short h; //!<Output Height
	unsigned short z; //!<Output Depth
	unsigned short m; //!<Output Channels
	int			 performance;	//!<rendering time in ms
};

/*! @brief Extra SW information for each run. Contains among several the names of the layers that were present in the original network before conversion.
*/
struct d_conv_run {
	unsigned short in_w;  //!< Input width (not used by HW - discovered on the fly)
	unsigned short in_h;  //!< Input height (not used by HW - discovered on the fly)
	unsigned short in_c;  //!< Input Channels (not used by HW - discovered on the fly)
	unsigned short out_w; //!< Output width (not used by HW - discovered on the fly)
	unsigned short out_h; //!< Output height (not used by HW - discovered on the fly)

	char			 conv_name[128]; //!< Convolution name, as defined in the original network before conversion

	unsigned int   weight_size;		//!< Actual size in bytes of LUT, weights and bias (in bytes);
								// POOL
	char			 pool_name[128];//!< Pooling name, as defined in the original network before conversion

	char			 act_name[128]; //!< Activation function name, as defined in the original network before conversion

	char  scale_name[128];//!< Scale name, as defined in the original network before conversion
	char  batchnorm_name[128];//!< Batch normalization name, as defined in the original network before conversion
	char  lrn_name[128];//!< Normalization name, as defined in the original network before conversion
};

/*! @brief Single structure regrouping the header, input, output and each run information.
*/
struct sw_conf
{
	struct d_conv_header header; //!< Header for software convolution info. Currently empty
	struct d_conv_input  input;  //!< Input for software convolution info. Currently empty
	struct d_conv_output output; //!< Output SW information for the convolution. 
	struct d_conv_run    run[CONV_RUNS_MAX];//!< Each run SW information for the convolution.
};

/*! @brief Full layer information containing the SW and HW configuration of the convolution layer. In practice, only the HW related structure is used to configure the FPGA - HW convolution module.
*/
struct top_conv_conf
{
	struct hw_conf hw;//!< Hardware structure. Must not be modified as it is copied directly to FPGA module (i.e. same number of fields and field types)
	struct sw_conf sw;//!< Extra information to help debugging or manipulating the network
};


/*! @brief Utility function: Return the number of runs in a convolution layer
*/
inline int conv_conf_num_runs( struct top_conv_conf *conf) {
  unsigned int i = conf->hw.header.topo; int n = 0;
  for (; i; i >>= 1, n++); // Faster for low n...
  return n;
}


/*! @brief Utility function: Size of the conv_conf struct in bytes (unused run structs not counted)
	@details This function is used to copy the structure to RISC-V. Only the  hw_conf is actually sent. The sw_conf structure is only there for storing usefull information for debugging/manipulating the network. 
*/
inline int hw_conf_size(struct top_conv_conf *conf) {
  int n = conv_conf_num_runs(conf);
  return sizeof(struct hw_conf) - (32-n)*sizeof(struct conv_run);
}


#endif // CONV_CONF_H_
