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

#ifndef HW_CONV_CONF_H_
#define HW_CONV_CONF_H_

#define HW_CONV_RUNS_MAX 32
////////////////////////////////////////////////////////////////////////////
//HW  configuration structures
////////////////////////////////////////////////////////////////////////////
/*! @brief Topology if the convolution layer. 
@details In the case the convolution consists in a single node, the topo value is equal to 1. 

In the case several runs exists, each run corresponds to 1 bit of the topo field. In case the corresponding bit is equal to 1, the output of the run is the external memory. In case it is equal to zero, the run output is the unified buffer.

For example, a topo equal to 11001 defines a topology containing 3 branches. The first branch is defined by run[0], and output its results to external memory. The second branch is defined by run[1], run[2] then run[3], corresponding respectively to the topo bits 0,0 and 1, run[3] being the last layer and outputing its result to external memory, run[1] and [2] outputing their results to unified buffer. The last branch is defined by run [4] and output its results to external memory directly, same as run[0].
*/
struct conv_header { // 4 bytes
  unsigned int topo; //!< [31:0] Output Destination of each run, 0 = UBUF, 1 = EXTMEM
};

/*! @brief Input of the FPGA convolution layer. 
*/
struct conv_input {
	unsigned short w; //!<Input Width
	unsigned short h; //!<Input Height
	unsigned short z; //!<Input Depth
	unsigned short c; //!<Input Channels
	unsigned int   input_base_addr; //!<Input byte address
	unsigned short input_circular_offset; //!<Input Depth circular offset
	unsigned short tiles; //!<Number of horizontal tiles (supported with restrictions)
};

/*! @brief Output of the FPGA convolution layer. Direct output or Element-wise  output can be defined here.
*/
struct conv_output {
	unsigned int   output_base_addr; //!<Output byte address
	unsigned int   eltwise_base_addr; //!<Input byte address for elementwise add (0 = UBUF Input Buffer)
	unsigned short output_mode; //!<0 = concat, 1 = eltwise add
	unsigned short ALIGN_0;//!< Unused. Default value is 0
};

/*! @brief FPGA convolution layer runs. Inside an FPGA convolution layer, several convolutions can be defined in each run. Typically, an entire GoogLeNet inception module can be mapped to a single FPGA convolution node. 
*/
struct conv_run {
	unsigned short m; //!< Output Channels
					  // CONV
	unsigned short conv_enable; //!< 1 = Enabled, 0 = Disabled, 3 = LRN
	unsigned short p; //!<Filter Size (width = height)
	unsigned short pz; //!<Filter Depth (1 in case of 2D convolution)
	unsigned int   conv_pad; //!<bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
	unsigned short conv_stride; //!<bits [7:0] = X stride, bits [15:8] = Y stride
	unsigned short conv_dilation; //!<bits [7:0] = X dilation, bits [15:8] = Y dilation
	unsigned int   weight_base_addr; //!<Filter Weight and Bias byte address
	unsigned short weight_fmt; //!<Weight format (0 = random access blocks, 1 = compact stream, 2 = 8-bit qunatized stream)
	unsigned short ALIGN_0;//!< Unused. Default value is 0
	// POOL
	unsigned short pool_enable; //!< 0 = disabled, 1 = max pooling, 2 = average pooling
	unsigned short pool_avg_param; //!< Must be set to 1/pool_size^2 in FP16 format when using average pooling (average pooling assumes square size)
	unsigned short pool_size; //!< bits [7:0] = width, bits [15:8] = height
	unsigned short pool_stride; //!<bits [7:0] = X stride, bits [15:8] = Y stride
	unsigned int   pool_pad; //!< bits [7:0] = left padding, bits [15:8] = right padding, bits [23:16] = top padding, bits [31:24] = bottom padding
							 // MISC
	unsigned short actfunc; //!< Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU
	unsigned short actfunc_param;//!< Leaky ReLU parameter (NOTE: 0x2E66 is 0.1 in FP16)
	unsigned short rectifi_en; //!< Rectification, i.e. max(0, x) (NOTE: Can be applied after non-ReLU activation function)
	unsigned short lrn; //!< [0] : 1 = LRN enable, 0 = LRN disable, [1] : 1 = incl. power func, 0 = excl., [8:11] = x^2 scale factor log2 
};

/*! @brief HW configuration of an FPGA convolution module. This structure is directly copied to the RISC-V controlling the hard. <B>Do not change field types or add new fields to any of those structures</B>. Because it is a direct memory copy, differences will lead to error on the HW module side and prevent it from working fine. 
*/
struct hw_conf
{
	struct conv_header header;//!< Header for HW configuration. Defines the topology of the subgraph contained in the FPGA convolution layer
	struct conv_input  input;//!< Input settings for the FPGA convolution layer
	struct conv_output output;//!<Ouput settings for the FPGA convolution layer
	struct conv_run    run[HW_CONV_RUNS_MAX];//!< Settings for the each convolution in the FPGA convolution layer
};

#endif // HW_CONV_CONF_H_
