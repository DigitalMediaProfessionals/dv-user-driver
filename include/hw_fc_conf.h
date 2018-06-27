#ifndef HW_FC_CONF_H_
#define HW_FC_CONF_H_

struct hw_fc_conf {	
	unsigned short input_size;
	unsigned short output_size;

	unsigned int stride;	  //equal to input size??
	unsigned int bias_size;	 // bias size (in bytes) = 2 times the output size 

	unsigned int param_base_addr;	//base address
	unsigned int weight_addr;		//weight address = param_base_addr + 2*256 (lut size/float16/2bytes)
	unsigned int bias_addr;	 		//bias address =  weight_addr + stride*input size 

	unsigned int input_base_addr;
	unsigned int output_base_addr;

	unsigned short param_fmt; // 0 = unquantized weight matrix, 1 = qunatized

	unsigned short actfunc; // Activation Function: 0 = None, 1 = Tanh, 2 = Leaky ReLU
	unsigned short actfunc_param; // Leaky ReLU parameter (in FP16 format), 0 = non-leaky
	unsigned short ALIGN_0;
};

#endif // FC_CONF_H_
