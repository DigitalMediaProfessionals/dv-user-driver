#pragma once

#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#include <stdint.h>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <iostream>
#include <ostream>
#include <string>
#include <cstring>
#include <algorithm>
#include <utility>
#include <cmath>

#include "stats.h"
#include "dmpSys.h"
#include "conv_conf.h"
#include "fc_conf.h"
#include "hw_module_manager.h"

enum layer_type {
	LT_INPUT,
	LT_CONV,
	LT_FC,
	LT_FLATTEN,
	LT_CONCAT,
	LT_COPY_CONCAT,
	LT_SOFTMAX,
	LT_CUSTOM,
};

struct fpga_layer;

typedef void (*run_custom_callback_proc)(fpga_layer &layer, void *custom_param);

struct fpga_layer {
	layer_type    type;
	void *        hw_conf;
	void *        addr_cpu_input;
	void *        addr_cpu_output;
	unsigned long addr_offset_input;
	unsigned long addr_offset_output;
	unsigned long output_size;
	int           input_dim[3];
	int           input_dim_size;
	int           output_dim[3];
	int           output_dim_size;
	bool          is_output;
	bool          is_f32_output;
	bool          is_input_hw_layout;
	union {
		struct {
			int   softmax_axis;
		};
		struct {
			int input_layer_num;
			fpga_layer **input_layers;
		};
		struct {
			run_custom_callback_proc custom_proc_ptr;
			void *custom_param;
		};
	};
};

class CDMP_Network {

private:

	/*!
		@brief Boolean value to enable or disable console debug output.
		@details If enabled, various information, such as memory size request, allocation etc. are displayed.
		*/
	bool _verbose;

protected:
	unsigned int num_layers;
	unsigned int num_output_layers;
	unsigned int num_conv_layers;
	unsigned int num_fc_layers;
	unsigned int weight_size;
	unsigned int buffer_size;
	void *weight_buffer_addr;
	std::vector<fpga_layer> layers;
	std::vector<fpga_layer*> output_layers;
	std::vector<top_conv_conf> conv_layers;
	std::vector<top_fc_conf> fc_layers;

	/*!
		@brief Vector containing list of memory size requested by the network (i.e. to store the weights, the temporary  buffers during inference etc.)
		*/
	std::vector<unsigned long> memory_size_request;

    std::vector<unsigned long> reserved_memory_addresses_fpga;
    std::vector<unsigned long> reserved_memory_addresses_cpu;

public:
    inline std::vector<fpga_layer*>& get_output_layers() {
      return output_layers;
    }

	fpga_layer& get_layer(unsigned int i);
	top_conv_conf& get_conv_layer(unsigned int i);
	top_fc_conf& get_ip_layer(unsigned int i);

	virtual unsigned int get_total_layer_count() = 0;
	virtual unsigned int get_output_layer_count() = 0;
	virtual unsigned int get_convolution_layer_count() = 0;
	virtual unsigned int get_innerproduct_layer_count() = 0;

	virtual int initialize() = 0;
	bool reserve_memory(bool set_addr = true);
	bool load_weights(const std::string& filename);

	/// @brief Runs the network.
	/// @prarm t_sleep Time for additional sleep in microseconds.
	void run_network(int t_sleep=0);

	void *get_network_input_addr_cpu();
	void get_final_output(std::vector<float>& out, unsigned int i = 0);

	int get_convolution_performance(int layerID = -1); //-1 means sum of all layers, other specific convolution id
	int get_innerproduct_performance(int layerID = -1);

	void verbose(bool);

	inline uint8_t *get_params_base_cpu() {
      return (uint8_t*)reserved_memory_addresses_cpu[0];
    }

    inline uint8_t *get_output_base_cpu() {
      return (uint8_t*)reserved_memory_addresses_cpu[1];
    }

    inline size_t get_params_base_fpga() {
      return reserved_memory_addresses_fpga[0];
    }

    inline size_t get_output_base_fpga() {
      return reserved_memory_addresses_fpga[1];
    }

	CDMP_Network() : _verbose(false) {};
	virtual ~CDMP_Network(){};

	void SetLayerAddresses();
};

void get_layer_input(fpga_layer &layer, std::vector<float> &layer_input);
void put_layer_output(fpga_layer &layer, std::vector<float> &layer_output, bool is_output_hw_layout = false);
