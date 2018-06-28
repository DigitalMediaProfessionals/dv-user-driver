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

#pragma once


namespace dmp {
namespace modules {
	
/*!
		@brief Utility function. Get the i/o memory address of the FPGA convolution module. Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
unsigned long get_iomap_cnv();

/*!
		@brief Utility function. Get the i/o memory address of the FPGA fully connected module. Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
unsigned long get_iomap_fc ();

/*!
		@brief Utility function. Get the i/o memory address of the FPGA display controller module. Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
unsigned long get_iomap_pdc();

/*!
		@brief Utility function. Get the i/o memory address of the FPGA display controller module. Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
unsigned long get_iomap_ddr();

/*!
		@brief Utility function. Get the address of the FPGA convolution module. Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
int          get_fdC();

/*!
		@brief Utility function. Get the address of the FPGA fully connected module. Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
int          get_fdF();
/*!
		@brief Utility function. Get the address of the FPGA display controler module. Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
int          get_fdP();
/*!
		@brief Utility function. Get the memory address which will be displayed by the FPGA display controller . Used by the driver, most likely not to be used in the application, except for debugging or verification.
*/
unsigned int get_fbA();


/*!
@brief Interrupt handles.  Read busy state of a module and wait until interrupt is raised. 
@param[in] module: Either convolution or fully connected module
*/
void cnvWaitInt(unsigned int module);

/*!
@brief Swap display buffer. 
*/
void swap_buffer();

/*!
@brief Initialize hardware modules, including convolution, fully connected, display controller and memory controller. 
*/
bool initialize();

/*!
@brief Close hardware modules, including convolution, fully connected, display controller and memory controller. 
*/
void shutdown();

/*!
@brief Display in the output console information about the hardware modules, including each module frequency. 
*/

void get_hw_info();

enum {
  FREQ_CONV = 0x01,
  FREQ_FC = 0x02,
  FREQ_PDC = 0x03,
  FREQ_PIX = 0x04,
};

void get_info(unsigned int v, unsigned int *ret);

/*!
@brief In the case the fpga has buttons, reset the state the state of all buttons. <B>This function may be modified depending on the FPGA board.</B> 
*/
void reset_button_state();

/*!
@brief In the case the fpga has buttons, read and reset the state the state of all buttons. <B>This function may be modified depending on the FPGA board.</B> 
*/
unsigned int get_button_state();

};//end of namespace modules 
};//end of dmp
