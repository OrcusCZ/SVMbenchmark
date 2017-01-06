# SVM training benchmark
This application is used to train SVM model using several open-source GPU accelerated implementations.

## Compilation
CMake version 3.1 or newer is used to generate makefiles or Visual Studio project files.
The application depends on several libraries. Only NVIDIA CUDA must be installed, the rest is optional and implementations depending on them are automatically disabled if dependencies are not found by CMake.

Dependencies:
- NVIDIA CUDA (required by all)
- NVIDIA CUDA Samples (optional, required by gpuSVM)
- Intel MKL (optional, required by wuSVM)
- Intel IPP (optional, required by wuSVM)
- Boost (optional, used by gtSVM if found, but can work without Boost)
- [OHD-SVM](https://github.com/OrcusCZ/OHD-SVM) (optional, only if this implementation is enabled in CMake)

### Compilation steps
1. Download the source codes
2. Install NVIDIA CUDA.
   - gpuSVM includes "helper_cuda.h" from CUDA Samples. If this implementation is not needed, CUDA Samples don't have to be installed.
   - If CUDA Samples are not installed in default location or are not found by CMake, environmental variable CUDA_COMMON_INCLUDE_DIRS must be set to point to "/common/inc/" subdirectory of CUDA Samples directory.
   - If CUDA is not installed in default directory, libcuda.so.1 may not be found by CMake. If this happens, set environmental variable CUDA_LIB_PATH to point to the directory with libcuda.so.1 prior to executing CMake.
3. Optionally install Intel MKL and IPP libraries.
   - These libraries are required by wuSVM, if they are not found, application will be compiled without wuSVM.
   - Environmental variables MKLROOT and IPPROOT must be set to point to the root directories of these libraries so CMake can properly find them.
4. If Boost is installed and not found by CMake, environmental variable BOOST_ROOT can be set to point to the Boost root directory.
   - Boost is required by gtSVM, but changes were made to the code to not require Boost. If Boost is not found, our workaround is used.
5. Optionally clone [OHD-SVM](https://github.com/OrcusCZ/OHD-SVM) repository into SVMbenchmark directory so you end up with folder structure `SVMbenchmark/OHD-SVM`.
   - Only needed if OHD-SVM implementation is enabled in CMake.
6. Use CMake to generate makefile / Visual Studio project files. Specific SVM implementations can be enabled / disabled during CMake configuration.
7. Compile using "make" or Visual Studio.

## Program options
	Use: SVMbenchmark.exe <data> <model> [-<attr1> <value1> ...]
	data   File containing data to be used for training.
	model  Where to store model in LibSVM format.
	attrx  Attribute to set.
	valuex Value of attribute x.
	Attributes:
	k  SVM kernel. Corresponding values:
		l   Linear
		p   Polynomial
		r   RBF
		s   Sigmoid
	c  Training parameter C.
	e  Stopping criteria.
	n  Training parameter nu.
	p  Training parameter p.
	d  Kernel parameter degree.
	g  Kernel parameter gamma.
	f  Kernel parameter coef0.
	t  Force data type. Values are:
		d   Dense data
		s   Sparse data
	i  Select implementation to use. Corresponding values:
		 1   LibSVM (default)
		 2   GPU-LibSVM (Athanasopoulos)
		 3   CuSVM (Carpenter)
		 4   GpuSVM (Catanzaro)
		 5   MultiSVM (Herrero-Lopez)
		 6   GTSVM - large clusters (Andrew Cotter)
		 7   GTSVM - small clusters (Andrew Cotter)
		 8   WuSVM<double, lasp, openMP> (Tyree et al.)
		 9   WuSVM<double, lasp, GPU> (Tyree et al.)
		10   WuSVM<double, pegasos, openMP> (Tyree et al.)
		11   WuSVM<double, pegasos, GPU> (Tyree et al.)
		12   WuSVM<float, lasp, openMP> (Tyree et al.)
		13   WuSVM<float, lasp, GPU> (Tyree et al.)
		14   WuSVM<float, pegasos, openMP> (Tyree et al.)
		15   WuSVM<float, pegasos, GPU> (Tyree et al.)
		16   OHD-SVM (Michalek,Vanek)
	b  Read input data in binary format (lasvm dense or sparse format)
	w  Working set size (currently only for implementation 12)
	r  Cache size in MB

If no kernel is specified, RBF is used.
Default value for epsilon (-e) is 1e-3.

Note that implementations 6 and 7 (gtSVM) are only available in separate GNU GPLv3 application due to incompatible licence.

## Running the application and examples
All training options are specified on the command line, there is no configuration file.
Input data can be either in LibSVM text file format or lasvm binary format, if using binary, option -b must be specified on the command line.
Trained model is saved to LibSVM text file format.
After the training is done, the application prints times of different parts of its execution (data loading, training, etc.) to standard output.

Example data are provided in "examples.zip" in [releases page on our GitHub](https://github.com/OrcusCZ/SVMbenchmark/releases).

> SVMbenchmark data.txt model.mdl -g 0.5 -c 1 -i 3

Use data from text file data.txt, save model to model.mdl, use RBF kernel with parameter gamma = 0.5, training parameter C = 4 and use CuSVM.

> SVMbenchmark a9a.txt model.mdl -g 0.5 -c 4 -i 4

Use data from text file a9a.txt, save model to model.mdl, use RBF kernel with parameter gamma = 0.5, training parameter C = 4 and use GpuSVM.

> SVMbenchmark timit_training.bin model.mdl -b -g 0.025 -c 1 -i 6

Use data from binary file timit_training.bin, save model to model.mdl, use RBF kernel with parameter g = 0.025, training parameter C = 1 and use GTSVM.

## Changes to SVM training implementation codes
There were problems with some libraries and changes to the code had to be made in order to make them work or fix their behaviour on specific system configurations.
All changes made by us are as follows:

- GpuSVM

  To make the code working, Cache.cpp had to be modified by filling the cache list in Cache class constructor.
  Minor changes were made to elapsed time measurement to be compatible with other implementations.
  Some prints of memory size had to be fixed for sizes larger than 4 GB.
- cuSVM

  cuSVMSolver.cu was modified for 3 reasons:
  1. cudaMemcpyToSymbol as different syntax in current CUDA API
  2. Implicit inter-warp synchronization doesn't work for current GPUs, it only used to work for old ones.
  3. An amount of free GPU memory is stored in 32 bit integer, but after changing it to 64 bit integer,
     the implementation may still fail with access memory violation when executed on GPU with 4 GB of memory or more.
     Workaround is limiting cache size to 2 GB using -r cache program option.
- MultiSVM

  Cache.cpp had to be modified the same way as in GpuSVM
- gtSVM

  Header files "helpers.h" and "headers.h" include Boost for features which are already in C++11.
  We created header files "boost_helpers.h" and "boost_headers.h" to use C++11 if Boost was not available to avoid this dependency.
