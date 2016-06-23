/*
Copyright (c) 2014, Washington University in St. Louis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Washington University in St. Louis nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL WASHINGTON UNIVERSITY BE LIABLE FOR ANY 
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef LASP_DEVICE_CONTEXT_H
#define LASP_DEVICE_CONTEXT_H

#define __STDC_LIMIT_MACROS

#ifdef WUCUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
using std::min;

namespace lasp {
	
	template<class T>
    class LaspMatrix;
	
    class DeviceContext;
	
    //Wrap CUDA call information
    struct DeviceParams{
		DeviceContext& context;
		int error;
		int blockDim;
		int gridDim;
		
		DeviceParams(DeviceContext& context_, int error_ = 0, int blockDim_ = 0, int gridDim_ = 0)
		:context(context_), error(error_), blockDim(blockDim_), gridDim(gridDim_) {}
	};
	
	//Use later for synchronization and load balancing
    /*
	 Thoughts on usage:
	 - Hold cublas context for access
	 - Manage streams and multiple devices
	 - Make calls asynchronous if possible, use to manage synchronization
	 - Manage low memory situations, either by paging values to main memory or
	 by condensing matricies into rows * cols mem-footprint
	 */
	class DeviceContext {
		char* nextKey;
		static DeviceContext* instance_;
		
		int ompLimit;
		
#ifndef WUCUDA
	    DeviceContext() : nextKey(0), ompLimit(2000) {}
		~DeviceContext(){}
		
#else
		cublasHandle_t cublas;
		cudaDeviceProp device_0;
		int numDevices;
	    
	    DeviceContext(): nextKey(0), ompLimit(2000) {
	    	numDevices = 1;
			cublasCreate(&cublas);
			cudaGetDeviceCount(&numDevices);
			cudaGetDeviceProperties(&device_0, 0);
		}
		
		~DeviceContext(){
			cublasDestroy(cublas);
			cudaDeviceReset();
		}
		
		
#endif
		
	public:
		
		int getNumDevices(){
			int devices = 0;
#ifdef WUCUDA
			devices = numDevices;
#endif
			return devices;
		}
		
		int getOmpLimit(){
			return ompLimit;
		}
		
		void setNumDevices(int devices){
#ifdef WUCUDA
			numDevices = min(devices, numDevices);
#endif
		}
		
		size_t getAvailableMemory(int device=0){
			size_t availableMemory = 0;
			size_t totalMemory = 0;
#ifdef WUCUDA
			cudaMemGetInfo(&availableMemory, &totalMemory);
#endif
			return availableMemory;
		}
		
#ifdef WUCUDA
		cublasHandle_t& getCuBlasHandle(){
			return cublas;
		}
#endif
		
		static DeviceContext* instance() {
			if(instance_ == 0){
				instance_ = new DeviceContext();
			}
			return instance_;
		}
		
		void* getNextKey(){
			return static_cast<void*>(nextKey++);
		}
		
		template<class T>
		DeviceParams setupMemTransfer(LaspMatrix<T>* matrix, LaspMatrix<T>* otherMatrix = 0) {
#ifdef WUCUDA
			cudaDeviceSynchronize();
#endif
			return DeviceParams(*this);
		}
		
		template<class T>
		DeviceParams setupOperation(LaspMatrix<T>* mat1, LaspMatrix<T>* mat2 = 0, LaspMatrix<T>* mat3 = 0) {
#ifdef WUCUDA
			cudaDeviceSynchronize(); //TODO: FIXME
#endif
			return DeviceParams(*this);
		}
		
	};
	
	
}

#endif
