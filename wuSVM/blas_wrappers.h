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

#ifndef LASP_WRAPPERS_H
#define LASP_WRAPPERS_H

#include "options.h"
#include "device_context.h"
#include <algorithm>

#ifdef WUCUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace lasp{

	enum {
		BLAS_SUCCESS = 0,
		CONTEXT_ERROR
	};
	
	int host_dgemm(bool transa, bool transb, int m, int n, int k, double alpha, double* a, int lda, double * b, int ldb, double beta, double* c, int ldc);
	int device_dgemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, double alpha, double* a, int lda, double * b, int ldb, double beta, double* c, int ldc);

	int host_sgemm(bool transa, bool transb, int m, int n, int k, float alpha, float* a, int lda, float * b, int ldb, float beta, float* c, int ldc);
	int device_sgemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, float alpha, float* a, int lda, float * b, int ldb, float beta, float* c, int ldc);

	int host_dger(int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda);
	int device_dger(DeviceParams params, int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda);

	int host_sger(int m, int n, float alpha, float* x, int incx, float* y, int incy, float* a, int lda);
	int device_sger(DeviceParams params, int m, int n, float alpha, float* x, int incx, float* y, int incy, float* a, int lda);

	int host_dgesv(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb);
	int device_dgesv(DeviceParams params, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb);

	int host_sgesv(int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb);
	int device_sgesv(DeviceParams params, int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb);

	int device_colSqSum(DeviceParams params, float* A, int n, int features, float* result, float scalar, int mRows, int out_mRows);
	int device_colSqSum(DeviceParams params, double* A, int n, int features, double* result, double scalar, int mRows, int out_mRows);
	
	int device_colSum(DeviceParams params, float* A, int n, int features, float* result, float scalar, int mRows, int out_mRows);
	int device_colSum(DeviceParams params, double* A, int n, int features, double* result, double scalar, int mRows, int out_mRows);

 	int device_ewiseOp(DeviceParams params, float* in, float* out, int length, float mult, float add, float pow1, int rows, int mRows, int out_mRows);
  	int device_ewiseOp(DeviceParams params, double* in, double* out, int length, double mult, double add, double pow1, int rows, int mRows, int out_mRows);

	int device_colWiseMult(DeviceParams params, float* mat, float* out, float* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows);
	int device_colWiseMult(DeviceParams params, double* mat, double* out, double* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows);

	int device_rowWiseMult(DeviceParams params, float* mat, float* out, float* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows);
	int device_rowWiseMult(DeviceParams params, double* mat, double* out, double* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows);

	int device_eWiseDiv(DeviceParams params, float* in1, float* in2, float* out, int length, float pow1, float pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows);
	int device_eWiseDiv(DeviceParams params, double* in1, double* in2, double* out, int length, double pow1, double pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows);
	
	int device_eWiseMult(DeviceParams params, float* in1, float* in2, float* out, int length, float pow1, float pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows);
	int device_eWiseMult(DeviceParams params, double* in1, double* in2, double* out, int length, double pow1, double pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows);

  	int device_gather(DeviceParams params, int* map, float* src, float* dst, int rows, int mRows, int out_mRows, int mapSize);
  	int device_gather(DeviceParams params, int* map, double* src, double* dst, int rows, int mRows, int out_mRows, int mapSize);

  	int device_exp(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows, float gamma);
  	int device_exp(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows, double gamma);

  	int device_tanh(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows);
  	int device_tanh(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows);

	int device_chooseNextHelper(DeviceParams params, float* d_x, int d_xInd, float* g, float* h, int select, float* d_out_minus1, float* dK2, int dK2rows, int dK2cols);
  	int device_chooseNextHelper(DeviceParams params, double* d_x, int d_xInd, double* g, double* h, int select, double* d_out_minus1, double* dK2, int dK2rows, int dK2cols);

	int device_addMatrix(DeviceParams params, double* a, double* b, double* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows);
	int device_subMatrix(DeviceParams params, double* a, double* b, double* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows);

	int device_addMatrix(DeviceParams params, float* a, float* b, float* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows);
	int device_subMatrix(DeviceParams params, float* a, float* b, float* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows);

	int device_transpose(DeviceParams params, float* in, float* out, int cols, int rows, int mRows, int out_mRows);
  	int device_transpose(DeviceParams params, double* in, double* out, int cols, int rows, int mRows, int out_mRows);
 
 	int device_gatherSum(DeviceParams params, int* map, float* src, float* dst, int rows, int mRows, int out_mRows, int mapSize, int outputRows);
  	int device_gatherSum(DeviceParams params, int* map, double* src, double* dst, int rows, int mRows, int out_mRows, int mapSize, int outputRows);

  	template<class T>
	int pinned_kernel_multiply(DeviceParams params, T* A, int lda, int aCols, T* aNorm, int aRows, T* B, int ldb, int bCols, T* bNorm, int bRows, T* Out, int ldOut, kernel_opt kernelOptions, bool doKernel, int a_cpuBlocks, int b_cpuBlocks, int a_gpuBlocks, int b_gpuBlocks, int num_streams_input, int num_device_input, bool a_on_device, bool b_on_device, bool out_on_device, bool transpose);

#ifdef WUCUDA
	
	#ifndef NDEBUG
		#define CUDA_CHECK(call) \
		if((call) != cudaSuccess) { \
			cudaError_t err = cudaGetLastError(); \
			cerr << "CUDA error calling \""#call"\", code is " << cudaGetErrorString(err) << endl; \
			return UNSPECIFIED_MATRIX_ERROR; }

	    #define CUDA_CHECK_THROW(call) \
			if((call) != cudaSuccess) { \
				cudaError_t err = cudaGetLastError(); \
				cerr << "CUDA error calling \""#call"\", code is " << cudaGetErrorString(err) << endl; \
				throw UNSPECIFIED_MATRIX_ERROR; }

	    #define CUBLAS_CHECK(call) \
				if((call) != CUBLAS_STATUS_SUCCESS) { \
					cudaError_t err = cudaGetLastError(); \
					cerr << "CUDA error calling \""#call"\", code is " << cudaGetErrorString(err) << endl; \
					return call; }
	#else			
		#define CUDA_CHECK(call) \
		if((call) != cudaSuccess) { \
			return UNSPECIFIED_MATRIX_ERROR; }

	    #define CUDA_CHECK_THROW(call) \
			if((call) != cudaSuccess) { \
				throw UNSPECIFIED_MATRIX_ERROR; }

	    #define CUBLAS_CHECK(call) \
				if((call) != CUBLAS_STATUS_SUCCESS) { \
					return call; }
	#endif


 	int device_ger(DeviceParams params, int m, int n, float alpha, float* x, int incx, float* y, int incy, float* a, int lda);
	int device_ger(DeviceParams params, int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda);

	int device_gemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, float alpha, float* a, int lda, float * b, int ldb, float beta, float* c, int ldc);
  	int device_gemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, double alpha, double* a, int lda, double * b, int ldb, double beta, double* c, int ldc);

  	int device_geam(DeviceParams params, bool transa, bool transb, int m, int n, double alpha, double* a, int lda, double beta,  double * b, int ldb,double* c, int ldc);
	int device_geam(DeviceParams params, bool transa, bool transb, int m, int n, float alpha, float* a, int lda, float beta, float * b, int ldb, float* c, int ldc);

  	int device_tanh_stream(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows, cudaStream_t &stream);
  	int device_tanh_stream(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows, cudaStream_t &stream);
 
  	int device_exp_stream(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows, float gamma, cudaStream_t &stream);
  	int device_exp_stream(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows, double gamma, cudaStream_t &stream);

    int device_ewiseOp_stream(DeviceParams params, float* in, float* out, int length, float add, float mult,  float pow1, int rows, int mRows, int out_mRows, cudaStream_t &stream);
    int device_ewiseOp_stream(DeviceParams params, double* in, double* out, int length, double add, double mult,  double pow1, int rows, int mRows, int out_mRows, cudaStream_t &stream);
#endif

#ifndef WUCUDA
  	template<class N, class T>
  	int device_convert(DeviceParams params, T* in, N* out, int rows, int cols, int mRows){
  		cerr << "Device operations not supported" << endl;
  		return 0;
  	}
#else
  	template<class N, class T>
  	int device_convert(DeviceParams params, T* in, N* out, int rows, int cols, int mRows);
#endif

  }
#endif
