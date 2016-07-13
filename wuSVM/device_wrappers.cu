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


#ifdef WUCUDA
#include "blas_wrappers.h"
#include "convert.h"
#include "abstract_matrix.h"

namespace lasp{

DeviceContext*  DeviceContext::instance_ = 0;

//Device wrappers for blas calls begin on line __ (Search for 'DEVICE_WRAPPERS')

////////////////////////////////////////////////////////////////////////////////
//                                 CUDA_KERNELS                               //
////////////////////////////////////////////////////////////////////////////////

// Kernels for elementwise addition/subraction, multiplication/division, and powers/roots
  __global__ void eWiseOp_kernel_float(float* in, float* out, int length, float mult, float add, float pow1, int rows, int mRows, int out_mRows){
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;
      int index_y = blockDim.y * blockIdx.y + threadIdx.y;
      int grid_width = gridDim.x * blockDim.x;
      int threadID = index_y * grid_width + index_x;
      
      if (threadID < length){
        if(mult == 0 || pow1 == 1){
          out[(threadID / rows) * out_mRows + (threadID % rows)] = in[(threadID / rows) * mRows + (threadID % rows)] * mult + add;
        } else if (pow1 == 2.0){
          float tempVal = in[(threadID / rows) * mRows + (threadID % rows)];
          out[(threadID / rows) * out_mRows + (threadID % rows)] = tempVal * tempVal * mult + add;
        } else {
          out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in[(threadID / rows) * mRows + (threadID % rows)], pow1) * mult + add;
        }
      }    
    }


    __global__ void eWiseOp_kernel_double(double* in, double* out, int length, double mult, double add, double pow1, int rows, int mRows, int out_mRows){
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;
      int index_y = blockDim.y * blockIdx.y + threadIdx.y;
      int grid_width = gridDim.x * blockDim.x;
      int threadID = index_y * grid_width + index_x;
      
      if (threadID < length){
        if(mult == 0 || pow1 == 1){
          out[(threadID / rows) * out_mRows + (threadID % rows)] = in[(threadID / rows) * mRows + (threadID % rows)] * mult + add;
        } else if (pow1 == 2.0){
          double tempVal = in[(threadID / rows) * mRows + (threadID % rows)];
          out[(threadID / rows) * out_mRows + (threadID % rows)] = tempVal * tempVal * mult + add;
        } else {
          out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in[(threadID / rows) * mRows + (threadID % rows)], pow1) * mult + add;
        }
      }    
    }


// Kernels for elementwise squaring
  __global__ void eWiseSquare_kernel_float(float* in, float* out, int length, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    
    if (threadID < length){
      float result = in[(threadID / rows) * mRows + (threadID % rows)];
      out[(threadID / rows) * out_mRows + (threadID % rows)] = result * result;
    }    
  }

  __global__ void eWiseSquare_kernel_double(double* in, double* out, int length, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    
    if (threadID < length){
      double result = in[(threadID / rows) * mRows + (threadID % rows)];
      out[(threadID / rows) * out_mRows + (threadID % rows)] = result * result;
    }    
  }

// Kernels for multiplying each column of a matrix, elmenentwise, by the corresponding entry of a vector
  __global__ void colWiseMult_kernel_float(float* in_mat, float* out_mat, float * vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    // mat is x by y, vec has length x
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    if (threadID < rows * cols){
      out_mat[(threadID / rows) * out_mRows + (threadID % rows)] = in_mat[(threadID / rows) * mRows + (threadID % rows)] * vec[(threadID % rows) * vec_mRows];
    }
  }

  __global__ void colWiseMult_kernel_double(double* in_mat, double* out_mat, double * vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    // mat is x by y, vec has length x
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    if (threadID < rows * cols){
      out_mat[(threadID / rows) * out_mRows + (threadID % rows)] = in_mat[(threadID / rows) * mRows + (threadID % rows)] * vec[(threadID % rows) * vec_mRows];
    }
  }
  __global__ void rowWiseMult_kernel_float(float* in_mat, float* out_mat, float * vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    if (threadID < rows * cols){
      out_mat[(threadID / rows) * out_mRows + (threadID % rows)] = in_mat[(threadID / rows) * mRows + (threadID % rows)] * vec[(threadID / rows) * vec_mRows];
    }
  }

  __global__ void rowWiseMult_kernel_double(double* in_mat, double* out_mat, double * vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    if (threadID < rows * cols){
      out_mat[(threadID / rows) * out_mRows + (threadID % rows)] = in_mat[(threadID / rows) * mRows + (threadID % rows)] * vec[(threadID / rows) * vec_mRows];
    }
  }

//Kernels for finding the sum of each column of a matrix, after squaring each entry elementwise (DEPRECATED)
  __global__ void colSqSum_kernel_float(float *A, int n, int features, float *result, float scalar, int mRows, int out_mRows){
   int index_x = blockDim.x * blockIdx.x + threadIdx.x;
   int index_y = blockDim.y * blockIdx.y + threadIdx.y;
   int grid_width = gridDim.x * blockDim.x;
   int threadID = index_y * grid_width + index_x;
   if (threadID < n){
      result[threadID] = 0;
      for(int i = 0; i < features; ++i){
         result[threadID * out_mRows] += A[threadID * mRows + i] * A[threadID * mRows + i];
      }
      result[threadID * out_mRows] *= scalar;
    }
  }

  __global__ void colSqSum_kernel_double(double *A, int n, int features, double *result, double scalar, int mRows, int out_mRows){
   int index_x = blockDim.x * blockIdx.x + threadIdx.x;
   int index_y = blockDim.y * blockIdx.y + threadIdx.y;
   int grid_width = gridDim.x * blockDim.x;
   int threadID = index_y * grid_width + index_x;
   if (threadID < n){
      result[threadID] = 0;
      for(int i = 0; i < features; ++i){
  result[threadID * out_mRows] += A[threadID * mRows + i] * A[threadID * mRows + i];
      }
      result[threadID * out_mRows] *= scalar;
    }
  }

//Kernels for finding the sum of a Matrix's columns (DEPRECATED, now achieved with matrix multiplication)
  __global__ void colSum_kernel_float(float *A, int n, int features, float *result, float scalar, int mRows, int out_mRows){
   int index_x = blockDim.x * blockIdx.x + threadIdx.x;
   int index_y = blockDim.y * blockIdx.y + threadIdx.y;
   int grid_width = gridDim.x * blockDim.x;
   int threadID = index_y * grid_width + index_x;
   if (threadID < n){
      result[threadID] = 0;
      for(int i = 0; i < features; ++i){
         result[threadID * out_mRows] += A[threadID * mRows + i];
      }
      result[threadID * out_mRows] *= scalar;
    }
  }

  __global__ void colSum_kernel_double(double *A, int n, int features, double *result, double scalar, int mRows, int out_mRows){
   int index_x = blockDim.x * blockIdx.x + threadIdx.x;
   int index_y = blockDim.y * blockIdx.y + threadIdx.y;
   int grid_width = gridDim.x * blockDim.x;
   int threadID = index_y * grid_width + index_x;
   if (threadID < n){
      result[threadID] = 0;
      for(int i = 0; i < features; ++i){
  result[threadID * out_mRows] += A[threadID * mRows + i];
      }
      result[threadID * out_mRows] *= scalar;
    }
  }


//Kernel for gathering the elements of one matrix into another, based on a map of indices  
  __global__ void device_kernel_gather_float(int* map, float* src, float* dst, int rows, int mRows, int out_mRows, int mapSize){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int mapCol = threadID / rows;
    int mapRow = threadID % rows;
    if (threadID < mapSize*rows){
      dst[mapCol * out_mRows + mapRow] = src[map[mapCol] * mRows + mapRow];
    }
  }
  
  __global__ void device_kernel_gather_double(int* map, double* src, double* dst, int rows, int mRows, int out_mRows, int mapSize){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int mapCol = threadID / rows;
    int mapRow = threadID % rows;
    if (threadID < mapSize*rows){
      dst[mapCol * out_mRows + mapRow] = src[map[mapCol] * mRows + mapRow];
    }
  }

//kernels for elementwise exponentiation
  __global__ void device_kernel_exp_float(float* in, float *out, int n, int rows, int mRows, int out_mRows, float gamma){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = exp(-gamma*in[col * mRows + row]);
    }
  }
  
  __global__ void device_kernel_exp_double(double* in, double *out, int n, int rows, int mRows, int out_mRows, double gamma){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = exp(-gamma*in[col * mRows + row]);
    }
  }

//Kernels for taking the elementwise tanh()
    __global__ void device_kernel_tanh_float(float* in, float *out, int n, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = tanh(in[col * mRows + row]);
    }
  }
  
  __global__ void device_kernel_tanh_double(double* in, double *out, int n, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = tanh(in[col * mRows + row]);
    }
  }

//Kernels for calculating which basis vector should be added next
  __global__ void device_kernel_chooseNextHelper_float(float* d_x, int d_xInd, float* g, float* h, int select, float* d_out_minus1, float* dK2, int dK2rows, int dK2cols){
    
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
   
    float xend = -g[select] / h[select];
 
    if(threadID == 0){
      d_x[d_xInd] = xend;
    }
    
    if (threadID < dK2rows){                                                                    
      d_out_minus1[threadID] = d_out_minus1[threadID] + dK2[select * dK2rows + threadID] * xend;
    }                                                                                          
  }

  __global__ void device_kernel_chooseNextHelper_double(double* d_x, int d_xInd, double* g, double* h, int select, double* d_out_minus1, double* dK2, int dK2rows, int dK2cols){
    
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
   
    float xend = -g[select] / h[select];
 
    if(threadID == 0){
      d_x[d_xInd] = xend;
    }
    
    if (threadID < dK2rows){                                                                    
      d_out_minus1[threadID] = d_out_minus1[threadID] + dK2[select * dK2rows + threadID] * xend;
    }                                                                                          
  }
  
//Kernels for adding/subracting two matrices
  __global__ void device_kernel_addSubMatrix(double* a, double* b, double* out, double scalar, int rows, int cols, int a_mRows, int b_mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;

    if(threadID < rows*cols){
      out[col * out_mRows + row] = a[col * a_mRows + row] + scalar * b[col * b_mRows + row];
    }
  }

  __global__ void device_kernel_addSubMatrix(float* a, float* b, float* out, float scalar, int rows, int cols, int a_mRows, int b_mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;

    if(threadID < rows*cols){
      out[col * out_mRows + row] = a[col * a_mRows + row] + scalar * b[col * b_mRows + row];
    }
  }

//kernels for elementwise multiplication of two matrices
  __global__ void eWiseMult_kernel_float(float* in1, float* in2, float* out, int length, float pow1, float pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
      
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;
      int index_y = blockDim.y * blockIdx.y + threadIdx.y;
      int grid_width = gridDim.x * blockDim.x;
      int threadID = index_y * grid_width + index_x;
     
      if (threadID < length){                                                                    
        out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in1[(threadID / rows) * in1_mRows + (threadID % rows)], pow1) * pow(in2[(threadID / rows) * in2_mRows + (threadID % rows)], pow2);                                     
      }                                                                                          
  } 

  __global__ void eWiseMult_kernel_double(double* in1, double* in2, double* out, int length, double pow1, double pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
      
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;
      int index_y = blockDim.y * blockIdx.y + threadIdx.y;
      int grid_width = gridDim.x * blockDim.x;
      int threadID = index_y * grid_width + index_x;
     
      if (threadID < length){                                                                    
        out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in1[(threadID / rows) * in1_mRows + (threadID % rows)], pow1) * pow(in2[(threadID / rows) * in2_mRows + (threadID % rows)], pow2);                                     
      }                                                                                          
  } 

//Kernels for summing columns at the indices indicated by map
   __global__ void device_kernel_gatherSum_float(int* map, float* src, float* dst, int rows, int mRows, int out_mRows, int mapSize, int outputRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int mapRow = threadID;
    int stride = outputRows == 1 ? out_mRows : 1;

    if (threadID < rows){
      for (int mapCol = 0; mapCol < mapSize; ++mapCol)
      {
        dst[mapRow * stride] += src[map[mapCol] * mRows + mapRow];
      }
    }
  }
  
  __global__ void device_kernel_gatherSum_double(int* map, double* src, double* dst, int rows, int mRows, int out_mRows, int mapSize, int outputRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int mapRow = threadID;
    int stride = outputRows == 1 ? out_mRows : 1;

    if (threadID < rows){
      for (int mapCol = 0; mapCol < mapSize; ++mapCol)
      {
        dst[mapRow * stride] += src[map[mapCol] * mRows + mapRow];
      }
    }
  }


//The pinned methods that follow allow these operations to be performed on matrices that don't fit on the gpu (in terms of memory)
//pinned kernel  ewise stuff (TODO:from whence did this come?)
__global__ void eWiseOp_kernel_pinned_float(float* in, float* out, int length, float mult, float add, float pow1, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    
    if (threadID < length){
      if(mult == 0 || pow1 == 1){
          out[(threadID / rows) * out_mRows + (threadID % rows)] = in[(threadID / rows) * mRows + (threadID % rows)] * mult + add;
        } else if (pow1 == 2.0){
          float tempVal = in[(threadID / rows) * mRows + (threadID % rows)];
          out[(threadID / rows) * out_mRows + (threadID % rows)] = tempVal * tempVal * mult + add;
        } else {
          out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in[(threadID / rows) * mRows + (threadID % rows)], pow1) * mult + add;
        }
    }    
  }

  __global__ void eWiseOp_kernel_pinned_double(double* in, double* out, int length, double mult, double add, double pow1, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    
    if (threadID < length){
      if(mult == 0 || pow1 == 1){
          out[(threadID / rows) * out_mRows + (threadID % rows)] = in[(threadID / rows) * mRows + (threadID % rows)] * mult + add;
        } else if (pow1 == 2.0){
          double tempVal = in[(threadID / rows) * mRows + (threadID % rows)];
          out[(threadID / rows) * out_mRows + (threadID % rows)] = tempVal * tempVal * mult + add;
        } else {
          out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in[(threadID / rows) * mRows + (threadID % rows)], pow1) * mult + add;
        }
    }    
  }

//kernel for pinned elementwise exponentiation
__global__ void device_kernel_exp_pinned_float(float* in, float *out, int n, int rows, int mRows, int out_mRows, float gamma){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = exp(-gamma*in[col * mRows + row]);
    }
  }
  
  __global__ void device_kernel_exp_pinned_double(double* in, double *out, int n, int rows, int mRows, int out_mRows, double gamma){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = exp(-gamma*in[col * mRows + row]);
    }
  }

//kernels for pinned elementwise tanh
  __global__ void device_kernel_tanh_pinned_float(float* in, float *out, int n, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = tanh(in[col * mRows + row]);
    }
  }
  
  __global__ void device_kernel_tanh_pinned_double(double* in, double *out, int n, int rows, int mRows, int out_mRows){
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadID = index_y * grid_width + index_x;
    int col = threadID / rows;
    int row = threadID % rows;
    if (threadID < n * rows){
      out[col * out_mRows + row] = tanh(in[col * mRows + row]);
    }
  }

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
//                                DEVICE_WRAPPERS                              //
/////////////////////////////////////////////////////////////////////////////////

//Wrappers for Matrix-Matrix Multiplication  
  int device_dgemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, double alpha, double* a, int lda, double * b, int ldb, double beta, double* c, int ldc){
    if(m == 0 || n == 0 || k == 0){
        return 0;
    }

    cublasOperation_t Atran = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t Btran = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    params.context.getCuBlasHandle();
    CUBLAS_CHECK(cublasDgemm(params.context.getCuBlasHandle(), Atran, Btran, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
    return 0;//BLAS_SUCCESS;
  }

  int device_sgemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, float alpha, float* a, int lda, float * b, int ldb, float beta, float* c, int ldc){
    if(m == 0 || n == 0 || k == 0){
        return 0;
    }

    cublasOperation_t Atran = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t Btran = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(cublasSgemm(params.context.getCuBlasHandle(), Atran, Btran, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
    return 0;//BLAS_SUCCESS;
  }

//Wrappers for A = alpha * x * y' + A
  int device_dger(DeviceParams params, int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda){
    if(m == 0 || n == 0){
      return 0;
    }
    CUBLAS_CHECK(cublasDger(params.context.getCuBlasHandle(), m, n, &alpha, x, incx, y, incy, a, lda));
    return 0;//BLAS_SUCCESS;
  }

  int device_sger(DeviceParams params, int m, int n, float alpha, float* x, int incx, float* y, int incy, float* a, int lda){
   if(m == 0 || n == 0){
      return 0;
    }
    CUBLAS_CHECK(cublasSger(params.context.getCuBlasHandle(), m, n, &alpha, x, incx, y, incy, a, lda));
    return 0;//BLAS_SUCCESS;
  }

//Wrappers for solving systems of linear equations
  extern "C" {
      void slaswp_(int* n, float *a, int* lda, int* k1, int* k2, int* ipiv, int* incx);
      void dlaswp_(int* n, double *a, int* lda, int* k1, int* k2, int* ipiv, int* incx);
  }

  int device_dgesv(DeviceParams params, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb){
    if(n == 0 || nrhs == 0){
      return 0;
    }

    int* info;
    double** array = 0;

    int* ipiv_host = new int[n];
    double* b_host = new double[n * nrhs];
    int one = 1;

    CUDA_CHECK(cudaMalloc((void**)&info, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&array, sizeof(double*)));
    CUDA_CHECK(cudaMemcpy(array, &a, sizeof(double*), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasDgetrfBatched(params.context.getCuBlasHandle(), n, array, lda, ipiv, info, 1));

    CUDA_CHECK(cudaMemcpy(ipiv_host, ipiv, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasGetMatrix(n, nrhs, sizeof(double), b, ldb, b_host, n));

    dlaswp_(&nrhs, b_host, &n, &one, &n, ipiv_host, &one);

    CUBLAS_CHECK(cublasSetMatrix(n, nrhs, sizeof(double), b_host, n, b, ldb));

    double alpha = 1;
    CUBLAS_CHECK(cublasDtrsm(params.context.getCuBlasHandle(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, nrhs, &alpha, a, lda, b, ldb));
    CUBLAS_CHECK(cublasDtrsm(params.context.getCuBlasHandle(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs, &alpha, a, lda, b, ldb));

    CUDA_CHECK(cudaFree(info));
    CUDA_CHECK(cudaFree(array));

    delete [] ipiv_host;
    delete [] b_host;

    return 0;
  }

  int device_sgesv(DeviceParams params, int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb){
   if(n == 0 || nrhs == 0){
      return 0;
    }

    int* info;
    float** array = 0;

    int* ipiv_host = new int[n];
    float* b_host = new float[n * nrhs];
    int one = 1;

    CUDA_CHECK(cudaMalloc((void**)&info, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&array, sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(array, &a, sizeof(float*), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasSgetrfBatched(params.context.getCuBlasHandle(), n, array, lda, ipiv, info, 1));

    CUDA_CHECK(cudaMemcpy(ipiv_host, ipiv, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasGetMatrix(n, nrhs, sizeof(float), b, ldb, b_host, n));

    slaswp_(&nrhs, b_host, &n, &one, &n, ipiv_host, &one);

    CUBLAS_CHECK(cublasSetMatrix(n, nrhs, sizeof(float), b_host, n, b, ldb));

    float alpha = 1;
    CUBLAS_CHECK(cublasStrsm(params.context.getCuBlasHandle(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, nrhs, &alpha, a, lda, b, ldb));
    CUBLAS_CHECK(cublasStrsm(params.context.getCuBlasHandle(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs, &alpha, a, lda, b, ldb));

    CUDA_CHECK(cudaFree(info));
    CUDA_CHECK(cudaFree(array));

    delete [] ipiv_host;
    delete [] b_host;

    return 0;
  }


    
//Wrappers for performing elementwise addition/subraction, multiplication/division, powers/roots on a matrix
  int device_ewiseOp(DeviceParams params, float* in, float* out, int length, float add, float mult,  float pow1, int rows, int mRows, int out_mRows){
    if(length == 0){
      return 0;
    }
    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    eWiseOp_kernel_float<<<blockspergrid, threadsperblock>>>(in, out, length, mult, add, pow1, rows, mRows, out_mRows);
  return 0;
}

  int device_ewiseOp(DeviceParams params, double* in, double* out, int length, double add, double mult,  double pow1, int rows, int mRows, int out_mRows){
    if(length == 0){
      return 0;
    }
    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    eWiseOp_kernel_double<<<blockspergrid, threadsperblock>>>(in, out, length, mult, add, pow1, rows, mRows, out_mRows);
  return 0;
}


// Wrappers for computing the sum of a matrix's columns, after each element has been elementwise squared
  int device_colSqSum(DeviceParams params, float* A, int n, int features, float* result, float scalar, int mRows, int out_mRows){
    if(n == 0 ||features == 0){
      return 0;
    }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (n * features + threadsperblock - 1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;

    float* B = 0;
    cudaError_t err = cudaMalloc((void**)&B, n*features*sizeof(float));
    if(err == cudaSuccess){
      eWiseSquare_kernel_float<<<blockspergrid, threadsperblock>>>(A, B, n * features, features, mRows, features);
      device_colSum(params, B, n, features, result, scalar, features, out_mRows);
      cudaFree(B);
    } else {
      threadsperblock = 256;
      blockspergrid.x = (n+threadsperblock-1) / threadsperblock / 100 + 1;
      blockspergrid.y = 100;
      colSqSum_kernel_float<<<blockspergrid, threadsperblock>>>(A, n, features, result, scalar, mRows, out_mRows);
    }

    return 0;
  }

  int device_colSqSum(DeviceParams params, double* A, int n, int features, double* result, double scalar, int mRows, int out_mRows){
    if(n == 0 ||features == 0){
      return 0;
    }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (n * features + threadsperblock - 1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;

    double* B = 0;
    cudaError_t err = cudaMalloc((void**)&B, n*features*sizeof(double));
    if(err == cudaSuccess){
      eWiseSquare_kernel_double<<<blockspergrid, threadsperblock>>>(A, B, n * features, features, mRows, features);
      device_colSum(params, B, n, features, result, scalar, features, out_mRows);
      cudaFree(B);
    } else {
      threadsperblock = 256;
      blockspergrid.x = (n+threadsperblock-1) / threadsperblock / 100 + 1;
      blockspergrid.y = 100;
      colSqSum_kernel_double<<<blockspergrid, threadsperblock>>>(A, n, features, result, scalar, mRows, out_mRows);
    }

    return 0;
  }

//Wrappers for finding the sum of a matrix's columns
   int device_colSum(DeviceParams params, float* A, int n, int features, float* result, float scalar, int mRows, int out_mRows){
    if(n == 0 ||features == 0){
      return 0;
    }

    //instantiate matrix of ones
     float* onesMatrix = 0;
     cudaError_t err = cudaMalloc((void**)&onesMatrix,features*sizeof(float));
     if(err == cudaSuccess){
       device_ewiseOp(params, onesMatrix, onesMatrix, features, 1, 0, 0, 1, 1, 1);
       
       //sums columns
       CUBLAS_CHECK(device_sgemm(params, false, false, 1, n, features, scalar, onesMatrix, 1, A, mRows, 0, result, out_mRows));
       cudaFree(onesMatrix);
     } else {
       int threadsperblock = 256;
       dim3 blockspergrid;
       blockspergrid.x = (n+threadsperblock-1) / threadsperblock / 100 + 1;
       blockspergrid.y = 100;
       colSum_kernel_float<<<blockspergrid, threadsperblock>>>(A, n, features, result, scalar, mRows, out_mRows);
     }

     return 0;
  }

 int device_colSum(DeviceParams params, double* A, int n, int features, double* result, double scalar, int mRows, int out_mRows){
   if(n == 0 ||features == 0){
      return 0;
    }

   //instantiate matrix of ones
   double* onesMatrix = 0;
   cudaError_t err = cudaMalloc((void**)&onesMatrix,features*sizeof(double));
   if(err == cudaSuccess){
     device_ewiseOp(params, onesMatrix, onesMatrix, features, 1, 0, 0, 1, 1, 1);
     
     //sums columns
     CUBLAS_CHECK(device_dgemm(params, false, false, 1, n, features, scalar, onesMatrix, 1, A, mRows, 0, result, out_mRows));
     cudaFree(onesMatrix);
   } else {
     int threadsperblock = 256;
     dim3 blockspergrid;
     blockspergrid.x = (n+threadsperblock-1) / threadsperblock / 100 + 1;
     blockspergrid.y = 100;
     colSum_kernel_double<<<blockspergrid, threadsperblock>>>(A, n, features, result, scalar, mRows, out_mRows);
   }

   return 0;
  }




// wrappers for multiplying the entries in a matrix's columns/rows elementwise by the elements in a vector
  int device_colWiseMult(DeviceParams params, float* mat, float* out, float* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    if(rows == 0 ||cols == 0){
        return 0;
      }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (rows * cols + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    colWiseMult_kernel_float<<<blockspergrid, threadsperblock>>>(mat, out, vec, rows, cols, mRows, out_mRows, vec_mRows);
    return 0;
  }

  int device_colWiseMult(DeviceParams params, double* mat, double* out, double* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    if(rows == 0 ||cols == 0){
        return 0;
      }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (rows * cols + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    colWiseMult_kernel_double<<<blockspergrid, threadsperblock>>>(mat, out, vec, rows, cols, mRows, out_mRows, vec_mRows);
    return 0;
  }

  int device_rowWiseMult(DeviceParams params, float* mat, float* out, float* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    if(rows == 0 ||cols == 0){
        return 0;
      }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (rows * cols + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    rowWiseMult_kernel_float<<<blockspergrid, threadsperblock>>>(mat, out, vec, rows, cols, mRows, out_mRows, vec_mRows);
    return 0;
  }

  int device_rowWiseMult(DeviceParams params, double* mat, double* out, double* vec, int rows, int cols, int mRows, int out_mRows, int vec_mRows){
    if(rows == 0 ||cols == 0){
        return 0;
      }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (rows * cols + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    rowWiseMult_kernel_double<<<blockspergrid, threadsperblock>>>(mat, out, vec, rows, cols, mRows, out_mRows, vec_mRows);
    return 0;
  }

//Kernels for the elementwise division of two matrices (each element of index x in matrix 1, is divided by the element at index x of matrix 2)
  __global__ void eWiseDiv_kernel_float(float* in1, float* in2, float* out, int length, float pow1, float pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
      
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;
      int index_y = blockDim.y * blockIdx.y + threadIdx.y;
      int grid_width = gridDim.x * blockDim.x;
      int threadID = index_y * grid_width + index_x;
     
      if (threadID < length){                                                                    
        out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in1[(threadID / rows) * in1_mRows + (threadID % rows)], pow1) / pow(in2[(threadID / rows) * in2_mRows + (threadID % rows)], pow2);                                     
      }                                                                                          
  } 

  __global__ void eWiseDiv_kernel_double(double* in1, double* in2, double* out, int length, double pow1, double pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
      
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;
      int index_y = blockDim.y * blockIdx.y + threadIdx.y;
      int grid_width = gridDim.x * blockDim.x;
      int threadID = index_y * grid_width + index_x;
     
      if (threadID < length){                                                                    
        out[(threadID / rows) * out_mRows + (threadID % rows)] = pow(in1[(threadID / rows) * in1_mRows + (threadID % rows)], pow1) / pow(in2[(threadID / rows) * in2_mRows + (threadID % rows)], pow2);                                     
      }                                                                                          
  } 

//Wrappers for the elementwise division of two matrices (each element of index x in matrix 1, is divided by the element at index x of matrix 2)
int device_eWiseDiv(DeviceParams params, float* in1, float* in2, float* out, int length, float pow1, float pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
  if(length == 0){
      return 0;
    }

  int threadsperblock = 256;
  dim3 blockspergrid;
  blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
  blockspergrid.y = 100;
  eWiseDiv_kernel_float<<<blockspergrid, threadsperblock>>>(in1, in2, out, length, pow1, pow2, rows, in1_mRows, in2_mRows, out_mRows);
  return 0;
}

int device_eWiseDiv(DeviceParams params, double* in1, double* in2, double* out, int length, double pow1, double pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
  if(length == 0){
      return 0;
    }

  int threadsperblock = 256;
  dim3 blockspergrid;
  blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
  blockspergrid.y = 100;
  eWiseDiv_kernel_double<<<blockspergrid, threadsperblock>>>(in1, in2, out, length, pow1, pow2, rows, in1_mRows, in2_mRows, out_mRows);
  return 0;
}



//Wrapper for gathering the elements of one matrix into another, based on a map of indices  
  int device_gather(DeviceParams params, int* map, float* src, float* dst, int rows, int mRows, int out_mRows, int mapSize){
    if(rows == 0 || mapSize == 0){
      return 0;
    }

     int threadsperblock = 256;                                    
     dim3 blockspergrid;
     blockspergrid.x = (rows * mapSize +threadsperblock-1) / threadsperblock / 100 +1;
     blockspergrid.y = 100;
     device_kernel_gather_float<<<blockspergrid, threadsperblock>>>(map, src, dst, rows, mRows, out_mRows, mapSize);
     return 0;
  }

  int device_gather(DeviceParams params, int* map, double* src, double* dst, int rows, int mRows, int out_mRows, int mapSize){
    if(rows == 0 || mapSize == 0){
      return 0;
    }
    
     int threadsperblock = 256;                                    
     dim3 blockspergrid;
     blockspergrid.x = (rows * mapSize +threadsperblock-1) / threadsperblock / 100 +1;
     blockspergrid.y = 100;
     device_kernel_gather_double<<<blockspergrid, threadsperblock>>>(map, src, dst, rows, mRows, out_mRows, mapSize);
     return 0;
  }



//Wrappers for elementwise exponentiation
  int device_exp(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows, float gamma){
    if(rows == 0 || n == 0){
      return 0;
    }
    

    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_exp_float<<<blockspergrid,threadsperblock>>>(in, out, n, rows, mRows, out_mRows, gamma);
    return 0;
  } 

  int device_exp(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows, double gamma){
    if(rows == 0 || n == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_exp_double<<<blockspergrid, threadsperblock>>>(in, out, n, rows, mRows, out_mRows, gamma);
    return 0;
  } 


//Wrappers for taking the elementwise tanh
  int device_tanh(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows){
    if(rows == 0 || n == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_tanh_float<<<blockspergrid,threadsperblock>>>(in, out, n, rows, mRows, out_mRows);
    return 0;
  } 

  int device_tanh(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows){
    if(rows == 0 || n == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_tanh_double<<<blockspergrid, threadsperblock>>>(in, out, n, rows, mRows, out_mRows);
    return 0;
  } 
  



//Wrappers for calculating which basis vector should be added next
  int device_chooseNextHelper(DeviceParams params, float* d_x, int d_xInd, float* g, float* h, int select, float* d_out_minus1, float* dK2, int dK2rows, int dK2cols){
    if(dK2cols == 0 || dK2rows == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    blockspergrid.x = (dK2cols + threadsperblock-1) / threadsperblock /100 + 1;
    blockspergrid.y = 100;

    device_kernel_chooseNextHelper_float<<<blockspergrid, threadsperblock>>>(d_x, d_xInd,g, h, select, d_out_minus1, dK2, dK2rows, dK2cols);
    return 0;
  }

  int device_chooseNextHelper(DeviceParams params, double* d_x, int d_xInd, double* g, double* h, int select, double* d_out_minus1, double* dK2, int dK2rows, int dK2cols){
    if(dK2cols == 0 || dK2rows == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    blockspergrid.x = (dK2cols + threadsperblock-1) / threadsperblock /100 + 1;
    blockspergrid.y = 100;

    device_kernel_chooseNextHelper_double<<<blockspergrid, threadsperblock>>>(d_x, d_xInd,g, h, select, d_out_minus1, dK2, dK2rows, dK2cols);
    return 0;
  }


//Wrappers for adding/subracting two matrices
  int device_addMatrix(DeviceParams params, double* a, double* b, double* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows){
    if(cols == 0 || rows == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    blockspergrid.x = (rows*cols +threadsperblock-1)/threadsperblock/100+1;
    blockspergrid.y = 100;
    device_kernel_addSubMatrix<<<blockspergrid, threadsperblock>>>(a, b, out, 1, rows, cols, a_mRows, b_mRows, out_mRows);
    return 0;
  }



  int device_subMatrix(DeviceParams params, double* a, double* b, double* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows){
    if(cols == 0 || rows == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    blockspergrid.x = (rows*cols +threadsperblock-1)/threadsperblock/100+1;
    blockspergrid.y = 100;
    device_kernel_addSubMatrix<<<blockspergrid, threadsperblock>>>(a, b, out, -1, rows, cols, a_mRows, b_mRows, out_mRows);
    return 0;
  }

  int device_addMatrix(DeviceParams params, float* a, float* b, float* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows){
    if(cols == 0 || rows == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    blockspergrid.x = (rows*cols+threadsperblock-1)/threadsperblock/100+1;
    blockspergrid.y = 100;
    device_kernel_addSubMatrix<<<blockspergrid, threadsperblock>>>(a, b, out, 1, rows, cols, a_mRows, b_mRows, out_mRows);
    return 0;
  }



  int device_subMatrix(DeviceParams params, float* a, float* b, float* out, int rows, int cols, int a_mRows, int b_mRows, int out_mRows){
    if(cols == 0 || rows == 0){
      return 0;
    }

    int threadsperblock(256);
    dim3 blockspergrid;
    blockspergrid.x = (rows*cols +threadsperblock-1)/threadsperblock/100+1;
    blockspergrid.y = 100;
    device_kernel_addSubMatrix<<<blockspergrid, threadsperblock>>>(a, b, out, -1, rows, cols, a_mRows, b_mRows, out_mRows);
    return 0;
  }



//Wrappers for elementwise multiplication of two matrices
  int device_eWiseMult(DeviceParams params, float* in1, float* in2, float* out, int length, float pow1, float pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
    if(length == 0){
      return 0;
    }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    eWiseMult_kernel_float<<<blockspergrid, threadsperblock>>>(in1, in2, out, length, pow1, pow2, rows, in1_mRows, in2_mRows, out_mRows);
    return 0;
  }

  int device_eWiseMult(DeviceParams params, double* in1, double* in2, double* out, int length, double pow1, double pow2, int rows, int in1_mRows, int in2_mRows, int out_mRows){
    if(length == 0){
      return 0;
    }

    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    eWiseMult_kernel_double<<<blockspergrid, threadsperblock>>>(in1, in2, out, length, pow1, pow2, rows, in1_mRows, in2_mRows, out_mRows);
    return 0;
  }

//Wrappers for transposing matrices
  int device_transpose(DeviceParams params, float* in, float* out, int cols, int rows, int mRows, int out_mRows){
    if(rows == 0 || cols == 0){
      return 0;
    }

    float alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasSgeam(params.context.getCuBlasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, cols, rows, &alpha, in, mRows, &beta, in, mRows, out, out_mRows));

    return 0;
  } 

  int device_transpose(DeviceParams params, double* in, double* out, int cols, int rows, int mRows, int out_mRows){
    if(rows == 0 || cols == 0){
      return 0;
    }

    double alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasDgeam(params.context.getCuBlasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, cols, rows, &alpha, in, mRows, &beta, in, mRows, out, out_mRows));

    return 0;
  } 


//Wrappers for summing columns at the indices indicated by map
  int device_gatherSum(DeviceParams params, int* map, float* src, float* dst, int rows, int mRows, int out_mRows, int mapSize, int outputRows){
    if(rows == 0 || mapSize == 0){
      return 0;
    }

    int threadsperblock = 256;                                    
    dim3 blockspergrid;
    blockspergrid.x = (rows +threadsperblock-1) / threadsperblock / 100 +1;
    blockspergrid.y = 100;
    device_kernel_gatherSum_float<<<blockspergrid, threadsperblock>>>(map, src, dst, rows, mRows, out_mRows, mapSize, outputRows);
    return 0;
  }

  int device_gatherSum(DeviceParams params, int* map, double* src, double* dst, int rows, int mRows, int out_mRows, int mapSize, int outputRows){
    if(rows == 0 || mapSize == 0){
    return 0;
  }
   int threadsperblock = 256;                                    
   dim3 blockspergrid;
   blockspergrid.x = (rows +threadsperblock-1) / threadsperblock / 100 +1;
   blockspergrid.y = 100;
   device_kernel_gatherSum_double<<<blockspergrid, threadsperblock>>>(map, src, dst, rows, mRows, out_mRows, mapSize, outputRows);
   return 0;
  }

  template<class N, class T>
  int device_convert(DeviceParams params, T* in, N* out, int rows, int cols, int mRows){
    return device_convert_helper(params, in, out, rows, cols, mRows);
  }

  //force double/float instantiation
  template int device_convert<double, float>(DeviceParams params, float* in, double* out, int rows, int cols, int mRows);
  template int device_convert<float, double>(DeviceParams params, double* in, float* out, int rows, int cols, int mRows);
  template int device_convert<double, double>(DeviceParams params, double* in, double* out, int rows, int cols, int mRows);
  template int device_convert<float, float>(DeviceParams params, float* in, float* out, int rows, int cols, int mRows);


  //Here and below is the device kernel code
  int device_gemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, double alpha, double* a, int lda, double * b, int ldb, double beta, double* c, int ldc) {
  cublasOperation_t Atran = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t Btran = transb ? CUBLAS_OP_T : CUBLAS_OP_N;  
  CUBLAS_CHECK(cublasDgemm(params.context.getCuBlasHandle(), Atran, Btran, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
  return 0;//BLAS_SUCCESS;
}

int device_gemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, float alpha, float* a, int lda, float * b, int ldb, float beta, float* c, int ldc){
  cublasOperation_t Atran = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t Btran = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemm(params.context.getCuBlasHandle(), Atran, Btran, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
  return 0;//BLAS_SUCCESS;
}

  int device_geam(DeviceParams params, bool transa, bool transb, int m, int n, double alpha, double* a, int lda, double beta,  double * b, int ldb,double* c, int ldc) {
  cublasOperation_t Atran = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t Btran = transb ? CUBLAS_OP_T : CUBLAS_OP_N;  
  CUBLAS_CHECK(cublasDgeam(params.context.getCuBlasHandle(), Atran, Btran, m, n, &alpha, a, lda, &beta, b, ldb, c, ldc));
  return 0;//BLAS_SUCCESS;
}

int device_geam(DeviceParams params, bool transa, bool transb, int m, int n, float alpha, float* a, int lda, float beta, float * b, int ldb, float* c, int ldc){
  cublasOperation_t Atran = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t Btran = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgeam(params.context.getCuBlasHandle(), Atran, Btran, m, n, &alpha, a, lda,  &beta, b, ldb, c, ldc));
  return 0;//BLAS_SUCCESS;
}

int device_ger(DeviceParams params, int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda){
  CUBLAS_CHECK(cublasDger(params.context.getCuBlasHandle(), m, n, &alpha, x, incx, y, incy, a, lda));
  return 0;//BLAS_SUCCESS;
}

 int device_ger(DeviceParams params, int m, int n, float alpha, float* x, int incx, float* y, int incy, float* a, int lda){
  CUBLAS_CHECK(cublasSger(params.context.getCuBlasHandle(), m, n, &alpha, x, incx, y, incy, a, lda));
  return 0;//BLAS_SUCCESS;
 }

//The pinned methods that follow allow these operations to be performed on matrices that don't fit on the gpu (in terms of memory)
//wrapper for pinned elementwise operations
  int device_ewiseOp_stream(DeviceParams params, float* in, float* out, int length, float add, float mult,  float pow1, int rows, int mRows, int out_mRows, cudaStream_t &stream){
    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    eWiseOp_kernel_pinned_float<<<blockspergrid, threadsperblock, 0, stream>>>(in, out, length, mult, add, pow1, rows, mRows, out_mRows);
  return 0;
}

  int device_ewiseOp_stream(DeviceParams params, double* in, double* out, int length, double add, double mult,  double pow1, int rows, int mRows, int out_mRows, cudaStream_t &stream){
    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (length + threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    eWiseOp_kernel_pinned_double<<<blockspergrid, threadsperblock, 0, stream>>>(in, out, length, mult, add, pow1, rows, mRows, out_mRows);
  return 0;
}




//wrappers for pinned elementwise exponentiation
  int device_exp_stream(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows, float gamma, cudaStream_t &stream){
    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_exp_pinned_float<<<blockspergrid,threadsperblock, 0, stream>>>(in, out, n, rows, mRows, out_mRows, gamma);
    return 0;
  } 

  int device_exp_stream(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows, double gamma, cudaStream_t &stream){
    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_exp_pinned_double<<<blockspergrid, threadsperblock, 0, stream>>>(in, out, n, rows, mRows, out_mRows, gamma);
    return 0;
  } 



//wrappers for pinned elementwise tanh
  int device_tanh_stream(DeviceParams params, float* in, float* out, int n, int rows, int mRows, int out_mRows, cudaStream_t &stream){
    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_tanh_pinned_float<<<blockspergrid,threadsperblock, 0, stream>>>(in, out, n, rows, mRows, out_mRows);
    return 0;
  } 

  int device_tanh_stream(DeviceParams params, double* in, double* out, int n, int rows, int mRows, int out_mRows, cudaStream_t &stream){
    int threadsperblock(256);
    dim3 blockspergrid;
    
    blockspergrid.x =((n * rows + threadsperblock-1) / threadsperblock)/100 + 1;
    blockspergrid.y = 100;
    device_kernel_tanh_pinned_double<<<blockspergrid, threadsperblock, 0, stream>>>(in, out, n, rows, mRows, out_mRows);
    return 0;
  } 


}

#endif

//TODO: 
//  -Improve memory access patterns for all kernels
//  -Add loop option to all kernels to reduce # of threads

// Code graveyard:
  /*  int device_colSum(DeviceParams params, float* A, int n, int features, float* result, float scalar, int mRows, int out_mRows){
    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (n+threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    //    blockspergrid.x = params.gridDim/100+1;
    //blockspergrid.y = 100;
    colSum_kernel_float<<<blockspergrid, threadsperblock>>>(A, n, features, result, scalar, mRows, out_mRows);
    return 0;
  }

  int device_colSum(DeviceParams params, double* A, int n, int features, double* result, double scalar, int mRows, int out_mRows){
    int threadsperblock = 256;
    dim3 blockspergrid;
    blockspergrid.x = (n+threadsperblock-1) / threadsperblock / 100 + 1;
    blockspergrid.y = 100;
    colSum_kernel_double<<<blockspergrid, threadsperblock>>>(A, n, features, result, scalar, mRows, out_mRows);
    return 0;
    }*/


  // __global__ void device_kernel_transpose_float(float* in, float *out, int cols, int rows, int mRows, int out_mRows){
  //   int index_x = blockDim.x * blockIdx.x + threadIdx.x;
  //   int index_y = blockDim.y * blockIdx.y + threadIdx.y;
  //   int grid_width = gridDim.x * blockDim.x;
  //   int threadID = index_y * grid_width + index_x;
  //   int col = threadID / rows;
  //   int row = threadID % rows;
  //   if (threadID < cols * rows){
  //     out[row * out_mRows + col] = in[col * mRows + row];
  //   }
  // }
  
  // __global__ void device_kernel_transpose_double(double* in, double *out, int cols, int rows, int mRows, int out_mRows){
  //   int index_x = blockDim.x * blockIdx.x + threadIdx.x;
  //   int index_y = blockDim.y * blockIdx.y + threadIdx.y;
  //   int grid_width = gridDim.x * blockDim.x;
  //   int threadID = index_y * grid_width + index_x;
  //   int col = threadID / rows;
  //   int row = threadID % rows;
  //   if (threadID < cols * rows){
  //     out[row * out_mRows + col] = in[col * mRows + row];
  //   }
  // }