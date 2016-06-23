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
#include "abstract_matrix.h"

#include <vector>

#ifdef CPP11
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#endif

namespace lasp{

using namespace std;

//TODO: host remapping, CUDA 6.0?...
template<class T>
  int pinned_kernel_multiply(DeviceParams params, T* A, int lda, int aCols, T* aNorm, int aRows, T* B, int ldb, int bCols, T* bNorm, int bRows, T* Out, int ldOut, kernel_opt kernelOptions, bool doKernel, int a_cpuBlocks, int b_cpuBlocks, int a_gpuBlocks, int b_gpuBlocks, int num_streams_input, int num_device_input, bool a_on_device, bool b_on_device, bool out_on_device, bool transpose){
    #ifdef _OPENMP
    int ompCount = 0;
    int ompLimit = params.context.getOmpLimit();
    #endif
    
    bool b_pinned = false;
    bool out_pinned = false;

    //Get the cublas handle
    cublasHandle_t &handle = params.context.getCuBlasHandle();

    //If we are transposed, do some tranposing stuff
    if(transpose){
      std::swap(aCols, aRows);
      std::swap(bCols, bRows);
    }

    //Check that our parameters are valid
    if(aRows != bRows || (transpose && doKernel)){
      return 1;
    }

    int rows = aRows;

    //Adjust leading dimensions if matricies are on the device already
    int lda_device = rows;
    int ldb_device = rows;

    if(a_on_device){
      lda_device = lda;
    }

    if(b_on_device){
      ldb_device = ldb;
    }

    //Number of CUDA streams to use
    int num_streams = num_streams_input;

    //Check that we don't have too many blocks
    if(a_gpuBlocks > aCols){
      a_gpuBlocks = 1;
    }

    if(b_gpuBlocks > bCols){
      b_gpuBlocks = 1;
    }

    //If one of our matricies is already on the device, then we don't actually allocate for it
    if(a_on_device){
      a_cpuBlocks = 1;
      a_gpuBlocks = 1;
    }

    //We still use multiple gpu "blocks" for b to take advantage of streaming out back to the host.
    if(b_on_device){
      b_cpuBlocks = 1;
    }

    //Calculate size of cpu blocks for each matrix
    int a_cpuBlockSize = (aCols + a_cpuBlocks - 1) / a_cpuBlocks;
    int b_cpuBlockSize = (bCols + b_cpuBlocks - 1) / b_cpuBlocks;

    //Calculate points in overflow blocks
    int a_cpuOverflow = aCols % a_cpuBlockSize;
    int b_cpuOverflow = bCols % b_cpuBlockSize;

    //Calculate size of gpu blocks for each matrix
    int a_gpuBlockSize = (a_cpuBlockSize + a_gpuBlocks - 1) / a_gpuBlocks;
    int b_gpuBlockSize = (b_cpuBlockSize + b_gpuBlocks - 1) / b_gpuBlocks;

    //Create ones matrix for ger operation
    int ones_size = max(a_gpuBlockSize, b_gpuBlockSize);
    T* ones_host = new T[ones_size];
    for(int i = 0; i < ones_size; ++i){
      ones_host[i] = 1;
    }

    //If we're transposed we have to try to pin all of B
    if(transpose){
      cudaError_t err_1 = cudaHostRegister(B, ldb * bRows * sizeof(T), 0);
      cudaError_t err_2 = cudaHostRegister(bNorm, bRows * sizeof(T), 0);

      if(err_1 != cudaSuccess || err_2 != cudaSuccess){
        cudaHostUnregister(B);
        cudaHostUnregister(bNorm);
      } else {
        b_pinned = true;
      }
    }

    //Number of devices to use (only supports >1 if we have C++11)
    int num_device = 1;

    //If C++11 is not supported then ignore the lambda
    #ifndef CPP11
    int device_id = 0;

    //Create lambda for inner loop
    #else
    num_device = num_device_input;
    atomic_flag device_failure = ATOMIC_FLAG_INIT; //Flag to test for failure

    //We also can only support one device if one of our input/output matricies is already on the device
    if(a_on_device || b_on_device || out_on_device){
      num_device = 1;
    }

    //Synchronization tools
    mutex mtx_AB, mtx_Out;
    int sync_ABCounter = 0, sync_OutCounter = 0;
    condition_variable cv_ABSync;
    condition_variable cv_OutSync;

    /*function<void()>*/ auto device_function = [&](int device_id) -> int {
      cudaSetDevice(device_id);

    #endif

        //Allocate gpu blocks
      vector<cudaError_t> gpu_err;
      T *a_device = 0, *aNorm_device = 0, *ones_device = 0;

      if(!a_on_device){
        gpu_err.push_back(cudaMalloc((void**)&a_device, rows * a_gpuBlockSize * sizeof(T)));
        gpu_err.push_back(cudaMalloc((void**)&aNorm_device, a_gpuBlockSize * sizeof(T)));
      }

      gpu_err.push_back(cudaMalloc((void**)&ones_device, ones_size * sizeof(T)));

      vector<T*> b_device(num_streams, NULL);
      vector<T*> out_device(num_streams, NULL);
      vector<T*> bNorm_device(num_streams, NULL);
      for(int str = 0; str < num_streams; ++str){
        if(!b_on_device){
          gpu_err.push_back(cudaMalloc((void**)&(b_device[str]), rows * b_gpuBlockSize * sizeof(T)));
          gpu_err.push_back(cudaMalloc((void**)&(bNorm_device[str]), b_gpuBlockSize * sizeof(T)));
        }

        gpu_err.push_back(cudaMalloc((void**)&(out_device[str]), a_gpuBlockSize * b_gpuBlockSize * sizeof(T)));
      }

        //If any of our allocations failed, start again with smaller blocks
      bool any_errors = false;
      for(int err = 0; err < gpu_err.size(); ++err){
        if(gpu_err[err] == cudaErrorMemoryAllocation){
          any_errors = true;
        }
      }

      if(any_errors){
        delete [] ones_host;

        if(a_device) CUDA_CHECK(cudaFree(a_device));
        if(aNorm_device) CUDA_CHECK(cudaFree(aNorm_device));
        if(ones_device) CUDA_CHECK(cudaFree(ones_device));

        for(int str = 0; str < num_streams; ++str){
          if(b_device[str]) CUDA_CHECK(cudaFree(b_device[str]));
          if(out_device[str]) CUDA_CHECK(cudaFree(out_device[str]));
          if(bNorm_device[str]) CUDA_CHECK(cudaFree(bNorm_device[str]));
        }

            //If we are in a function, set the failure flag and exit, otherwise just retry with smaller blocks
            #ifdef CPP11
          device_failure.test_and_set();
          return 0;
          #else
          //Unregister the full B matrix
        if(transpose && b_pinned){
          cudaHostUnregister(B);
          cudaHostUnregister(bNorm);
          b_pinned = false;
        }

        if(transpose){
          std::swap(aCols, aRows);
          std::swap(bCols, bRows);
        }

        return pinned_kernel_multiply(params, A, lda, aCols, aNorm, aRows, B, ldb, bCols, bNorm, bRows, Out, ldOut, kernelOptions, doKernel, a_cpuBlocks, b_cpuBlocks, a_gpuBlocks * 2, b_gpuBlocks * 2, num_streams, num_device_input, a_on_device, b_on_device, out_on_device, transpose);
            #endif
      }

        //Copy ones to the gpu
      CUDA_CHECK(cudaMemcpy(ones_device, ones_host, ones_size * sizeof(T), cudaMemcpyHostToDevice));
      delete [] ones_host;

        //Create our streams
      vector<cudaStream_t> stream(num_streams);
      for(int str = 0; str < num_streams; ++str){
        CUDA_CHECK(cudaStreamCreate(&(stream[str])));
      }

        //Loop through cpu blocks in A
      for(int a_cpuIter = 0; a_cpuIter < a_cpuBlocks; ++a_cpuIter){
        int a_cpuBlockSizeTemp = a_cpuBlockSize;
        int a_gpuBlocksTemp = a_gpuBlocks;
        int a_gpuBlockSizeTemp = a_gpuBlockSize;

          //Check if we are in the overflow block
        if(a_cpuIter == a_cpuBlocks - 1 && a_cpuOverflow != 0){
          a_cpuBlockSizeTemp = a_cpuOverflow;
          a_gpuBlocksTemp = a_gpuBlocks <= a_cpuBlockSizeTemp ? a_gpuBlocks : 1;
          a_gpuBlockSizeTemp = (a_cpuBlockSizeTemp + a_gpuBlocksTemp - 1) / a_gpuBlocksTemp;
        }

        a_gpuBlocksTemp = (a_cpuBlockSizeTemp + a_gpuBlockSizeTemp - 1) / a_gpuBlockSizeTemp;

        int a_gpuBlockSizeTemp_final = a_gpuBlockSizeTemp;

          //Calculate the memory offset for this block
        int aFull_offset = transpose ? a_cpuBlockSize * a_cpuIter : lda * a_cpuBlockSize * a_cpuIter;
        int aNorm_offset = a_cpuBlockSize * a_cpuIter;

          //Loop through cpu blocks in B
        for(int b_cpuIter = 0; b_cpuIter < b_cpuBlocks; ++b_cpuIter){
          int b_cpuBlockSizeTemp = b_cpuBlockSize;
          int b_gpuBlocksTemp = b_gpuBlocks;
          int b_gpuBlockSizeTemp_final = b_gpuBlockSize;

            //Check if we are in the overflow block
          if(b_cpuIter == b_cpuBlocks - 1 && b_cpuOverflow != 0){
            b_cpuBlockSizeTemp = b_cpuOverflow;
            b_gpuBlocksTemp = b_gpuBlocks <= b_cpuBlockSizeTemp ? b_gpuBlocks : 1;
            b_gpuBlockSizeTemp_final = (b_cpuBlockSizeTemp + b_gpuBlocksTemp - 1) / b_gpuBlocksTemp;
          }

          b_gpuBlocksTemp = (b_cpuBlockSizeTemp + b_gpuBlockSizeTemp_final - 1) / b_gpuBlockSizeTemp_final;

            //Calculate the memory offset for this block
          int bFull_offset = transpose ? b_cpuBlockSize * b_cpuIter : ldb * b_cpuBlockSize * b_cpuIter;
          int bNorm_offset = b_cpuBlockSize * b_cpuIter;

          if(device_id == 0 && !transpose){
            cudaError_t err_1 = cudaHostRegister(B + bFull_offset, ldb * b_cpuBlockSizeTemp * sizeof(T), 0);
            cudaError_t err_2 = cudaHostRegister(bNorm + bNorm_offset, b_cpuBlockSizeTemp * sizeof(T), 0);

            if(err_1 != cudaSuccess || err_2 != cudaSuccess){
              cudaHostUnregister(B + bFull_offset);
              cudaHostUnregister(bNorm + bNorm_offset);
              b_pinned = false;
            } else {
              b_pinned = true;
            }
          }

          int out_cpuOffset = b_cpuIter * ldOut * b_cpuBlockSize + a_cpuIter * a_cpuBlockSize;
          if(device_id == 0){
            cudaError_t err_1 = cudaHostRegister(Out + b_cpuIter * ldOut * b_cpuBlockSize, ldOut * b_cpuBlockSizeTemp * sizeof(T), 0);
            
            if(err_1 != cudaSuccess){
              cudaHostUnregister(Out + b_cpuIter * ldOut * b_cpuBlockSize);
              out_pinned = false;
            } else {
              out_pinned = true;
            }
          }

                //Synchronize threads here to make sure that we finished the copy to pinned memory
                #ifdef CPP11
          unique_lock<mutex> lock_AB(mtx_AB);
          ++sync_ABCounter;

		  cv_ABSync.wait(lock_AB, [sync_ABCounter, num_device]() -> bool { return sync_ABCounter == num_device; });

          sync_ABCounter = 0;
          lock_AB.unlock();
          cv_ABSync.notify_all();
                #endif

                //Reset our gpu block size
          a_gpuBlockSizeTemp = a_gpuBlockSizeTemp_final;

            //Calculate points in overflow blocks
          int a_gpuOverflow = a_cpuBlockSizeTemp % a_gpuBlockSizeTemp;
          int b_gpuOverflow = b_cpuBlockSizeTemp % b_gpuBlockSizeTemp_final;

            //Loop through gpu blocks in A's cpu block (each device is responsible for a different set of a blocks)
          for(int a_gpuIter = device_id; a_gpuIter < a_gpuBlocksTemp; a_gpuIter += num_device){

                    //Calculate the memory offset for this block
            int a_gpuOffset = (transpose) ? a_gpuIter * a_gpuBlockSizeTemp :  a_gpuIter * a_gpuBlockSizeTemp * lda;
            int aNorm_gpuOffset = a_gpuIter * a_gpuBlockSizeTemp;

              //Check if we are in the overflow block
            if(a_gpuIter == a_gpuBlocksTemp - 1 && a_gpuOverflow != 0){
              a_gpuBlockSizeTemp = a_gpuOverflow;
            }

              //Copy the block to the gpu (if needed)
            if(!a_on_device){
              //CUDA_CHECK(cudaMemcpy(a_device, a_hostPinned + a_gpuOffset, a_gpuBlockSizeTemp * rows * sizeof(T), cudaMemcpyHostToDevice));
              if(!transpose){
                CUBLAS_CHECK(cublasSetMatrix(aRows, a_gpuBlockSizeTemp, sizeof(T), A + aFull_offset + a_gpuOffset, lda, a_device, lda_device));
              } else {
                lda_device = a_gpuBlockSizeTemp;
                CUBLAS_CHECK(cublasSetMatrix(a_gpuBlockSizeTemp, aRows, sizeof(T), A + aFull_offset + a_gpuOffset, lda, a_device, lda_device));
              }

              if(doKernel) CUDA_CHECK(cudaMemcpy(aNorm_device, aNorm + aNorm_gpuOffset + aNorm_offset, a_gpuBlockSizeTemp * sizeof(T), cudaMemcpyHostToDevice));
            } else {
              a_device = A;
              if (doKernel) aNorm_device = aNorm;
            }
              //Loop through gpu blocks in B's cpu block
            for(int b_gpuIter = 0; b_gpuIter < b_gpuBlocksTemp; b_gpuIter += num_streams){
              vector<int> b_gpuBlockSizeTemp(num_streams, b_gpuBlockSizeTemp_final);

                        //Calculate the memory offset for this block
              vector<int> b_gpuOffset(num_streams);
              vector<int> bNorm_gpuOffset(num_streams);

              for(int str = 0; str < num_streams; ++str){
                int b_gpuIter_str = b_gpuIter + str;

                b_gpuOffset[str] = (transpose/* && b_on_device*/) ? b_gpuIter_str * b_gpuBlockSizeTemp_final : b_gpuIter_str * b_gpuBlockSizeTemp_final * ldb;
                bNorm_gpuOffset[str] = b_gpuIter_str * b_gpuBlockSizeTemp_final;

                            //Check if the block is an overflow or out of bounds
                if(b_gpuIter_str == b_gpuBlocksTemp - 1 && b_gpuOverflow != 0){
                  b_gpuBlockSizeTemp[str] = b_gpuOverflow;
                } else if(b_gpuIter_str >= b_gpuBlocksTemp) {
                  b_gpuBlockSizeTemp[str] = 0;
                }

                            //Asychronously copy B's block to the device (if needed)
                if(b_gpuBlockSizeTemp[str] != 0 && !b_on_device){
                  if(!transpose){
                    if(!b_pinned){
                      cudaStreamSynchronize(stream[str]);
                      CUBLAS_CHECK(cublasSetMatrix(bRows, b_gpuBlockSizeTemp[str], sizeof(T), B + b_gpuOffset[str] + bFull_offset, ldb, b_device[str], ldb_device));
                    } else {
                      CUBLAS_CHECK(cublasSetMatrixAsync(bRows, b_gpuBlockSizeTemp[str], sizeof(T), B + b_gpuOffset[str] + bFull_offset, ldb, b_device[str], ldb_device, stream[str]));
                    }
                  } else {
                    ldb_device = b_gpuBlockSize;
                    if(!b_pinned){
                      cudaStreamSynchronize(stream[str]);
                      CUBLAS_CHECK(cublasSetMatrix(b_gpuBlockSizeTemp[str], bRows, sizeof(T), B + b_gpuOffset[str] + bFull_offset, ldb, b_device[str], ldb_device));
                    } else {
                      CUBLAS_CHECK(cublasSetMatrixAsync(b_gpuBlockSizeTemp[str], bRows, sizeof(T), B + b_gpuOffset[str] + bFull_offset, ldb, b_device[str], ldb_device, stream[str]));
                    }
                  }

                  if(doKernel) CUDA_CHECK(cudaMemcpyAsync(bNorm_device[str], bNorm + bNorm_gpuOffset[str] + bNorm_offset, b_gpuBlockSizeTemp[str] * sizeof(T), cudaMemcpyHostToDevice, stream[str]));

                } else if (b_gpuBlockSizeTemp[str] != 0){
                  b_device[str] = B + b_gpuOffset[str];
                  if(doKernel) bNorm_device[str] = bNorm + bNorm_gpuOffset[str];
                }
              }
                //For multiplication (gemm) call
              T alpha = 1;
              T beta = 0;

                //For polynomial kernel, set the coefficient
              if (doKernel && kernelOptions.kernel == POLYNOMIAL || kernelOptions.kernel == SIGMOID){
                alpha = kernelOptions.gamma;
                beta = 1;

                for(int str = 0; str < num_streams; ++str){
                  if(b_gpuBlockSizeTemp[str] != 0){
                    device_ewiseOp_stream(params, out_device[str], out_device[str], a_gpuBlockSizeTemp * b_gpuBlockSizeTemp[str], kernelOptions.coef, 0, 1, a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, stream[str]);
                  }
                }
              }

                //For RBF kernel, do distance calculation and exp
              if(doKernel && kernelOptions.kernel != LINEAR && kernelOptions.kernel != POLYNOMIAL && kernelOptions.kernel != SIGMOID){
                for(int str = 0; str < num_streams; ++str){
                  if(b_gpuBlockSizeTemp[str] != 0){
                    cublasSetStream(handle, stream[str]);
                    device_gemm(params, true, false, a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], rows, -2, a_device, lda_device, b_device[str], ldb_device, 0, out_device[str], a_gpuBlockSizeTemp);
                    device_ger(params, a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], 1, ones_device, 1, bNorm_device[str], 1, out_device[str], a_gpuBlockSizeTemp);
                    device_ger(params, a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], 1, aNorm_device, 1, ones_device, 1, out_device[str], a_gpuBlockSizeTemp); 

                    if (kernelOptions.kernel == RBF){
                      device_exp_stream(params, out_device[str], out_device[str], b_gpuBlockSizeTemp[str], a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, kernelOptions.gamma, stream[str]);
                    }
                  }
                }

                } else { //Otherwise just do out = A'B or whatever
                for(int str = 0; str < num_streams; ++str){
                  if(b_gpuBlockSizeTemp[str] != 0){
                    cublasSetStream(handle, stream[str]);

                                  //Adjust transpose parameters if matricies could not be previously adjusted
                    bool transA = !transpose;
                    bool transB = transpose;
                    device_gemm(params, transA, transB, a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], rows, alpha, a_device, lda_device, b_device[str], ldb_device, beta, out_device[str], a_gpuBlockSizeTemp);
                  }
                }
              }

                //Raise to degree for polynomial kernel
              if (doKernel && kernelOptions.kernel == POLYNOMIAL){
                for(int str = 0; str < num_streams; ++str){
                  if(b_gpuBlockSizeTemp[str] != 0){
                    device_ewiseOp_stream(params, out_device[str], out_device[str], a_gpuBlockSizeTemp * b_gpuBlockSizeTemp[str], 0, 1, kernelOptions.degree, a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, stream[str]);
                  }
                }             
              }

                        //Do hyperbolic tanget for sigmoid kernel
              if (doKernel && kernelOptions.kernel == SIGMOID){
                for(int str = 0; str < num_streams; ++str){
                  if(b_gpuBlockSizeTemp[str] != 0){
                    device_tanh_stream(params, out_device[str], out_device[str], b_gpuBlockSizeTemp[str], a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, stream[str]);
                  }           
                }                    
              }
              //Transfer each block back to the host
              for(int str = 0; str < num_streams; ++str){
                            //Calculate the output memory offsets
                int b_gpuIter_str = b_gpuIter + str;
                int out_offset_str = a_gpuIter * a_gpuBlockSizeTemp_final + b_gpuIter_str * b_gpuBlockSizeTemp_final * ldOut;

                if(b_gpuBlockSizeTemp[str] != 0 && !out_on_device){
                  //CUBLAS_CHECK(cublasGetMatrixAsync(a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], sizeof(T), out_device[str], a_gpuBlockSizeTemp, out_hostPinned + out_offset_str, a_cpuBlockSizeTemp, stream[str]));
                  if(!out_pinned){
                    cudaStreamSynchronize(stream[str]);
                    CUBLAS_CHECK(cublasGetMatrix(a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], sizeof(T), out_device[str], a_gpuBlockSizeTemp, Out + out_cpuOffset + out_offset_str, ldOut));
                  } else {
                    CUBLAS_CHECK(cublasGetMatrixAsync(a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], sizeof(T), out_device[str], a_gpuBlockSizeTemp, Out + out_cpuOffset + out_offset_str, ldOut, stream[str]));
                  }
                } else if(b_gpuBlockSizeTemp[str] != 0) {
                  int out_cpuOffset = b_cpuIter * ldOut * b_cpuBlockSize + a_cpuIter * a_cpuBlockSize;
                  out_offset_str = a_gpuIter * a_gpuBlockSizeTemp_final + b_gpuIter_str * b_gpuBlockSizeTemp_final * ldOut;
                  //device_geam(params, false, false, a_gpuBlockSizeTemp, b_gpuBlockSizeTemp[str], alpha, out_device[str], a_gpuBlockSizeTemp, beta, out_device[str], a_gpuBlockSizeTemp, Out + out_cpuOffset + out_offset_str, ldOut);
                  device_ewiseOp_stream(params, out_device[str], Out + out_cpuOffset + out_offset_str, a_gpuBlockSizeTemp * b_gpuBlockSizeTemp[str], 0, 1,  1, a_gpuBlockSizeTemp, a_gpuBlockSizeTemp, ldOut, stream[str]);                                
                }
              }
            }
                    //Synchronize the device before we move to a new A block
            for(int str = 0; str < num_streams; ++str){
              cudaStreamSynchronize(stream[str]);
            }
          }

          //Synchronize threads again here to make sure pinned out is fully populated before copying it back to the main array
          #ifdef CPP11
          unique_lock<mutex> lock_Out(mtx_Out);
          ++sync_OutCounter;

		  cv_OutSync.wait(lock_Out, [sync_OutCounter, num_device]() -> bool { return sync_OutCounter == num_device; });

          sync_OutCounter = 0;
          lock_Out.unlock();
          cv_OutSync.notify_all();
                #endif

          //Unregister partial B matrix
          if(device_id == 0 && !transpose && b_pinned){
            cudaHostUnregister(B + bFull_offset);
            cudaHostUnregister(bNorm + bNorm_offset);
            b_pinned = false;
          }

          //Unregister partial Out matrix
          if(device_id == 0 && out_pinned){
            cudaHostUnregister(Out + b_cpuIter * ldOut * b_cpuBlockSize);
            out_pinned = false;
          }

        }
      }

        //Free device memory
      if(!a_on_device){
        CUDA_CHECK(cudaFree(a_device));
        CUDA_CHECK(cudaFree(aNorm_device));
      }

      CUDA_CHECK(cudaFree(ones_device));

      for(int str = 0; str < num_streams; ++str){
        CUDA_CHECK(cudaFree(out_device[str]));

        if(!b_on_device){
          CUDA_CHECK(cudaFree(b_device[str]));
          CUDA_CHECK(cudaFree(bNorm_device[str]));
        }

        CUDA_CHECK(cudaStreamDestroy(stream[str]));
      }

      cudaDeviceSynchronize();
      cublasSetStream(handle, NULL);
    //End of the lambda 
    #ifdef CPP11
	};

    //Start a thread for each device other than 0
    vector<thread> threads;
    for(int t = 1; t < num_device; ++t){
      threads.push_back(thread(device_function, t));
    }

    //Run device 0 in the main thread
    device_function(0);

    //Wait for everything to finish
    for(int t = 0; t < threads.size(); ++t){
      threads[t].join();
    }
    #endif

    //Unregister the full B matrix
    if(transpose && b_pinned){
      cudaHostUnregister(B);
      cudaHostUnregister(bNorm);
      b_pinned = false;
    }

    //Check that there were no device memory errors, if so start again with smaller blocks
    #ifdef CPP11
    if(device_failure.test_and_set()){
     if(transpose){
      std::swap(aCols, aRows);
      std::swap(bCols, bRows);
    }

    return pinned_kernel_multiply(params, A, lda, aCols, aNorm, aRows, B, ldb, bCols,  bNorm, bRows, Out, ldOut, kernelOptions, doKernel, a_cpuBlocks, b_cpuBlocks, a_gpuBlocks * 2, b_gpuBlocks * 2, num_streams, num_device_input, a_on_device, b_on_device, out_on_device, transpose);
  }
    #endif
  return 0;
}  

template int pinned_kernel_multiply<float>(DeviceParams params, float* A, int lda, int aCols, float* aNorm, int aRows, float* B, int ldb, int bCols,  float* bNorm, int bRows, float* Out, int ldOut, kernel_opt kernelOptions, bool doKernel, int a_cpuBlocks, int b_cpuBlocks, int a_gpuBlocks, int b_gpuBLocks, int num_streams_input, int num_device_input, bool a_on_device, bool b_on_device, bool out_on_device, bool transpose);
template int pinned_kernel_multiply<double>(DeviceParams params, double* A, int lda, int aCols, double* aNorm, int aRows, double* B, int ldb, int bCols,  double* bNorm, int bRows, double* Out, int ldOut, kernel_opt kernelOptions, bool doKernel, int a_cpuBlocks, int b_cpuBlocks, int a_gpuBlocks, int b_gpuBLocks, int num_streams_input, int num_device_input, bool a_on_device, bool b_on_device, bool out_on_device, bool transpose);
}
#endif
