#ifndef REDUCE_H
#define REDUCE_H
#include "reductionOperators.h"

template<int stepSize>
__device__ void sumStep(float* temps) {
  if (threadIdx.x < stepSize) {
    temps[threadIdx.x] += temps[threadIdx.x + stepSize];
  }
  __syncthreads();
}

__device__ void sumReduce(float* temps) {
  if (256 < BLOCKSIZE) sumStep<256>(temps);
  if (128 < BLOCKSIZE) sumStep<128>(temps);
  if ( 64 < BLOCKSIZE) sumStep< 64>(temps);
  if ( 32 < BLOCKSIZE) sumStep< 32>(temps);
  if ( 16 < BLOCKSIZE) sumStep< 16>(temps);
  if (  8 < BLOCKSIZE) sumStep<  8>(temps);
  if (  4 < BLOCKSIZE) sumStep<  4>(temps);
  if (  2 < BLOCKSIZE) sumStep<  2>(temps);
  if (  1 < BLOCKSIZE) sumStep<  1>(temps);
}

template<int stepSize>
__device__ void maxStep(float* temps) {
  if (threadIdx.x < stepSize) {
    maxOperator(temps[threadIdx.x], temps[threadIdx.x + stepSize], temps + threadIdx.x);
  }
  __syncthreads();

}

__device__ void maxReduce(float* temps) {
  if (256 < BLOCKSIZE) maxStep<256>(temps);
  if (128 < BLOCKSIZE) maxStep<128>(temps);
  if ( 64 < BLOCKSIZE) maxStep< 64>(temps);
  if ( 32 < BLOCKSIZE) maxStep< 32>(temps);
  if ( 16 < BLOCKSIZE) maxStep< 16>(temps);
  if (  8 < BLOCKSIZE) maxStep<  8>(temps);
  if (  4 < BLOCKSIZE) maxStep<  4>(temps);
  if (  2 < BLOCKSIZE) maxStep<  2>(temps);
  if (  1 < BLOCKSIZE) maxStep<  1>(temps);
}

template<int stepSize>
__device__ void argminStep(float* values, int* indices) {
  if (threadIdx.x < stepSize) {
    int compOffset = threadIdx.x + stepSize;
    argMin(indices[threadIdx.x], values[threadIdx.x], indices[compOffset], values[compOffset], indices + threadIdx.x, values + threadIdx.x);
  }
  __syncthreads();
}

__device__ void argminReduce(float* values, int* indices) {
  if (256 < BLOCKSIZE) argminStep<256>(values, indices);
  if (128 < BLOCKSIZE) argminStep<128>(values, indices);
  if ( 64 < BLOCKSIZE) argminStep< 64>(values, indices);
  if ( 32 < BLOCKSIZE) argminStep< 32>(values, indices);
  if ( 16 < BLOCKSIZE) argminStep< 16>(values, indices);
  if (  8 < BLOCKSIZE) argminStep<  8>(values, indices);
  if (  4 < BLOCKSIZE) argminStep<  4>(values, indices);
  if (  2 < BLOCKSIZE) argminStep<  2>(values, indices);
  if (  1 < BLOCKSIZE) argminStep<  1>(values, indices);
}

template<int stepSize>
__device__ void argmaxStep(float* values, int* indices) {
  if (threadIdx.x < stepSize) {
    int compOffset = threadIdx.x + stepSize;
    argMax(indices[threadIdx.x], values[threadIdx.x], indices[compOffset], values[compOffset], indices + threadIdx.x, values + threadIdx.x);
  }
  __syncthreads();
}

__device__ void argmaxReduce(float* values, int* indices) {
  if (256 < BLOCKSIZE) argmaxStep<256>(values, indices);
  if (128 < BLOCKSIZE) argmaxStep<128>(values, indices);
  if ( 64 < BLOCKSIZE) argmaxStep< 64>(values, indices);
  if ( 32 < BLOCKSIZE) argmaxStep< 32>(values, indices);
  if ( 16 < BLOCKSIZE) argmaxStep< 16>(values, indices);
  if (  8 < BLOCKSIZE) argmaxStep<  8>(values, indices);
  if (  4 < BLOCKSIZE) argmaxStep<  4>(values, indices);
  if (  2 < BLOCKSIZE) argmaxStep<  2>(values, indices);
  if (  1 < BLOCKSIZE) argmaxStep<  1>(values, indices);
}

#endif
