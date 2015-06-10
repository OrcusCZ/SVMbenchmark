#ifndef MEMORY_ROUTINES
#define MEMORY_ROUTINES

__device__ void coopExtractRowVector(float* data, int dataPitch, int index, int dimension, float* destination) {
  float* xiRowPtr = data + (index * dataPitch) + threadIdx.x;
  for(int currentDim = threadIdx.x; currentDim < dimension; currentDim += blockDim.x) {
    destination[currentDim] = *xiRowPtr;
    xiRowPtr += blockDim.x;
  }
}

#endif
