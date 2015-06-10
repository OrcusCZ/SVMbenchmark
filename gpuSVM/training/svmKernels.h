#ifndef SVM_KERNELS
#define SVM_KERNELS

#include <math.h>
#include "../common/framework.h"
#include "reduce.h"
#include "kernelType.h"



struct Linear {

  
  static __device__ __host__ float selfKernel(float* pointerA, int pitchA, float* pointerAEnd, float parameterA, float parameterB, float parameterC) {
    float accumulant = 0.0f;
    do {
      float value = *pointerA;
      accumulant += value * value;
      pointerA += pitchA;
    } while (pointerA < pointerAEnd);
    return accumulant;
  }

  static __device__ __host__ float kernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float parameterA, float parameterB, float parameterC) {
    float accumulant = 0.0f;
    do {
      accumulant += (*pointerA) * (*pointerB);
      pointerA += pitchA;
      pointerB += pitchB;
    } while (pointerA < pointerAEnd);
    return accumulant;
  }

  static __device__ void parallelKernel(float* pointerA, float* pointerAEnd, float* pointerB, float* sharedTemps, float parameterA, float parameterB, float parameterC) {
    pointerA += threadIdx.x;
    pointerB += threadIdx.x;
    sharedTemps[threadIdx.x] = 0.0f;
    while (pointerA < pointerAEnd) {
      sharedTemps[threadIdx.x] += (*pointerA) * (*pointerB);
      pointerA += blockDim.x;
      pointerB += blockDim.x;
    }
    __syncthreads();

    sumReduce(sharedTemps);
  }

  //This function assumes we're doing two kernel evaluations at once:
  //Phi1(a, b) and Phi2(a, c)
  //b and c are already in shared memory, so we don't care about minimizing
  //their memory accesses, but a is in global memory
  //So we have to worry about not accessing a twice
  static __device__ __host__ void dualKernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float* pointerC, int pitchC, float parameterA, float parameterB, float parameterC, float& phi1, float& phi2) {
    float accumulant1 = 0.0f;
    float accumulant2 = 0.0f;
    do {
      float xa = *pointerA;
      accumulant1 += xa * (*pointerB);
      accumulant2 += xa * (*pointerC);
      pointerA += pitchA;
      pointerB += pitchB;
      pointerC += pitchC;
    } while (pointerA < pointerAEnd);
    phi1 = accumulant1;
    phi2 = accumulant2;
  }
};

struct Polynomial {
  static __device__ __host__ float selfKernel(float* pointerA, int pitchA, float* pointerAEnd, float a, float r, float d) {
    float accumulant = 0.0f;
    do {
      float value = *pointerA;
      accumulant += value * value;
      pointerA += pitchA;
    } while (pointerA < pointerAEnd);
    accumulant = accumulant * a + r;
    float result = accumulant;
    for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
      result *= accumulant;
    }
    
    return result;
  }

  static __device__ __host__ float kernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float a, float r, float d) {
    float accumulant = 0.0f;
    do {
      accumulant += (*pointerA) * (*pointerB);
      pointerA += pitchA;
      pointerB += pitchB;
    } while (pointerA < pointerAEnd);
    accumulant = accumulant * a + r;
    float result = accumulant;
    for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
      result *= accumulant;
    }
    
    return result;
  }

  static __device__ void parallelKernel(float* pointerA, float* pointerAEnd, float* pointerB, float* sharedTemps, float a, float r, float d) {
    pointerA += threadIdx.x;
    pointerB += threadIdx.x;
    sharedTemps[threadIdx.x] = 0.0f;
    while (pointerA < pointerAEnd) {
      sharedTemps[threadIdx.x] += (*pointerA) * (*pointerB);
      pointerA += blockDim.x;
      pointerB += blockDim.x;
    }
    __syncthreads();

    sumReduce(sharedTemps);
    if (threadIdx.x == 0) {
      float accumulant = sharedTemps[0] * a + r;
     
      float result = accumulant;
      for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
        result *= accumulant;
      }
      sharedTemps[0] = result;
    }
    
  }
  
  //This function assumes we're doing two kernel evaluations at once:
  //Phi1(a, b) and Phi2(a, c)
  //b and c are already in shared memory, so we don't care about minimizing
  //their memory accesses, but a is in global memory
  //So we have to worry about not accessing a twice
  static __device__ __host__ void dualKernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float* pointerC, int pitchC, float a, float r, float d, float& phi1, float& phi2) {
    float accumulant1 = 0.0f;
    float accumulant2 = 0.0f;
    do {
      float xa = *pointerA;
      accumulant1 += xa * (*pointerB);
      accumulant2 += xa * (*pointerC);
      pointerA += pitchA;
      pointerB += pitchB;
      pointerC += pitchC;
    } while (pointerA < pointerAEnd);
    
    accumulant1 = accumulant1 * a + r;
    float result = accumulant1;
    for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
      result *= accumulant1;
    }
    phi1 = result;

    accumulant2 = accumulant2 * a + r;
    result = accumulant2;
    for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
      result *= accumulant2;
    }
    phi2 = result;
  }
};

struct Gaussian {
  static __device__ __host__ float selfKernel(float* pointerA, int pitchA, float* pointerAEnd, float ngamma, float parameterB, float parameterC) {
    return 1.0f;
  }

  static __device__ __host__ float kernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float ngamma, float parameterB, float parameterC) {
    float accumulant = 0.0f;
    do {
      float diff = *pointerA - *pointerB;
      accumulant += diff * diff;
      pointerA += pitchA;
      pointerB += pitchB;
    } while (pointerA < pointerAEnd);
    return exp(ngamma * accumulant);
  }

  static __device__ void parallelKernel(float* pointerA, float* pointerAEnd, float* pointerB, float* sharedTemps, float ngamma, float parameterB, float parameterC) {
    pointerA += threadIdx.x;
    pointerB += threadIdx.x;
    sharedTemps[threadIdx.x] = 0.0f;
    while (pointerA < pointerAEnd) {
      float diff = (*pointerA) - (*pointerB);
      sharedTemps[threadIdx.x] += diff * diff;
      pointerA += blockDim.x;
      pointerB += blockDim.x;
    }
    __syncthreads();

    sumReduce(sharedTemps);
    if (threadIdx.x == 0) {
      sharedTemps[0] = exp(sharedTemps[0] * ngamma);
    }
  }
  
  //This function assumes we're doing two kernel evaluations at once:
  //Phi1(a, b) and Phi2(a, c)
  //b and c are already in shared memory, so we don't care about minimizing
  //their memory accesses, but a is in global memory
  //So we have to worry about not accessing a twice
  static __device__ __host__ void dualKernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float* pointerC, int pitchC, float ngamma, float parameterB, float parameterC, float& phi1, float& phi2) {
    float accumulant1 = 0.0f;
    float accumulant2 = 0.0f;
    do {
      float xa = *pointerA;
      float diff = xa - (*pointerB);
      accumulant1 += diff * diff;
      diff = xa - (*pointerC);
      accumulant2 += diff * diff;
      pointerA += pitchA;
      pointerB += pitchB;
      pointerC += pitchC;
    } while (pointerA < pointerAEnd);
    phi1 = exp(ngamma * accumulant1);
    phi2 = exp(ngamma * accumulant2);
  }
  
};

struct Sigmoid {

  
  
  static __device__ __host__ float selfKernel(float* pointerA, int pitchA, float* pointerAEnd, float a, float r, float parameterC) {
    float accumulant = 0.0f;
    do {
      float value = *pointerA;
      accumulant += value * value;
      pointerA += pitchA;
    } while (pointerA < pointerAEnd);
    accumulant = accumulant * a + r;
    return tanh(accumulant);
  }

  static __device__ __host__ float kernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float a, float r, float parameterC) {
    float accumulant = 0.0f;
    do {
      accumulant += (*pointerA) * (*pointerB);
      pointerA += pitchA;
      pointerB += pitchB;
    } while (pointerA < pointerAEnd);
    accumulant = accumulant * a + r;
    return tanh(accumulant);

  }

  static __device__ void parallelKernel(float* pointerA, float* pointerAEnd, float* pointerB, float* sharedTemps, float a, float r, float parameterC) {
    pointerA += threadIdx.x;
    pointerB += threadIdx.x;
    sharedTemps[threadIdx.x] = 0.0f;
    while (pointerA < pointerAEnd) {
      sharedTemps[threadIdx.x] += (*pointerA) * (*pointerB);
      pointerA += blockDim.x;
      pointerB += blockDim.x;
    }
    __syncthreads();

    sumReduce(sharedTemps);
    if (threadIdx.x == 0) {
      float accumulant = sharedTemps[0];
      sharedTemps[0] = tanh(accumulant);
    }
    
  }

  //This function assumes we're doing two kernel evaluations at once:
  //Phi1(a, b) and Phi2(a, c)
  //b and c are already in shared memory, so we don't care about minimizing
  //their memory accesses, but a is in global memory
  //So we have to worry about not accessing a twice
  static __device__ __host__ void dualKernel(float* pointerA, int pitchA, float* pointerAEnd, float* pointerB, int pitchB, float* pointerC, int pitchC, float a, float r, float parameterC, float& phi1, float& phi2) {
    float accumulant1 = 0.0f;
    float accumulant2 = 0.0f;
    do {
      float xa = *pointerA;
      accumulant1 += xa * (*pointerB);
      accumulant2 += xa * (*pointerC);
      pointerA += pitchA;
      pointerB += pitchB;
      pointerC += pitchC;
    } while (pointerA < pointerAEnd);
    accumulant1 = accumulant1 * a + r;
    phi1= tanh(accumulant1);
    accumulant2 = accumulant2 * a + r;
    phi2= tanh(accumulant2);
    
  }
};
    

#endif
