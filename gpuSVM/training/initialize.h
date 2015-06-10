#ifndef INITIALIZE
#define INITIALIZE

#include "svmKernels.h"

template<class Kernel>
__global__ void initializeArrays(float* devData, int devDataPitchInFloats, int nPoints, int nDimension, float parameterA, float parameterB, float parameterC, float* devKernelDiag, float* devAlpha, float* devF, float* devLabels) { 
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int index = bx * blockDim.x + tx;
	
	if (index < nPoints) {
    devKernelDiag[index] = Kernel::selfKernel(devData + index, devDataPitchInFloats, devData + (nDimension * devDataPitchInFloats), parameterA, parameterB, parameterC);
		devF[index] = -devLabels[index];
		devAlpha[index] = 0;
	}
}

void launchInitialization(float* devData, int devDataPitchInFloats, int nPoints, int nDimension, int kType, float parameterA, float parameterB, float parameterC, float* devKernelDiag, float* devAlpha, float* devF, float* devLabels, dim3 blockConfig, dim3 threadConfig) {
  switch (kType) {
  case LINEAR:
    initializeArrays<Linear><<<blockConfig, threadConfig>>>(devData, devDataPitchInFloats, nPoints, nDimension, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels);
    break;
  case POLYNOMIAL:
    initializeArrays<Polynomial><<<blockConfig, threadConfig>>>(devData, devDataPitchInFloats, nPoints, nDimension, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels);
    break;
  case GAUSSIAN:
    initializeArrays<Gaussian><<<blockConfig, threadConfig>>>(devData, devDataPitchInFloats, nPoints, nDimension, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels);
    break;  
  case SIGMOID:
    initializeArrays<Sigmoid><<<blockConfig, threadConfig>>>(devData, devDataPitchInFloats, nPoints, nDimension, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels);
    break;
  }
}

template<class Kernel>
__global__ void takeFirstStep(void* devResult, float* devKernelDiag, float* devData, int devDataPitchInFloats, float* devAlpha, float cost, int nDimension, int iLow, int iHigh, float parameterA, float parameterB, float parameterC) { 
                                     
	float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
  float* pointerA = devData + iHigh;
  float* pointerB = devData + iLow;
  float* pointerAEnd = devData + IMUL(nDimension, devDataPitchInFloats);
  float phiAB = Kernel::kernel(pointerA, devDataPitchInFloats, pointerAEnd, pointerB, devDataPitchInFloats, parameterA, parameterB, parameterC);
	
	eta = eta - 2*phiAB;
	//For the first step, we know alpha1Old == alpha2Old == 0, and we know sign == -1
	//labels[iLow] = -1
	//labels[iHigh] = 1
	//float sign = -1;
 
	//And we know eta > 0
	float alpha2New = 2/eta; //Just boil down the algebra
	if (alpha2New > cost) {
		alpha2New = cost;
	}
	//alpha1New == alpha2New for the first step

	devAlpha[iLow] = alpha2New;
	devAlpha[iHigh] = alpha2New;
	
	*((float*)devResult + 0) = 0.0;
	*((float*)devResult + 1) = 0.0;
	*((float*)devResult + 2) = 1.0;
	*((float*)devResult + 3) = -1.0;
	*((float*)devResult + 6) = alpha2New;
	*((float*)devResult + 7) = alpha2New;
}

void launchTakeFirstStep(void* devResult, float* devKernelDiag, float* devData, int devDataPitchInFloats, float* devAlpha, float cost, int nDimension, int iLow, int iHigh, int kType, float parameterA, float parameterB, float parameterC, dim3 blockConfig, dim3 threadConfig) {
  switch (kType) {
  case LINEAR:
    takeFirstStep<Linear><<<blockConfig, threadConfig>>>(devResult, devKernelDiag, devData, devDataPitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, parameterA, parameterB, parameterC);
    break;
  case POLYNOMIAL:
    takeFirstStep<Polynomial><<<blockConfig, threadConfig>>>(devResult, devKernelDiag, devData, devDataPitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, parameterA, parameterB, parameterC);
    break;
  case GAUSSIAN:
    takeFirstStep<Gaussian><<<blockConfig, threadConfig>>>(devResult, devKernelDiag, devData, devDataPitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, parameterA, parameterB, parameterC);
    break;  
  case SIGMOID:
    takeFirstStep<Sigmoid><<<blockConfig, threadConfig>>>(devResult, devKernelDiag, devData, devDataPitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, parameterA, parameterB, parameterC);
    break;
  }
}


#endif
