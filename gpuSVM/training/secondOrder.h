#ifndef SECONDORDERH
#define SECONDORDERH
#include "../common/framework.h"
#include "reduce.h"
#include "memoryRoutines.h"

int secondOrderPhaseOneSize(bool iLowCompute, bool iHighCompute, int nDimension) {
	int size = 0;
	if (iHighCompute) { size+= sizeof(float) * nDimension; }
	if (iLowCompute) { size+= sizeof(float) * nDimension; }
	return size;
}

template<bool iLowCompute, bool iHighCompute, class Kernel>
  __global__ void	secondOrderPhaseOne(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, int nPoints, int nDimension, float epsilon, float cEpsilon, float* devAlpha, float* devF, float alpha1Diff, float alpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, int* devLocalIndicesRH, float* devLocalFsRH) {
  
	extern __shared__ float xIHigh[];
  float* xILow;
  __shared__ int tempLocalIndices[BLOCKSIZE];
  __shared__ float tempLocalFs[BLOCKSIZE];

	if (iHighCompute) {
		xILow = &xIHigh[nDimension];
	} else {
    xILow = xIHigh;
  }


  if (iHighCompute) {
		coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iHigh, nDimension, xIHigh);
	}

  if (iLowCompute) {
    coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iLow, nDimension, xILow);
	}
	
	__syncthreads();



	int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

  float alpha;
  float f;
  float label;
  int reduceFlag;

	if (globalIndex < nPoints) {
		alpha = devAlpha[globalIndex];
		f = devF[globalIndex];
    label = devLabels[globalIndex];
  }
  
  if ((globalIndex < nPoints) &&
      (((label > 0) && (alpha < cEpsilon)) ||
       ((label < 0) && (alpha > epsilon)))) {
    reduceFlag = REDUCE0;
  } else {
    reduceFlag = NOREDUCE;
  }

	if (globalIndex < nPoints) {
		float highKernel;
		float lowKernel;
		if (iHighCompute) {
			highKernel = 0;
		} else {
			highKernel = devCache[((size_t)devCachePitchInFloats * iHighCacheIndex) + globalIndex];
		}
		if (iLowCompute) {
			lowKernel = 0;
		} else {
			lowKernel = devCache[((size_t)devCachePitchInFloats * iLowCacheIndex) + globalIndex];
		}

    if (iHighCompute && iLowCompute) {
      Kernel::dualKernel(devData + globalIndex, devDataPitchInFloats, devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1, xILow, 1, parameterA, parameterB, parameterC, highKernel, lowKernel);
    } else if (iHighCompute) {
      highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats, devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1, parameterA, parameterB, parameterC);
    } else if (iLowCompute) {
      lowKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats, devData + globalIndex + (devDataPitchInFloats * nDimension), xILow, 1, parameterA, parameterB, parameterC);
    }
			
		f = f + alpha1Diff * highKernel;
		f = f + alpha2Diff * lowKernel;
		
		if (iLowCompute) {
			devCache[((size_t)devCachePitchInFloats * iLowCacheIndex) + globalIndex] = lowKernel;
		}
		if (iHighCompute) {
			devCache[((size_t)devCachePitchInFloats * iHighCacheIndex) + globalIndex] = highKernel;
		}
    devF[globalIndex] = f;
	}
	__syncthreads();

  if ((reduceFlag & REDUCE0) == 0) {
    tempLocalFs[threadIdx.x] = FLT_MAX; //Ignore me
  } else {
    tempLocalFs[threadIdx.x] = f;
    tempLocalIndices[threadIdx.x] = globalIndex;
  }
  __syncthreads(); 
  argminReduce(tempLocalFs, tempLocalIndices);
  if (threadIdx.x == 0) {
		devLocalIndicesRH[blockIdx.x] = tempLocalIndices[0];
		devLocalFsRH[blockIdx.x] = tempLocalFs[0];
  }
}

int secondOrderPhaseTwoSize() {
	int size = 0;
	return size;
}

__global__ void secondOrderPhaseTwo(void* devResult, int* devLocalIndicesRH, float* devLocalFsRH, int inputSize) {
  __shared__ int tempIndices[BLOCKSIZE];
  __shared__ float tempFs[BLOCKSIZE];

  //Load elements
  if (threadIdx.x < inputSize) {
    tempIndices[threadIdx.x] = devLocalIndicesRH[threadIdx.x];
    tempFs[threadIdx.x] = devLocalFsRH[threadIdx.x];
  } else {
    tempFs[threadIdx.x] = FLT_MAX;
  }

  if (inputSize > BLOCKSIZE) {
    for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
      argMin(tempIndices[threadIdx.x], tempFs[threadIdx.x], devLocalIndicesRH[i], devLocalFsRH[i], tempIndices + threadIdx.x, tempFs + threadIdx.x);
    }
  }
  __syncthreads();
  argminReduce(tempFs, tempIndices);
  int iHigh = tempIndices[0];
  float bHigh = tempFs[0];

  if (threadIdx.x == 0) {
    *((float*)devResult + 3) = bHigh;
    *((int*)devResult + 7) = iHigh;
  }
}


int secondOrderPhaseThreeSize(bool iHighCompute, int nDimension) {
	int size = 0;
	if (iHighCompute) { size+= sizeof(float) * nDimension; }
	return size;
}

template <bool iHighCompute, class Kernel>
  __global__ void secondOrderPhaseThree(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float* devKernelDiag, float epsilon, float cEpsilon, float* devAlpha, float* devF, float bHigh, int iHigh, float* devCache, int devCachePitchInFloats, int iHighCacheIndex, int nDimension, int nPoints, float parameterA, float parameterB, float parameterC, float* devLocalFsRL, int* devLocalIndicesMaxObj, float* devLocalObjsMaxObj) {
	extern __shared__ float xIHigh[];
  __shared__ int tempIndices[BLOCKSIZE];
  __shared__ float tempValues[BLOCKSIZE];
  __shared__ float iHighSelfKernel;
  
	
	if (iHighCompute) {
    //Load xIHigh into shared memory
    coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iHigh, nDimension, xIHigh);
	}

  if (threadIdx.x == 0) {
    iHighSelfKernel = devKernelDiag[iHigh];
  }
  
	__syncthreads();


  int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

  float alpha;
  float f;
  float label;
  int reduceFlag;
	float obj;
	
  if (globalIndex < nPoints) {
    alpha = devAlpha[globalIndex];

		f = devF[globalIndex];
		label = devLabels[globalIndex];
    
		float highKernel;
		
		if (iHighCompute) {
			highKernel = 0;
		} else {
			highKernel = devCache[((size_t)devCachePitchInFloats * iHighCacheIndex) + globalIndex];
		}
		
    if (iHighCompute) {
      highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats, devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1, parameterA, parameterB, parameterC);
      devCache[((size_t)devCachePitchInFloats * iHighCacheIndex) + globalIndex] = highKernel;
    }

    
    float beta = bHigh - f;

    float kappa = iHighSelfKernel + devKernelDiag[globalIndex] - 2 * highKernel;
    
    if (kappa <= 0) {
      kappa = epsilon;
    }
    
    obj = beta * beta / kappa;
    if (((label > 0) && (alpha > epsilon)) ||
        ((label < 0) && (alpha < cEpsilon))) {
      if (beta <= epsilon) {
        reduceFlag = REDUCE1 | REDUCE0;
      } else {        
        reduceFlag = REDUCE0;
      }
    } else {
      reduceFlag = NOREDUCE;
    }
  } else {
    reduceFlag = NOREDUCE;
  }

  if ((reduceFlag & REDUCE0) == 0) {
    tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
  } else {
    tempValues[threadIdx.x] = f;
  }
  
	__syncthreads();
  
  maxReduce(tempValues);
  if (threadIdx.x == 0) {
  	devLocalFsRL[blockIdx.x] = tempValues[0];
  }

  if ((reduceFlag & REDUCE1) == 0) {
    tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
    tempIndices[threadIdx.x] = 0;
  } else {
    tempValues[threadIdx.x] = obj;
    tempIndices[threadIdx.x] = globalIndex;
  }
  __syncthreads();
  argmaxReduce(tempValues, tempIndices);
  
	if (threadIdx.x == 0) {
		devLocalIndicesMaxObj[blockIdx.x] = tempIndices[0];
		devLocalObjsMaxObj[blockIdx.x] = tempValues[0];
	}
}


int secondOrderPhaseFourSize() {
	int size = 0;
	return size;
}


__global__ void secondOrderPhaseFour(float* devLabels, float* devKernelDiag, float* devF, float* devAlpha, float cost, int iHigh, float bHigh, void* devResult, float* devCache, int devCachePitchInFloats, int iHighCacheIndex, float* devLocalFsRL, int* devLocalIndicesMaxObj, float* devLocalObjsMaxObj, int inputSize, int iteration) {
  __shared__ int tempIndices[BLOCKSIZE];
  __shared__ float tempValues[BLOCKSIZE];

  
  if (threadIdx.x < inputSize) {
    tempIndices[threadIdx.x] = devLocalIndicesMaxObj[threadIdx.x];
    tempValues[threadIdx.x] = devLocalObjsMaxObj[threadIdx.x];
  } else {
    tempValues[threadIdx.x] = -FLT_MAX;
    tempIndices[threadIdx.x] = -1;
  }

  if (inputSize > BLOCKSIZE) {
    for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
      argMax(tempIndices[threadIdx.x], tempValues[threadIdx.x], devLocalIndicesMaxObj[i], devLocalObjsMaxObj[i], tempIndices + threadIdx.x, tempValues + threadIdx.x);
    }
  }
  __syncthreads();

  
  argmaxReduce(tempValues, tempIndices);

  __syncthreads();
  int iLow;
  if (threadIdx.x == 0) {
    iLow = tempIndices[0];
    /* int temp = tempIndices[0]; */
/*     if (temp < 0) { */
/*       iLow = 17; */
/*     } else if (temp > 4176) { */
/*       iLow = 18; */
/*     } else { */
/*       iLow = temp; */
/*     } */
    
  }

  /* if (iteration > 1721) { */
/*     if (threadIdx.x == 0) { */
/*       //\*((float*)devResult + 0) = devAlpha[iLow]; */
/*       //\*((float*)devResult + 1) = devAlpha[iHigh]; */
/*       //\*((float*)devResult + 2) = bLow; */
/*       //\*((float*)devResult + 3) = bHigh; */
 
/*       *((int*)devResult + 6) = iLow; */
/*       *((int*)devResult + 7) = iHigh; */
/*     } */
/*     return; */
/*   } */
 
  if (threadIdx.x < inputSize) {
    tempValues[threadIdx.x] = devLocalFsRL[threadIdx.x];
  } else {
    tempValues[threadIdx.x] = -FLT_MAX;
  }
  
  if (inputSize > BLOCKSIZE) {
    for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
      maxOperator(tempValues[threadIdx.x], devLocalFsRL[i], tempValues + threadIdx.x);
    }
  }
  __syncthreads();
  maxReduce(tempValues);
  __syncthreads();
  float bLow;
  if (threadIdx.x == 0) {
    bLow = tempValues[0];
  }
  
  
  if (threadIdx.x == 0) {
    
    float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
   
    eta = eta - 2 * (*(devCache + ((size_t)devCachePitchInFloats * iHighCacheIndex) + iLow));
   
    
    float alpha1Old = devAlpha[iHigh];
    float alpha2Old = devAlpha[iLow];
    float alphaDiff = alpha2Old - alpha1Old;
    float lowLabel = devLabels[iLow];
    float sign = devLabels[iHigh] * lowLabel;
    float alpha2UpperBound;
    float alpha2LowerBound;
    if (sign < 0) {
      if (alphaDiff < 0) {
        alpha2LowerBound = 0;
				alpha2UpperBound = cost + alphaDiff;
      } else {
        alpha2LowerBound = alphaDiff;
        alpha2UpperBound = cost;
      }
    } else {
      float alphaSum = alpha2Old + alpha1Old;
      if (alphaSum < cost) {
        alpha2UpperBound = alphaSum;
        alpha2LowerBound = 0;
      } else {
        alpha2LowerBound = alphaSum - cost;
        alpha2UpperBound = cost;
			}
    }
    float alpha2New;
    if (eta > 0) {
      alpha2New = alpha2Old + lowLabel*(devF[iHigh] - devF[iLow])/eta;
      if (alpha2New < alpha2LowerBound) {
        alpha2New = alpha2LowerBound;
      } else if (alpha2New > alpha2UpperBound) {
				alpha2New = alpha2UpperBound;
      }
    } else {
      float slope = lowLabel * (bHigh - bLow);
      float delta = slope * (alpha2UpperBound - alpha2LowerBound);
      if (delta > 0) {
        if (slope > 0) {
          alpha2New = alpha2UpperBound;
        } else {
          alpha2New = alpha2LowerBound;
        }
      } else {
        alpha2New = alpha2Old;
      }
    }
    float alpha2Diff = alpha2New - alpha2Old;
    float alpha1Diff = -sign*alpha2Diff;
    float alpha1New = alpha1Old + alpha1Diff;
    devAlpha[iLow] = alpha2New;
    devAlpha[iHigh] = alpha1New;
   
    *((float*)devResult + 0) = alpha2Old;
    *((float*)devResult + 1) = alpha1Old;
    *((float*)devResult + 2) = bLow;
    *((float*)devResult + 3) = bHigh;
    *((float*)devResult + 4) = alpha2New;
    *((float*)devResult + 5) = alpha1New;
    *((int*)devResult + 6) = iLow;
    *((int*)devResult + 7) = iHigh;
  }
}

void launchSecondOrder(bool iLowCompute, bool iHighCompute, int kType, int nPoints, int nDimension, dim3 blocksConfig, dim3 threadsConfig, dim3 globalThreadsConfig, float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float epsilon, float cEpsilon, float* devAlpha, float* devF, float sAlpha1Diff, float sAlpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, CacheGpuSVM* kernelCache, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, int* devLocalIndicesRH, float* devLocalFsRH, float* devLocalFsRL, int* devLocalIndicesMaxObj, float* devLocalObjsMaxObj, float* devKernelDiag, void* devResult, float* hostResult, float cost, int iteration) {

  int phaseOneSize = secondOrderPhaseOneSize(iLowCompute, iHighCompute, nDimension);
  int phaseTwoSize = secondOrderPhaseTwoSize();
 
  if (iLowCompute == true) {
    if (iHighCompute == true) {
      switch (kType) {
      case LINEAR:
        secondOrderPhaseOne <true, true, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case POLYNOMIAL:
        secondOrderPhaseOne <true, true, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case GAUSSIAN:
        secondOrderPhaseOne <true, true, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case SIGMOID:
        secondOrderPhaseOne <true, true, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      }
    } else if (iHighCompute == false) {
      switch (kType) {
      case LINEAR:
        secondOrderPhaseOne <true, false, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case POLYNOMIAL:
        secondOrderPhaseOne <true, false, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case GAUSSIAN:
        secondOrderPhaseOne <true, false, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case SIGMOID:
        secondOrderPhaseOne <true, false, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      }
          
    }
  } else if (iLowCompute == false) {
    if (iHighCompute == true) {
      switch (kType) {
      case LINEAR:
        secondOrderPhaseOne <false, true, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case POLYNOMIAL:
        secondOrderPhaseOne <false, true, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case GAUSSIAN:
        secondOrderPhaseOne <false, true, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case SIGMOID:
        secondOrderPhaseOne <false, true, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      }
    } else if (iHighCompute == false) {
      switch (kType) {
      case LINEAR:
        secondOrderPhaseOne <false, false, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case POLYNOMIAL:
        secondOrderPhaseOne <false, false, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case GAUSSIAN:
        secondOrderPhaseOne <false, false, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      case SIGMOID:
        secondOrderPhaseOne <false, false, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
        break;
      }
          
    }
  }


  secondOrderPhaseTwo<<<1, globalThreadsConfig, phaseTwoSize>>>(devResult, devLocalIndicesRH, devLocalFsRH, blocksConfig.x);
   

   CUDA_SAFE_CALL(cudaMemcpy((void*)hostResult, devResult, 8*sizeof(float), cudaMemcpyDeviceToHost));
    
		//float eta = *(hostResult);
		float bHigh = *(hostResult + 3);
		iHigh = *((int*)hostResult + 7);
   
   kernelCache->findData(iHigh, iHighCacheIndex, iHighCompute);
 
   int phaseThreeSize = secondOrderPhaseThreeSize(iHighCompute, nDimension);
  
 
   if (iHighCompute == true) {
     switch (kType) {
     case LINEAR:
       secondOrderPhaseThree<true, Linear><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     case POLYNOMIAL:
       secondOrderPhaseThree<true, Polynomial><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     case GAUSSIAN:
       secondOrderPhaseThree<true, Gaussian><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     case SIGMOID:
       secondOrderPhaseThree<true, Sigmoid><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     }
   } else {
     switch (kType) {
     case LINEAR:
       secondOrderPhaseThree<false, Linear><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     case POLYNOMIAL:
       secondOrderPhaseThree<false, Polynomial><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     case GAUSSIAN:
       secondOrderPhaseThree<false, Gaussian><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     case SIGMOID:
       secondOrderPhaseThree<false, Sigmoid><<<blocksConfig, threadsConfig, phaseThreeSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
       break;
     }
   }
   /* if (iteration == 1722) { */
/*    float* localObjsMaxObj = (float*)malloc(blocksConfig.x * sizeof(float)); */
/*    int* localIndicesMaxObj = (int*)malloc(blocksConfig.x * sizeof(int)); */
/*    cudaMemcpy(localObjsMaxObj, devLocalObjsMaxObj, sizeof(float) * blocksConfig.x, cudaMemcpyDeviceToHost); */
/*    cudaMemcpy(localIndicesMaxObj, devLocalIndicesMaxObj, sizeof(int) * blocksConfig.x, cudaMemcpyDeviceToHost); */
/*    for(int i = 0; i < blocksConfig.x; i++) { */
/*      printf("(%i: %f)\n", localIndicesMaxObj[i], localObjsMaxObj[i]); */
/*    } */
/*    free(localObjsMaxObj); */
/*    free(localIndicesMaxObj); */
/*    }       */          
   
   secondOrderPhaseFour<<<1, globalThreadsConfig, phaseTwoSize>>>(devLabels, devKernelDiag, devF, devAlpha, cost, iHigh, bHigh, devResult, devCache, devCachePitchInFloats, iHighCacheIndex, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj, blocksConfig.x, iteration);
    
} 



#endif
