#include "svmTrain.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "../common/framework.h"
#include "../common/deviceSelect.h"
#include "Cache.h"
#include "Controller.h"
#include "svmKernels.h"
#include "initialize.h"
#include "firstOrder.h"
#include "secondOrder.h"

void formModel(float* trainingPoints, int nTrainingPoints, int nDimension, float* trainingAlpha, float* trainingLabels, float** supportVectors, int* nSV, float** alpha, float epsilon) {
  int count = 0;
  
  for(int i = 0; i < nTrainingPoints; i++) {
    if (trainingAlpha[i] > epsilon) {
      count++;
    }
  }
  *nSV = count;
  printf("%i support vectors found\n", count);
  float* mySupportVectors = *supportVectors;
  mySupportVectors = (float*)malloc(count * nDimension * sizeof(float));
  *supportVectors = mySupportVectors;

  float* myAlpha = *alpha;
  myAlpha = (float*)malloc(count * sizeof(float));
  *alpha = myAlpha;
  int currentOutput = 0;
  for(int i = 0; i < nTrainingPoints; i++) {
    if (trainingAlpha[i] > epsilon) {
      float* sourcePointer = &trainingPoints[i];
      float* destinationPointer = &mySupportVectors[currentOutput];
      for(int j = 0; j < nDimension; j++) {
        *destinationPointer = *sourcePointer;
        sourcePointer += nTrainingPoints;
        destinationPointer += count;
      }
      myAlpha[currentOutput] = trainingAlpha[i] * trainingLabels[i];
      currentOutput++;
    }
  }
}


void performTraining(float* data, int nPoints, int nDimension, float* labels, float** p_alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData) {
  //chooseLargestGPU(true);
  
  float cEpsilon = cost - epsilon;
  Controller progress(2.0, heuristicMethod, 64, nPoints);

  int kType = GAUSSIAN;
  float parameterA = 0.0F;
  float parameterB = 0.0F;
  float parameterC = 0.0F;
  if (kp->kernel_type.compare(0,3,"rbf") == 0) {
    parameterA = -kp->gamma;
    kType = GAUSSIAN;
    printf("Gaussian kernel: gamma = %f\n", -parameterA);
  } else if (kp->kernel_type.compare(0,10,"polynomial") == 0) {
    parameterA = kp->gamma;
    parameterB = kp->coef0;
    parameterC = kp->degree;
    kType = POLYNOMIAL;
    printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
  } else if (kp->kernel_type.compare(0,6,"linear") == 0) {
    kType = LINEAR;
    printf("Linear kernel\n");
  } else if (kp->kernel_type.compare(0,7,"sigmoid") == 0) {
    kType = SIGMOID;
    parameterA = kp->gamma;
    parameterB = kp->coef0;
    printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
    if ((parameterA <= 0) || (parameterB < 0)) {
      printf("Invalid Parameters\n");
      exit(1);
    }
  }
  printf("Cost: %f, Tolerance: %f, Epsilon: %f\n", cost, tolerance, epsilon);
  
  
  float* devData;
  float* devTransposedData;
  size_t devDataPitch;
  size_t devTransposedDataPitch;
  int hostPitchInFloats = nPoints;

  float* hostData;
  bool hostDataAlloced = false;
  
  CUDA_SAFE_CALL(cudaMallocPitch((void**)&devData, &devDataPitch, nPoints*sizeof(float), nDimension));
  CUDA_SAFE_CALL(cudaMemset(devData, 0, devDataPitch * nDimension));
  if (devDataPitch == nPoints * sizeof(float)) {
    printf("Data is already aligned\n");
    hostData = data;
  } else {
    hostPitchInFloats = devDataPitch/sizeof(float);	
    hostData = (float*)malloc(devDataPitch*nDimension);
    hostDataAlloced = true;
    printf("Realigning data to a pitch of %i floats\n", hostPitchInFloats);
    for(int i=nDimension-1;i>=0;i--)
      {
        for(int j=nPoints-1;j>=0;j--)
          {
            hostData[i*hostPitchInFloats+j]=data[i*nPoints+j];
          }
      }
  }
  CUDA_SAFE_CALL(cudaMemcpy(devData, hostData, devDataPitch*nDimension, cudaMemcpyHostToDevice));
  bool transposedDataAlloced = false;
  if (transposedData == 0) {
    transposedData = (float*)malloc(sizeof(float) * nPoints * nDimension);
    transposedDataAlloced = true;
    for(int i = 0; i < nPoints; i++) {
      for (int j = 0; j < nDimension; j++) {
        transposedData[i*nDimension + j] = hostData[j*hostPitchInFloats + i];
      }
    }
  }

  float* alpha = (float*)malloc(sizeof(float) * nPoints);
  *p_alpha = alpha;
  CUDA_SAFE_CALL(cudaMallocPitch((void**)&devTransposedData, &devTransposedDataPitch, nDimension*sizeof(float), nPoints));
  CUDA_SAFE_CALL(cudaMemcpy2D(devTransposedData, devTransposedDataPitch, transposedData, nDimension*sizeof(float), nDimension*sizeof(float),
                              nPoints, cudaMemcpyHostToDevice));
																
  float* devLabels;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devLabels, nPoints*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(devLabels, labels, nPoints*sizeof(float), cudaMemcpyHostToDevice));


  float* devKernelDiag;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devKernelDiag, nPoints*sizeof(float)));
	
	
  float* devAlpha;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devAlpha, nPoints*sizeof(float)));

  float* devF;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devF, nPoints*sizeof(float)));
	
  void* devResult;
  CUDA_SAFE_CALL(cudaMalloc(&devResult, 8*sizeof(float)));
  float* hostResult = (float*)malloc(8*sizeof(float));

	
	

  int blockWidth = intDivideRoundUp(nPoints, BLOCKSIZE);

  float* devLocalFsRL;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalFsRL, blockWidth*sizeof(float)));
  float* devLocalFsRH;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalFsRH, blockWidth*sizeof(float))); 
  int* devLocalIndicesRL;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesRL, blockWidth*sizeof(int)));
  int* devLocalIndicesRH;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesRH, blockWidth*sizeof(int)));

  float* devLocalObjsMaxObj;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalObjsMaxObj, blockWidth*sizeof(float)));
  int* devLocalIndicesMaxObj;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesMaxObj, blockWidth*sizeof(int)));
  
  
  void* temp;
  size_t rowPitch;
  CUDA_SAFE_CALL(cudaMallocPitch(&temp, &rowPitch, nPoints*sizeof(float), 2));
  CUDA_SAFE_CALL(cudaFree(temp));


	
  size_t remainingMemory;
  size_t totalMemory;
  cuMemGetInfo(&remainingMemory, &totalMemory);

  size_t sizeOfCache = remainingMemory/rowPitch;
  sizeOfCache = (int)((float)sizeOfCache*0.95);//If I try to grab all the memory available, it'll fail
  if (nPoints < sizeOfCache) {
    sizeOfCache = nPoints;
  }

#ifdef __DEVICE_EMULATION__
  sizeOfCache = nPoints;
#endif
		
  printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);
  printf("%u rows of kernel matrix will be cached (%u bytes per row)\n", sizeOfCache, rowPitch);

  float* devCache;
  size_t cachePitch;
  CUDA_SAFE_CALL(cudaMallocPitch((void**)&devCache, &cachePitch, nPoints*sizeof(float), sizeOfCache));
  //cudaMemset2D(devCache, cachePitch, 0x00, nPoints*sizeof(float), sizeOfCache);
  CacheGpuSVM kernelCache(nPoints, sizeOfCache);
  int devCachePitchInFloats = (int)cachePitch/(sizeof(float));

  cudaError_t err = cudaGetLastError();
  if(err) printf("Error: %s\n", cudaGetErrorString(err));
  printf("Allocated arrays on GPU\n");


	
  dim3 threadsLinear(BLOCKSIZE);
  dim3 blocksLinear(blockWidth);

  
  int devDataPitchInFloats = ((int)devDataPitch) >> 2;
  int devTransposedDataPitchInFloats = ((int)devTransposedDataPitch) >> 2;
	
  
  launchInitialization(devData, devDataPitchInFloats, nPoints, nDimension, kType, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels, blocksLinear, threadsLinear);
  err = cudaGetLastError();
  if(err) printf("Error: %s\n", cudaGetErrorString(err));
  printf("Initialization complete\n");

  //Choose initial points
  float bLow = 1;
  float bHigh = -1;
  int iteration = 0;
  int iLow = -1;
  int iHigh = -1;
  for (int i = 0; i < nPoints; i++) {
    if (labels[i] < 0) {
      if (iLow == -1) {
        iLow = i;
        if (iHigh > -1) {
          i = nPoints; //Terminate
        }
      }
    } else {
      if (iHigh == -1) {
        iHigh = i;
        if (iLow > -1) {
          i = nPoints; //Terminate
        }
      }
    }
  }

	
  dim3 singletonThreads(1);
  dim3 singletonBlocks(1);
  launchTakeFirstStep(devResult, devKernelDiag, devData, devDataPitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, kType, parameterA, parameterB, parameterC, singletonBlocks, singletonThreads);
  CUDA_SAFE_CALL(cudaMemcpy((void*)hostResult, devResult, 8*sizeof(float), cudaMemcpyDeviceToHost));


  float alpha2Old = *(hostResult + 0);
  float alpha1Old = *(hostResult + 1);
  bLow = *(hostResult + 2);
  bHigh = *(hostResult + 3);
  float alpha2New = *(hostResult + 6);
  float alpha1New = *(hostResult + 7);

  float alpha1Diff = alpha1New - alpha1Old;
  float alpha2Diff = alpha2New - alpha2Old;
	
  int iLowCacheIndex;
  int iHighCacheIndex;
  bool iLowCompute;
  bool iHighCompute; 
  

  dim3 reduceThreads(BLOCKSIZE);

  
  printf("Starting iterations\n");
	
  for (iteration = 1; true; iteration++) {
	
    if (bLow <= bHigh + 2*tolerance) {
      printf("Converged\n");
      break; //Convergence!!
    }

    /*if ((iteration & 0x7ff) == 0) {
      printf("iteration: %d; gap: %f\n",iteration, bLow - bHigh);
    }*/
        
    if ((iteration & 0x7f) == 0) {
      heuristicMethod = progress.getMethod();
    }

    
    kernelCache.findData(iHigh, iHighCacheIndex, iHighCompute);
    
    kernelCache.findData(iLow, iLowCacheIndex, iLowCompute);


   
    if (heuristicMethod == FIRSTORDER) {
      launchFirstOrder(iLowCompute, iHighCompute, kType, nPoints, nDimension, blocksLinear, threadsLinear, reduceThreads, devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, epsilon, cEpsilon, devAlpha, devF, alpha1Diff * labels[iHigh], alpha2Diff * labels[iLow], iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRH, devLocalFsRL, devKernelDiag, devResult, cost);
    } else {
      launchSecondOrder(iLowCompute, iHighCompute, kType, nPoints, nDimension, blocksLinear, threadsLinear, reduceThreads, devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, epsilon, cEpsilon, devAlpha, devF, alpha1Diff * labels[iHigh], alpha2Diff * labels[iLow], iLow, iHigh, parameterA, parameterB, parameterC, &kernelCache, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj, devKernelDiag, devResult, hostResult, cost, iteration);
    }
    CUDA_SAFE_CALL(cudaMemcpy((void*)hostResult, devResult, 8*sizeof(float), cudaMemcpyDeviceToHost));

    alpha2Old = *(hostResult + 0);
    alpha1Old = *(hostResult + 1);
    bLow = *(hostResult + 2);
    bHigh = *(hostResult + 3);
    iLow = *((int*)hostResult + 6);
    iHigh = *((int*)hostResult + 7);
    alpha2New = *(hostResult + 4);
    alpha1New = *(hostResult + 5);
    alpha1Diff = alpha1New - alpha1Old;
    alpha2Diff = alpha2New - alpha2Old;
    progress.addIteration(bLow-bHigh);
    
  }

	
	
  printf("%d iterations\n", iteration);
  printf("bLow: %f, bHigh: %f\n", bLow, bHigh);
  kp->b = (bLow + bHigh) / 2;
  kernelCache.printStatistics();
  CUDA_SAFE_CALL(cudaMemcpy((void*)alpha, devAlpha, nPoints*sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(devData);
  cudaFree(devTransposedData);
  cudaFree(devLabels);
  cudaFree(devAlpha);
  cudaFree(devF);
  cudaFree(devCache);
  cudaFree(devLocalIndicesRL);
  cudaFree(devLocalIndicesRH);
  cudaFree(devLocalFsRH);
  cudaFree(devLocalFsRL);
  cudaFree(devKernelDiag);
  cudaFree(devResult);
  cudaFree(devLocalIndicesMaxObj);
  cudaFree(devLocalObjsMaxObj);
  if (hostDataAlloced) {
    free(hostData);
  }
  if (transposedDataAlloced) {
    free(transposedData);
  }
  free(hostResult);
}
