#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_
#include <stdio.h>

#define SUCCESS 0
#define FAILURE 1

#define TRUE 1
#define FALSE 0

#define CUDA_SAFE_MALLOC(MEM_PTR, SIZE) CUDA_SAFE_MFREE(*MEM_PTR) \
if (cudaMalloc(MEM_PTR, SIZE) != cudaSuccess) { \
	printf("Error(%s): Unable to allocate %d B of memory on GPU!\n", cudaGetErrorString(cudaGetLastError()), SIZE); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MALLOC_2D(MEM_PTR, PITCH, WIDTH, HEIGHT) CUDA_SAFE_MFREE(*MEM_PTR) \
if (cudaMallocPitch(MEM_PTR, PITCH, WIDTH, HEIGHT) != cudaSuccess) { \
	printf("Error(%s): Unable to allocate %d B of 2D memory on GPU!\n", cudaGetErrorString(cudaGetLastError()), WIDTH * HEIGHT); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MEMCPY(DST, SRC, SIZE, KIND) \
if (cudaMemcpy(DST, SRC, SIZE, KIND) != cudaSuccess) { \
	printf("Error: Unable to copy %d B of memory from RAM (%llX) to GPU (%llX)!\n", SIZE, SRC, DST); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MEMCPY_2D(DST, SRC, PITCH, WIDTH, HEIGHT, KIND) \
if (cudaMemcpy2D(DST, PITCH, SRC, WIDTH, WIDTH, HEIGHT, KIND) != cudaSuccess) { \
	printf("Error: Unable to copy %d B of 2D memory from RAM (%llX) to GPU (%llX) due to %s !\n", WIDTH * HEIGHT, SRC, DST, cudaGetErrorString(cudaGetLastError())); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MFREE(MEM_PTR) if (MEM_PTR != NULL) { \
if (cudaFree(MEM_PTR) != cudaSuccess) { \
	printf("Error(%s): Unable to free GPU memory at address %llX!\n", cudaGetErrorString(cudaGetLastError()), MEM_PTR); \
	exit(EXIT_FAILURE); \
} \
MEM_PTR = NULL; \
}

#define CUDA_SAFE_MFREE_HOST(MEM_PTR) if (MEM_PTR != NULL) { \
if (cudaFreeHost(MEM_PTR) != cudaSuccess) { \
	printf("Error(%s): Unable to free host data at address %llX!\n", cudaGetErrorString(cudaGetLastError()), MEM_PTR); \
	exit(EXIT_FAILURE); \
} \
MEM_PTR = NULL; \
}
	
#endif /* _CUDA_UTILS_H_ */
