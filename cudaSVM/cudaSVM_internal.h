#ifndef __CUDASVM_INTERNAL_H
#define __CUDASVM_INTERNAL_H

#include <float.h>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <svm.h>

#define ON 1
#define OFF 0

#define FLOATS_IN_FLOAT4 4
#define FLOATS_IN_FLOAT4_MOVE 2

#define MAX_ITERS (1 << 20) // (8388608 = 1 << 23)
#define FORCE_CACHE OFF

#define WARP_SIZE (32)
#define NUM_THREADS (WARP_SIZE * 8) /*from 32, 64, 128, ... , 768 or 1024 - depends on GPU.
							Needs to be a multiply of WARP_SIZE! */
#define NUM_THREADS_X (WARP_SIZE)
#define NUM_THREADS_Y (WARP_SIZE >> 2)
#define NUM_THREADS_X_INIT (16)
#define NUM_THREADS_Y_INIT (NUM_THREADS_X_INIT >> 2)
#define NUM_THREADS_REDUCE (NUM_THREADS_X * NUM_THREADS_Y)

#define UNROLL_X (1)
#define NEG_INF (-FLT_MAX)
#define TAU (1e-12f)

#define F4_1 make_float4(1.0f, 1.0f, 1.0f, 1.0f)
#define F4_0 make_float4(0.0f, 0.0f, 0.0f, 0.0f)
#define ALIGN_UP(x, align) ((align) * (((x) + align - 1) / (align)))
#define ALIGN_DOWN(x, align) ((align) * ((x) / (align)))

#define CUDA_CHECK_KERNEL	{cudaThreadSynchronize(); \
cudaError_t _error_ = cudaGetLastError(); \
if (_error_ != cudaSuccess) { \
	printf("Kernel launch failed (error: %d - %s, file:%s, line:%d)\n", \
		_error_, cudaGetErrorString(_error_), __FILE__, __LINE__); \
	return _error_; \
}}

#define CUDA_DEBUG_PRINT(PTR, TYPE, COUNT) {\
TYPE * _ptr_ = (TYPE *) malloc(COUNT * sizeof(TYPE)); \
cudaMemcpy(_ptr_, PTR, COUNT * sizeof(TYPE), cudaMemcpyDeviceToHost); \
for (int _a_ = 0; _a_ < COUNT; _a_++) { \
	printf("%d: %d (%f)\n", _a_, ((int *) _ptr_)[_a_], ((float *) _ptr_)[_a_]); \
} \
printf("\n"); \
getchar(); \
free(_ptr_); \
}

#define CUDA_DEBUG_FPRINT_BIN(PTR, TYPE, WIDTH, PITCH, HEIGHT, NAME) {\
const int _count_ = (WIDTH) * (HEIGHT); \
TYPE * _ptr_ = (TYPE *) malloc(_count_ * sizeof(TYPE)); \
FILE * _fid_ = fopen(NAME, "wb"); \
cudaMemcpy2D(_ptr_, (WIDTH) * sizeof(TYPE), PTR, PITCH, (WIDTH) * sizeof(TYPE), HEIGHT, cudaMemcpyDeviceToHost); \
fwrite(&(HEIGHT), sizeof(int), 1, _fid_); \
fwrite(&(WIDTH), sizeof(int), 1, _fid_); \
fwrite(_ptr_, sizeof(TYPE), (WIDTH) * (HEIGHT), _fid_); \
fclose(_fid_); \
free(_ptr_); \
}

// Operators for Cuda specific data types.
__inline __device__ void operator+=(float4 &a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

__inline __device__ void operator*=(float4 &a, float4 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}

__inline __device__ float4 operator-(float4 a, float4 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__inline __device__ float4 operator+(float4 a, float b) {
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__inline __device__ float4 operator*(float4 a, float4 b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__inline __device__ float4 operator*(float a, float4 b) {
	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__inline __device__ int4 operator*(int a, int4 b) {
	return make_int4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__inline __device__ float4 operator*(int4 a, float4 b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

// Inline functions used inside kernels
__inline __device__ void minVandI(int value, int &minValue, int idx, int &minIdx){
	if (value > -1 && value <= minValue) {
		minValue = value;
		minIdx = idx;
	}
}

__inline __device__ float4 expf4(float4 value) {
	return make_float4(expf(value.x), expf(value.y), expf(value.z), expf(value.w));
}

__inline __device__ float4 powf4(float4 value, float power) {
	return make_float4(pow(value.x, power), pow(value.y, power),
					   pow(value.z, power), pow(value.w, power));
}

__inline __device__ float4 tanhf4(float4 value) {
	return make_float4(tanh(value.x), tanh(value.y), tanh(value.z), tanh(value.w));
}

#endif /* __CUDASVM_INTERNAL_H */
