#ifndef __CUDASVM_KERNELS_H
#define __CUDASVM_KERNELS_H

#include "cudaSVM_internal.h"

__device__ void LocalReduce(float &maxValue, int &maxIdx, float *maxs, int *idxs);
__device__ void GlobalReduce(float &maxValue, int &maxIdx, float *maxs, int *idxs,
							 int *glob_mutex, float2 *glob_max, int &glob_sync);
__inline __device__ float SearchForMinGapLocal(int * alphaStatus, double * F, float2 * result,
											  float * maxs, int * idxs);

__inline __device__ void maxFandI(float value, float &maxValue, int idx, int &maxIdx){
	if (value > maxValue) {
		maxValue = value;
		maxIdx = idx;
	}
}

// Sum of all 4 values from float4 variable stored in float.
#define SUM_FLOAT4_TO_FLOAT(V) (V.x + V.y + V.z + V.w)

/* Universal (not SM dependent). */
__inline __device__ void final_reduction_indep(float * sh_val, int * sh_idx,
											   int tid, int num_threads) {
	/*volatile float * sh_val = _sh_val;
	volatile int * sh_idx = _sh_idx;*/
	if (num_threads >= 512) {
		if (tid < 256) {
			maxFandI(sh_val[256], *sh_val, sh_idx[256], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 256) {
		if (tid < 128) {
			maxFandI(sh_val[128], *sh_val, sh_idx[128], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 128) {
		if (tid < 64) {
			maxFandI(sh_val[64], *sh_val, sh_idx[64], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 64) {
		if (tid < 32) {
			maxFandI(sh_val[32], *sh_val, sh_idx[32], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 32) {
		if (tid < 16) {
			maxFandI(sh_val[16], *sh_val, sh_idx[16], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 16) {
		if (tid < 8) {
			maxFandI(sh_val[8], *sh_val, sh_idx[8], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 8) {
		if (tid < 4) {
			maxFandI(sh_val[4], *sh_val, sh_idx[4], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 4) {
		if (tid < 2) {
			maxFandI(sh_val[2], *sh_val, sh_idx[2], *sh_idx);
		}
		__syncthreads();
	}
	if (num_threads >= 2) {
		if (tid < 1) {
			maxFandI(sh_val[1], *sh_val, sh_idx[1], *sh_idx);
		}
		__syncthreads();
	}
}

// Final reduction for block size higher or equal to 8.
#define final_reduction_8(sh_res, tid) { \
	if (NUM_THREADS_X >= 512) { \
		if (tid < 256) \
			sh_res[tid] += sh_res[tid + 256]; \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X >= 256) { \
		if (tid < 128) \
			sh_res[tid] += sh_res[tid + 128]; \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X >= 128) { \
		if (tid < 64) \
			sh_res[tid] += sh_res[tid + 64]; \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X >= 64) { \
		if (tid < 32) \
			sh_res[tid] += sh_res[tid + 32]; \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X >= 32) { \
		if (tid < 16) \
			sh_res[tid] += sh_res[tid + 16]; \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X >= 16) { \
		if (tid < 8) \
			sh_res[tid] += sh_res[tid + 8]; \
		__syncthreads(); \
	} \
	if (tid < 4) \
		sh_res[tid] += sh_res[tid + 4]; \
	__syncthreads(); \
	if (tid < 2) \
		sh_res[tid] += sh_res[tid + 2]; \
	__syncthreads(); \
	if (tid < 1) \
		sh_res[tid] += sh_res[tid + 1]; \
	__syncthreads(); \
}

#define final_quadreduction(sh_res_0, sh_res_1, sh_res_2, sh_res_3, tid) { \
	if (NUM_THREADS_X_INIT >= 512) { \
		if (tid < 256) { \
			(sh_res_0)[tid] += (sh_res_0)[tid + 256]; \
			(sh_res_1)[tid] += (sh_res_1)[tid + 256]; \
			(sh_res_2)[tid] += (sh_res_2)[tid + 256]; \
			(sh_res_3)[tid] += (sh_res_3)[tid + 256]; \
		} \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X_INIT >= 256) { \
		if (tid < 128) { \
			(sh_res_0)[tid] += (sh_res_0)[tid + 128]; \
			(sh_res_1)[tid] += (sh_res_1)[tid + 128]; \
			(sh_res_2)[tid] += (sh_res_2)[tid + 128]; \
			(sh_res_3)[tid] += (sh_res_3)[tid + 128]; \
		} \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X_INIT >= 128) { \
		if (tid < 64) { \
			(sh_res_0)[tid] += (sh_res_0)[tid + 64]; \
			(sh_res_1)[tid] += (sh_res_1)[tid + 64]; \
			(sh_res_2)[tid] += (sh_res_2)[tid + 64]; \
			(sh_res_3)[tid] += (sh_res_3)[tid + 64]; \
		} \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X_INIT >= 64) { \
		if (tid < 32) { \
			(sh_res_0)[tid] += (sh_res_0)[tid + 32]; \
			(sh_res_1)[tid] += (sh_res_1)[tid + 32]; \
			(sh_res_2)[tid] += (sh_res_2)[tid + 32]; \
			(sh_res_3)[tid] += (sh_res_3)[tid + 32]; \
		} \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X_INIT >= 32) { \
		if (tid < 16) { \
			(sh_res_0)[tid] += (sh_res_0)[tid + 16]; \
			(sh_res_1)[tid] += (sh_res_1)[tid + 16]; \
			(sh_res_2)[tid] += (sh_res_2)[tid + 16]; \
			(sh_res_3)[tid] += (sh_res_3)[tid + 16]; \
		} \
		__syncthreads(); \
	} \
	if (NUM_THREADS_X_INIT >= 16) { \
		if (tid < 8) { \
			(sh_res_0)[tid] += (sh_res_0)[tid + 8]; \
			(sh_res_1)[tid] += (sh_res_1)[tid + 8]; \
			(sh_res_2)[tid] += (sh_res_2)[tid + 8]; \
			(sh_res_3)[tid] += (sh_res_3)[tid + 8]; \
		} \
		__syncthreads(); \
	} \
	if (tid < 4) { \
		(sh_res_0)[tid] += (sh_res_0)[tid + 4]; \
		(sh_res_1)[tid] += (sh_res_1)[tid + 4]; \
		(sh_res_2)[tid] += (sh_res_2)[tid + 4]; \
		(sh_res_3)[tid] += (sh_res_3)[tid + 4]; \
	} \
	__syncthreads(); \
	if (tid < 2) { \
		(sh_res_0)[tid] += (sh_res_0)[tid + 2]; \
		(sh_res_1)[tid] += (sh_res_1)[tid + 2]; \
		(sh_res_2)[tid] += (sh_res_2)[tid + 2]; \
		(sh_res_3)[tid] += (sh_res_3)[tid + 2]; \
	} \
	__syncthreads(); \
	if (tid < 1) { \
		(sh_res_0)[tid] += (sh_res_0)[tid + 1]; \
		(sh_res_1)[tid] += (sh_res_1)[tid + 1]; \
		(sh_res_2)[tid] += (sh_res_2)[tid + 1]; \
		(sh_res_3)[tid] += (sh_res_3)[tid + 1]; \
	} \
	__syncthreads(); \
}

#define sign(i) ((int)(i > 0) - (int)(i < 0))
#define TRUE (1)
#define FALSE (0)
//__device__ int sign(int i) {
//	return ((int)(i > 0) - (int)(i < 0));
//}

//KERNELS
#define TID (threadIdx.y * blockDim.x + threadIdx.x)
#define TID_X (threadIdx.x)
#define TID_Y (threadIdx.y)
#define BID (blockIdx.x)

//<<<numBlocks, NUM_THREADS>>> 
//alphaStatus[gid] is set to y(i) at the begining 
//goal si to set F to -y and alphaStatus ti y * (1,2,3) = LOW, HIGH, FREE
__global__ void InitFandStatus(int *alphaStatus, double *F, int N) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < N) {
		int y = alphaStatus[gid];
		F[gid] = (double) -y;
	} else {
		F[gid] = 0.0;
		alphaStatus[gid] = 0;
	}

} //InitFandStatus

// InitReduceBuffer<<<1, numBlocksWarp>>>(devReduceTmp1);
__global__ void InitReduceBuffer(float2 * ReduceTmp1, int2 * ReduceTmp2) {
	//float2 value = make_float2(NEG_INF, __int_as_float(-1));
	ReduceTmp1[TID_X] = make_float2(NEG_INF, __int_as_float(-1));;
	if (TID_X < 5) {
		ReduceTmp2[TID_X] = make_int2(0, 0);
	}

} // InitReduceBuffer

// InitReduceBufferMutex<<<1, numBlocksWarp>>>(devReduceTmp1);
__global__ void InitReduceBufferAndMutex(float2 *ReduceTmp1, float2 *ReduceTmp2, int *MutexTmp1,
								 int *MutexTmp2, int *MutexTmp3, int *MutexTmp4) {
	float2 value = make_float2(NEG_INF, __int_as_float(-1));
	int idx = TID_X << 1;
	ReduceTmp1[TID_X] = value;
	ReduceTmp2[TID_X] = value;
	MutexTmp1[idx] = 0;
	MutexTmp1[idx + 1] = 0;
	MutexTmp2[idx] = 0;
	MutexTmp2[idx + 1] = 0;
	MutexTmp3[idx] = 0;
	MutexTmp3[idx + 1] = 0;
	MutexTmp4[idx] = 0;
	MutexTmp4[idx + 1] = 0;

} // End of kernel InitReduceBufferMutex.

__global__ void SelectI_stage1(int *alphaStatus, double *F, float2 *result, int numIt) {
	
	__shared__ float maxs[NUM_THREADS];
	__shared__ int idxs[NUM_THREADS];

	int k;
	int gid = numIt * blockDim.x * BID + TID_X;
	float maxValue = NEG_INF;
	int maxIdx = gid;
	
	//for (numIt--;numIt>0;numIt--) {
	for (; numIt > 0; numIt -= UNROLL_X) {
#pragma unroll
		for (k = 0; k < UNROLL_X; k++) {
			int as = alphaStatus[gid];
			float fg = (float) -F[gid];
			float f;
			f = (as == 1 || as == 3)? fg : NEG_INF; //if idx == 1 and not HIGH
			f = (as < -1)? fg : f; //if idx == -1 and not LOW
			maxFandI(f, maxValue, gid, maxIdx);
			gid += NUM_THREADS;
		}
	}

	maxs[TID_X] = maxValue;
	idxs[TID_X] = maxIdx;
	__syncthreads();
 
	if (TID_X < WARP_SIZE) {
#pragma unroll
		for (k = TID_X + WARP_SIZE; k < NUM_THREADS; k += WARP_SIZE) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}
		maxs[TID_X] = maxValue;
		idxs[TID_X] = maxIdx;
	}
	__syncthreads();

	if (TID_X == 0) {
#pragma unroll
		for (k = 1; k < WARP_SIZE; k++) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}

		//store results as float&int into float2
		float2 out;
		out.x = maxValue;
		out.y = __int_as_float(maxIdx);
		result[BID] = out;
	}
}

__global__ void SelectJ_stage1(int *alphaStatus, double *F, float *Q, size_t pitch, 
							  float2 *maxI_, float2 *result, int numIt) {
	
	__shared__ float maxs[NUM_THREADS];
	__shared__ int idxs[NUM_THREADS];

	int gid = numIt * blockDim.x * BID + TID_X;
	float maxValue = NEG_INF;
	int maxIdx = gid;
	float2 maxI = * maxI_;
	float y_i = sign(alphaStatus[__float_as_int(maxI.y)]); //load y_i

	// Shift matrix Q
	Q += __float_as_int(maxI.y) * pitch;
	
	//for (numIt--;numIt>0;numIt--) {
	for (; numIt > 0; numIt -= UNROLL_X) {
#pragma unroll
		for (int k = 0; k < UNROLL_X; k++, gid += NUM_THREADS) {
			int as = alphaStatus[gid];
			float f = (float) F[gid];
			bool b1 = (as > 1);
			bool b2 = (as == -1 || as == -3);
			float grad_diff = maxI.x + f;

			if ((b1 || b2) && (grad_diff > 0)) {
				float q = Q[gid];
				float eta = max(TAU, 2 - 2 * q);
				float obj = grad_diff * grad_diff / eta;
				maxFandI(obj, maxValue, gid, maxIdx);
			}
		}
	}

	maxs[TID_X] = maxValue;
	idxs[TID_X] = maxIdx;
	__syncthreads();
 
	if (TID_X < WARP_SIZE) {
#pragma unroll
		for (int k = TID_X + WARP_SIZE; k < NUM_THREADS; k += WARP_SIZE) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}
		maxs[TID_X] = maxValue;
		idxs[TID_X] = maxIdx;
	}
	__syncthreads();

	if (TID_X == 0) {
#pragma unroll
		for (int k = 1; k < WARP_SIZE; k++) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}

		//store results as float&int into float2
		float2 out;
		out.x = maxValue;
		out.y = __int_as_float(maxIdx);
		result[BID] = out;
	}
}

//__global__ void Reduce_stage2(float2 *y, float2 *r) {
//	
//	__shared__ float maxs[2*NUM_BLOCKS];
//	__shared__ int idxs[2*NUM_BLOCKS];
//
//	float2 f2 = y[TID_X];
//
//	float maxValue = f2.x;
//	int maxIdx = __float_as_int(f2.y);
//	
//	maxs[TID_X] = maxValue;
//	idxs[TID_X] = maxIdx;
//	__syncthreads();
//
//	if (TID_X<32) {
//#pragma unroll
//		for (int k=TID_X+32;k<NUM_BLOCKS;k+=32) {
//			float f = maxs[k];
//			int idx = idxs[k];
//			maxIdx = (maxValue < f)? idx : maxIdx;
//			maxValue = max(f, maxValue);
//		}
//		maxs[TID_X] = maxValue;
//		idxs[TID_X] = maxIdx;
//	}
//	__syncthreads();
//
//	if (TID_X<32) {
//#pragma unroll
//		for (int k=1;k<32;k++) {
//			float f = maxs[k];
//			int idx = idxs[k];
//			maxIdx = (maxValue < f)? idx : maxIdx;
//			maxValue = max(f, maxValue);
//		}
//
//		//store results as float&int into float2
//		if (TID_X==0) {
//			float2 out;
//			//out.x = maxs[0];
//			//((int *)&(out.y))[0] = idxs[0];
//			out.x = maxValue;
//			out.y = __int_as_float(maxIdx);
//			r[blockIdx.x] = out;
//		}
//	}
//}

__global__ void Reduce_finish(float2 *y, float2 *r, int numBlocks) {
	
	__shared__ float maxs[WARP_SIZE];
	__shared__ int idxs[WARP_SIZE];

	float2 f2;
	float maxValue;
	int maxIdx;
	
	if (numBlocks < WARP_SIZE && TID_X >= numBlocks) {
		f2 = make_float2(NEG_INF, __int_as_float(-1));
	} else {
		f2 = y[TID_X];
	}
	maxValue = f2.x;
	maxIdx = __float_as_int(f2.y);

	for (int k = TID_X + WARP_SIZE; k < numBlocks; k += WARP_SIZE) {
		f2 = y[k];
		maxFandI(f2.x, maxValue, __float_as_int(f2.y), maxIdx);
	}

	maxs[TID_X] = maxValue;
	idxs[TID_X] = maxIdx;
	__syncthreads();

	if (TID_X == 0) {
#pragma unroll
		for (int k = 1; k < WARP_SIZE; k++) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}

		//store results as float&int into float2
		f2.x = maxValue;
		f2.y = __int_as_float(maxIdx);
		r[BID] = f2;
	}
}


//"single thread" <<<1, 32>>> 
__global__ void TakeStep (int * IandJ, double * alpha, int * alphaStatus, double * F,
						  float * Q, size_t QPitchInFloats, double C, double eps, double * results) {
	int i1 = IandJ[1];
	int i2 = IandJ[3];
	
	//load stuff:
	double a1 = alpha[i1];
	double a2 = alpha[i2];
	double y1 = sign(alphaStatus[i1]);
	double y2 = sign(alphaStatus[i2]);
	double F1 = F[i1];
	double F2 = F[i2];
	double q = Q[QPitchInFloats * i1 + i2];
	
	//calculate new alphas
	double H,L;
	if (y1 == y2) {
		H = min(C, a1 + a2);
		L = max(0.0, a1 + a2 - C);
	} else {
		H = min(C, C + a2 - a1);
		L = max(0.0, a2 - a1);
	}
	
	double eta = y1 * y2 * 2 * q - 2.0;
	double a2_new = a2 - y2 * (F1 - F2) / eta;
	double a2_clip = min(max(a2_new, L), H);
	double a1_new = a1 + y1 * y2 * (a2 - a2_clip);

	//calc alpha diff
	double da1 = a1_new - a1;
	double da2 = a2_clip - a2;

	//correct rouding errors
    double s = -y1 * y2;

	if (a1 + da1 > C - TAU) {
        da2 = da2 + s * (C - a1 - da1);
        da1 = C - a1;
	}
	if (a2 + da2 > C - TAU) {
        da1 = da1 + s * (C - a2 - da2);
        da2 = C - a2;
	}
	if (a1 + da1 < TAU) {
        da2 = da2 - s * (a1 + da1);
        da1 = -a1;
	}
	if (a2 + da2 < TAU) {
        da1 = da1 - s * (a2 + da2);
        da2 = -a2;
	}

	//store alpha diffs
	results[0] = da1;
	results[1] = da2;
	
	a1 += da1;
	a2 += da2;
	alpha[i1] = a1;
	alpha[i2] = a2;
	
	C -= eps;
	alphaStatus[i1] = (int) y1 * (2 * (a1 > eps) + (a1 < C));
	alphaStatus[i2] = (int) y2 * (2 * (a2 > eps) + (a2 < C));

} //TakeStep

//<<<numBlocks, NUM_THREADS>>> 
__global__ void UpdateF (int * IandJ, double * AlphaDiff, int * AlphaStatus, double * F,
						 float * Q, size_t QPitchInFloats, int height) {
	int gid = blockDim.x * BID + TID_X;

	if (gid < height) {
		int i1 = IandJ[1];
		int i2 = IandJ[3];

		double Fi = F[gid];
		double y = sign(AlphaStatus[gid]);
		double da1 = AlphaDiff[0];
		double da2 = AlphaDiff[1];
		float Q1 = Q[i1 * QPitchInFloats + gid];
		float Q2 = Q[i2 * QPitchInFloats + gid];

		F[gid] = Fi + y * (Q1 * da1 + Q2 * da2);
	}

} //UpdateF

__inline __device__ float SearchForMinGapLocal(int * alphaStatus, double * F, float2 * result,
											  float * maxs, int * idxs) {
	int gid = blockDim.x * BID + TID_X;
	float maxValue = NEG_INF;
	int maxIdx = gid;

	F += gid;
	alphaStatus += gid;
	maxIdx = gid;

#pragma unroll
	for (int k = 0;
		 k < UNROLL_X;
		 k++, F += NUM_THREADS, alphaStatus += NUM_THREADS, gid += NUM_THREADS) {
		int as = *alphaStatus;
		float f;
		if (as > 1                    //if idx == 1 and not LOW
			|| as == -1 || as == -3) {//if idx == -1 and not HIGH
			f = (float) *F;
		} else {
			f = NEG_INF;
		}

		maxFandI(f, maxValue, gid, maxIdx);
	}

	maxs[TID_X] = maxValue;
	idxs[TID_X] = maxIdx;
	__syncthreads();
 
	if (TID_X < WARP_SIZE) {
#pragma unroll
		for (int k = TID_X + WARP_SIZE; k < NUM_THREADS; k += WARP_SIZE) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}
		maxs[TID_X] = maxValue;
		idxs[TID_X] = maxIdx;
	}
	__syncthreads();

	if (TID_X == 0) {
#pragma unroll
		for (int k = 1; k < WARP_SIZE; k++) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}
		maxs[TID_X] = maxValue;
	}
	__syncthreads();

	return maxs[TID_X];
}// End of function SearchForMinGapLocal

// <<<numBlocks, NUM_THREADS>>> 
__global__ void SearchForMinGap_stage1(int * alphaStatus, double * F, float2 * result, int numIt) {
	__shared__ float maxs[NUM_THREADS];
	__shared__ int idxs[NUM_THREADS];

	int gid = numIt * blockDim.x * blockIdx.x + TID_X;
	float maxValue = NEG_INF;
	int maxIdx = gid;

	F += gid;
	alphaStatus += gid;
	maxIdx = gid;
	
	//for (numIt--;numIt>0;numIt--) {
	for (; numIt > 0; numIt -= UNROLL_X) {
#pragma unroll
		for (int k = 0;
			 k < UNROLL_X;
			 k++, F += NUM_THREADS, alphaStatus += NUM_THREADS, gid += NUM_THREADS) {
			int as = *alphaStatus;
			float f;
			if (as > 1                    //if idx == 1 and not LOW
				|| as == -1 || as == -3) {//if idx == -1 and not HIGH
				f = (float) *F;
			} else {
				f = NEG_INF;
			}

			maxFandI(f, maxValue, gid, maxIdx);
		}
	}

	maxs[TID_X] = maxValue;
	idxs[TID_X] = maxIdx;
	__syncthreads();
 
	if (TID_X < WARP_SIZE) {
#pragma unroll
		for (int k = TID_X + WARP_SIZE; k < NUM_THREADS; k += WARP_SIZE) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}
		maxs[TID_X] = maxValue;
		idxs[TID_X] = maxIdx;
	}
	__syncthreads();

	if (TID_X == 0) {
#pragma unroll
		for (int k = 1; k < WARP_SIZE; k++) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}

		//store results as float&int into float2
		float2 out;
		out.x = maxValue;
		out.y = __int_as_float(maxIdx);
		result[blockIdx.x] = out;
	}

} //SearchForMinGap_stage1

__device__ void LocalReduce(float &maxValue, int &maxIdx, float *maxs, int *idxs) {
	maxs[TID_X] = maxValue;
	idxs[TID_X] = maxIdx;
	__syncthreads();
 
	if (TID_X < WARP_SIZE) {
#pragma unroll
		for (int k = TID_X + WARP_SIZE; k < NUM_THREADS; k += WARP_SIZE) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}
		maxs[TID_X] = maxValue;
		idxs[TID_X] = maxIdx;
	}
	__syncthreads();

	if (TID_X == 0) {
#pragma unroll
		for (int k = 1; k < WARP_SIZE; k++) {
			maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
		}
		maxs[TID_X] = maxValue;
		idxs[TID_X] = maxIdx;
	}
	__syncthreads();
	maxValue = maxs[0];
	maxIdx = idxs[0];
} //LocalReduce

__device__ void GlobalReduce(float &maxValue, int &maxIdx, float *maxs, int *idxs,
							 float2 *glob_max, int *_glob_mutex, int glob_sync) {
	const int num = gridDim.x;
	int * glob_mutex_in = _glob_mutex;
	int * glob_mutex_out = _glob_mutex + num;
	volatile int * _glob_mutex_in = glob_mutex_in;
	volatile int * _glob_mutex_out = glob_mutex_out;
	//GLOBAL SYNC:
	//atomic add & atomic max
	float2 loc_max = make_float2(maxValue, __int_as_float(maxIdx));
	if (TID_X == 0) {
		glob_max[BID] = loc_max;
		glob_mutex_in[BID] = glob_sync;
	}
	
	//global sync as a local sync
	if (BID == 0) {
		loc_max.y = -1;
		if (TID_X < num) {
			while (_glob_mutex_in[TID_X] != glob_sync)
				; //wait
			loc_max = glob_max[TID_X];
			maxs[TID_X] = loc_max.x;
			idxs[TID_X] = __float_as_int(loc_max.y);
		}

		__syncthreads(); //all threads done = all block done
		
		if (TID_X == 0) {
			maxValue = loc_max.x;
			maxIdx = __float_as_int(loc_max.y);
	#pragma unroll
			for (int k = 1; k < num; k++) {
				maxFandI(maxs[k], maxValue, idxs[k], maxIdx);
			}
			glob_max[0] = make_float2(maxValue, __int_as_float(maxIdx));
			glob_mutex_out[0] = glob_sync;
		}
	}
	
	if (TID_X == 0) {
		while (_glob_mutex_out[0] != glob_sync)
			;
	}
	__syncthreads();
	loc_max = glob_max[0];
	maxValue = loc_max.x;	
	maxIdx = float_as_int(loc_max.y);

} //GlobalReduce

__device__ void TakeStepLocal (double &a1, double &a2, double y1, double y2, double F1, double F2,
							   double q, double C, double eps, double &da1, double &da2) {
	
	//calculate new alphas
	double H,L;
	if (y1 == y2) {
		H = min(C, a1 + a2);
		L = max(0.0, a1 + a2 - C);
	} else {
		H = min(C, C + a2 - a1);
		L = max(0.0, a2 - a1);
	}
	
	double eta = y1 * y2 * 2 * q - 2.0;
	double a2_new = a2 - y2 * (F1 - F2) / eta;
	double a2_clip = min(max(a2_new, L), H);
	double a1_new = a1 + y1 * y2 * (a2 - a2_clip);

	//calc alpha diff
	da1 = a1_new - a1;
	da2 = a2_clip - a2;

	//correct rouding errors
    double s = -y1 * y2;

	if (a1 + da1 > C - TAU) {
        da2 = da2 + s * (C - a1 - da1);
        da1 = C - a1;
	}
	if (a2 + da2 > C - TAU) {
        da1 = da1 + s * (C - a2 - da2);
        da2 = C - a2;
	}
	if (a1 + da1 < TAU) {
        da2 = da2 - s * (a1 + da1);
        da1 = -a1;
	}
	if (a2 + da2 < TAU) {
        da1 = da1 - s * (a2 + da2);
        da2 = -a2;
	}

	a1 += da1;
	a2 += da2;
	
	//C -= eps;
	//alphaStatus[i1] = (int)y1 * (2*(a1 > eps) + (a1 < C));
	//alphaStatus[i2] = (int)y2 * (2*(a2 > eps) + (a2 < C));
} //TakeStepLocal

__device__ int getAlphaStatus(double a, int y, double C, double eps) {
	return y * (2 * (a > eps) + (a < C));
} //getAlphaStatus

__global__ void SelectIandJ_PT(double *_d_alpha, int *_alphaStatus, double *_d_F,
							   float *Q, size_t Qpitch, const double C, const double eps,
							   int N_, float2 *glob_max1, float2 *glob_max2,
							   int *glob_mutex1, int *glob_mutex2, int *glob_mutex3,
							   int *glob_mutex4, double *_alphaDiff) {
	
	__shared__ float maxs[NUM_THREADS << 1];
	__shared__ int idxs[NUM_THREADS << 1];

	const int gid = blockDim.x*blockIdx.x + threadIdx.x;
	float maxValue;
	int maxIdx;
	volatile double *d_F = _d_F;
	volatile double *d_alpha = _d_alpha;
	volatile int *alphaStatus = _alphaStatus;
	volatile double *alphaDiff = _alphaDiff;
	float F = (float) _d_F[gid];
	int as = _alphaStatus[gid];
	double y = sign(as);
	double alpha = _d_alpha[gid];
	float2 tmp_res = make_float2(0.0F, 0.0F);

	//main loop
	int iter = 0;
	do {
		iter++;

		//select i
		maxIdx=gid;
		float fg = -F;
		maxValue = (as == 1 || as == 3)? fg : NEG_INF; //if idx == 1 and not HIGH
		maxValue = (as < -1)? fg : maxValue; //if idx == -1 and not LOW

		//if (iter == 25 && gid == 623) printf("%d %f %d %e %d\n", TID_X, maxValue, maxIdx, -F, as); //LLLL

		LocalReduce(maxValue, maxIdx, maxs, idxs);
		GlobalReduce(maxValue, maxIdx, maxs, idxs, glob_max1, glob_mutex1, iter);

		///////////////////////////////
		
		float maxValueI = maxValue;
		int maxIdxI = maxIdx;
		maxIdx=gid;
		double y_i = sign(_alphaStatus[maxIdxI]); //load y_i

		float f = F;
		float q = Q[Qpitch * maxIdxI + gid];
		float eta1 = 1;
		float eta2 = 1;
		eta1 = 2 - 2 * q;
		eta2 = 2 + 2 * y_i * q;
		bool b1 = (as > 1);
		bool b2 = (as == -1 || as == -3);
		float eta = max(TAU, b1 * eta1 + b2 * eta2);
		float grad_diff = maxValueI + f;
		float obj = grad_diff * grad_diff / eta;
		maxValue = ((b1 || b2) && (grad_diff > 0))? obj : NEG_INF;


		LocalReduce(maxValue, maxIdx, maxs, idxs);
		//if (threadIdx.x == 0) printf("%d %f %d\n", blockIdx.x, maxValue, maxIdx);
		GlobalReduce(maxValue, maxIdx, maxs, idxs, glob_max2, glob_mutex2, iter);

		//if (BID == 0 && TID_X == 0) {
		//	((float2 *)reduceResult)[0] = make_float2(maxValueI, __int_as_float(maxIdxI));
		//	((float2 *)reduceResult)[1] = make_float2(maxValue, __int_as_float(maxIdx));
		//}

		double daI, daJ;
		if (BID == 0 && TID_X < WARP_SIZE) { //local step by the first warp & the first block
			double aI = d_alpha[maxIdxI];
			double aJ = d_alpha[maxIdx];
			double y_j = sign(alphaStatus[maxIdx]);
			double FI = d_F[maxIdxI];
			double FJ = d_F[maxIdx];
			double q = Q[Qpitch * maxIdxI + maxIdx];
			TakeStepLocal(aI, aJ, y_i, y_j, FI, FJ, q, C, eps, daI, daJ);
			if (TID_X == 0) {
				_alphaDiff[0] = daI;
				_alphaDiff[1] = daJ;
				glob_mutex3[0] = iter;

				// Global alpha and alpha difference updates.
				_d_alpha[maxIdxI] = aI;
				_d_alpha[maxIdx] = aJ;
				_alphaStatus[maxIdxI] = getAlphaStatus(aI, (int) y_i, C, eps);
				_alphaStatus[maxIdx] = getAlphaStatus(aJ, (int) y_j, C, eps);
			}
		}
		//__syncthreads(); //be shure to store alphaDiffs
		//if (BID == 0 && TID_X == 0) {
		//	printf("iter = %d, i = %d, j = %d, daI = %f, daJ = %f\n", iter, maxIdxI, maxIdx, daI, daJ);
		//}

		if (TID_X == 0) {
			volatile int *p = glob_mutex3;
			while (*p != iter)
				; //wait
		}
		__syncthreads();

		//load alphaDiffs and update alphaStatus and F
		daI = alphaDiff[0];
		daJ = alphaDiff[1];

		if (gid == maxIdxI) { // Update alpha_I locally.
			alpha += daI;
			as = getAlphaStatus(alpha, (int) y, C, eps);
		}
		if (gid == maxIdx) { // Update alpha_J locally.
			alpha += daJ;
			as = getAlphaStatus(alpha, (int) y, C, eps);
		}

		if (gid < N_) { // Update vector F.
			float Q1 = Q[maxIdxI * Qpitch + gid];
			float Q2 = Q[maxIdx * Qpitch + gid];
			F += y * ( Q1 * daI + Q2 * daJ );
			_d_F[gid] = F;
		}
		//if (gid == 129) {
		//	printf("F = %e, Q1 = %e, Q2 = %e, daI = %e, daJ = %e, y = %d, GS %d\n", F, Q1, Q2, daI, daJ, y, glob_sync2);
		//}

		//__syncthreads();

		//if (TID_X == 0) glob_mutex4[BID] = iter;

		//if (TID_X < gridDim.x) {
		//	volatile int *p = &(glob_mutex4[TID_X]);
		//	while(*p != iter); //wait
		//}
		//__syncthreads();

		////reread data
		//F = d_F[gid];
		//alpha = d_alpha[gid];
		//as = alphaStatus[gid];


		// Check for stopping condition once per every 64 iterations.
		if ((iter & 0x3F) == 0) {
			// Calculate global maximum through vector F.
			float max_gap = maxValueI + SearchForMinGapLocal(_alphaStatus, _d_F, glob_max1, maxs, idxs);
			
			if (TID_X == 0) {
				//printf("%03d: %f\n", BID, max_gap);
				if (max_gap >= eps) {
					glob_mutex4[BID] = TRUE;
				} else {
					glob_mutex4[BID] = FALSE;
				}
			}
		}

		//an iteration later check stop criterion (omit en extra global sync) - work only for NUM_BLOCK <= WARP_SIZE
		if (iter > 1 && ((iter - 1) & 0x3F) == 0) {
			/*if ((TID_X + BID) == 0) {
				printf("=============\n");
			}*/
			__shared__ int sh_cond;

			if (TID_X == 0) { // Set shared result fo false by first thread.
				sh_cond = FALSE;
			}

			if (TID_X < gridDim.x) { // Load results from all blocks to local variable (per thread)
				if (glob_mutex4[TID_X] == TRUE) { // If TID_X-th block has not converged, set shared result.
					sh_cond = TRUE;
				}
			}
			__syncthreads();
			if (sh_cond == FALSE) { // If non of blocks has not converged (every block has converged).
				--iter;
				break;          // Break the training loop.
			}
		}


	} //main loop - do-while
	while (iter < MAX_ITERS);

	// store number of iterations to global variable.
	if ((BID + TID_X) == 0) {
		glob_mutex1[0] = iter;
	}


} // End of function SelectIandJ_PT

template <unsigned int kernel_type>
__global__ void Q_init(float4 * matrix_f4, int4 * y_i4, int width, int height, size_t pitch,
						float4 * result_f4, size_t pitch_result, float coef0, float degree,
						float gamma) {
	__shared__ float4 sums_f4[(NUM_THREADS_Y_INIT) * (NUM_THREADS_X_INIT << 2 + 1)];
	int i, j;
	unsigned int row_idx = BID * NUM_THREADS_Y_INIT + TID_Y;
	int width_f4 = (width + 3) >> 2;
	int pitch_f4 = pitch >> 2;
	int pitch_f4_2 = pitch >> 1;
	int pitch_f4_3 = pitch_f4_2 + pitch_f4;
	int pitch_result_f4 = pitch_result >> 2;
	float4 sum_0_f4;
	float4 sum_1_f4;
	float4 sum_2_f4;
	float4 sum_3_f4;
	float4 * sums_local_0_f4 = sums_f4 + TID_Y * (NUM_THREADS_X_INIT << 2 + 1);
	float4 * sums_local_1_f4 = sums_local_0_f4 + NUM_THREADS_X_INIT;
	float4 * sums_local_2_f4 = sums_local_1_f4 + NUM_THREADS_X_INIT;
	float4 * sums_local_3_f4 = sums_local_2_f4 + NUM_THREADS_X_INIT;

	// Shift row of matrix
	float4 * matrix_row = matrix_f4 + row_idx * pitch;
	// Shift vector y
	int4 * y_row = y_i4 + row_idx;
	// Shift result matrix.
	result_f4 += row_idx * pitch_result;

	// Do the multiplication
	// every block does the multiplication for NUM_THREADS lines of matrix
	for (i = 0; i < height; i += 4) {
		sum_0_f4 = F4_0;
		sum_1_f4 = F4_0;
		sum_2_f4 = F4_0;
		sum_3_f4 = F4_0;
		for (j = TID_X; j < width_f4; j += NUM_THREADS_X_INIT) {
			float4 tmp_0_x;
			float4 tmp_1_x;
			float4 tmp_2_x;
			float4 tmp_3_x;
			float4 tmp_0_y;
			float4 tmp_1_y;
			float4 tmp_2_y;
			float4 tmp_3_y;
			float4 tmp_0_z;
			float4 tmp_1_z;
			float4 tmp_2_z;
			float4 tmp_3_z;
			float4 tmp_0_w;
			float4 tmp_1_w;
			float4 tmp_2_w;
			float4 tmp_3_w;
			float4 matrix_rj0 = matrix_row[j];
			float4 matrix_rj1 = matrix_row[pitch_f4 + j];
			float4 matrix_rj2 = matrix_row[pitch_f4_2 + j];
			float4 matrix_rj3 = matrix_row[pitch_f4_3 + j];

			float4 matrix_tmp = matrix_f4[i * pitch_f4 + j];
			if (kernel_type == RBF) {
				tmp_0_x = matrix_rj0 - matrix_tmp;
				tmp_1_x = matrix_rj1 - matrix_tmp;
				tmp_2_x = matrix_rj2 - matrix_tmp;
				tmp_3_x = matrix_rj3 - matrix_tmp;
				tmp_0_x *= tmp_0_x;
				tmp_1_x *= tmp_1_x;
				tmp_2_x *= tmp_2_x;
				tmp_3_x *= tmp_3_x;
			} else {
				tmp_0_x = matrix_rj0 * matrix_tmp;
				tmp_1_x = matrix_rj1 * matrix_tmp;
				tmp_2_x = matrix_rj2 * matrix_tmp;
				tmp_3_x = matrix_rj3 * matrix_tmp;
			}

			matrix_tmp = matrix_f4[(i + 1) * pitch_f4 + j];
			if (kernel_type == RBF) {
				tmp_0_y = matrix_rj0 - matrix_tmp;
				tmp_1_y = matrix_rj1 - matrix_tmp;
				tmp_2_y = matrix_rj2 - matrix_tmp;
				tmp_3_y = matrix_rj3 - matrix_tmp;
				tmp_0_y *= tmp_0_y;
				tmp_1_y *= tmp_1_y;
				tmp_2_y *= tmp_2_y;
				tmp_3_y *= tmp_3_y;
			} else {
				tmp_0_y = matrix_rj0 * matrix_tmp;
				tmp_1_y = matrix_rj1 * matrix_tmp;
				tmp_2_y = matrix_rj2 * matrix_tmp;
				tmp_3_y = matrix_rj3 * matrix_tmp;
			}
			
			matrix_tmp = matrix_f4[(i + 2) * pitch_f4 + j];
			if (kernel_type == RBF) {
				tmp_0_z = matrix_rj0 - matrix_tmp;
				tmp_1_z = matrix_rj1 - matrix_tmp;
				tmp_2_z = matrix_rj2 - matrix_tmp;
				tmp_3_z = matrix_rj3 - matrix_tmp;
				tmp_0_z *= tmp_0_z;
				tmp_1_z *= tmp_1_z;
				tmp_2_z *= tmp_2_z;
				tmp_3_z *= tmp_3_z;
			} else {
				tmp_0_z = matrix_rj0 * matrix_tmp;
				tmp_1_z = matrix_rj1 * matrix_tmp;
				tmp_2_z = matrix_rj2 * matrix_tmp;
				tmp_3_z = matrix_rj3 * matrix_tmp;
			}


			matrix_tmp = matrix_f4[(i + 3) * pitch_f4 + j];
			if (kernel_type == RBF) {
				tmp_0_w = matrix_rj0 - matrix_tmp;
				tmp_1_w = matrix_rj1 - matrix_tmp;
				tmp_2_w = matrix_rj2 - matrix_tmp;
				tmp_3_w = matrix_rj3 - matrix_tmp;
				tmp_0_w *= tmp_0_w;
				tmp_1_w *= tmp_1_w;
				tmp_2_w *= tmp_2_w;
				tmp_3_w *= tmp_3_w;
			} else {
				tmp_0_w = matrix_rj0 * matrix_tmp;
				tmp_1_w = matrix_rj1 * matrix_tmp;
				tmp_2_w = matrix_rj2 * matrix_tmp;
				tmp_3_w = matrix_rj3 * matrix_tmp;
			}
			
			sum_0_f4 += make_float4(SUM_FLOAT4_TO_FLOAT(tmp_0_x),
				SUM_FLOAT4_TO_FLOAT(tmp_0_y),
				SUM_FLOAT4_TO_FLOAT(tmp_0_z),
				SUM_FLOAT4_TO_FLOAT(tmp_0_w));
			sum_1_f4 += make_float4(SUM_FLOAT4_TO_FLOAT(tmp_1_x),
				SUM_FLOAT4_TO_FLOAT(tmp_1_y),
				SUM_FLOAT4_TO_FLOAT(tmp_1_z),
				SUM_FLOAT4_TO_FLOAT(tmp_1_w));
			sum_2_f4 += make_float4(SUM_FLOAT4_TO_FLOAT(tmp_2_x),
				SUM_FLOAT4_TO_FLOAT(tmp_2_y),
				SUM_FLOAT4_TO_FLOAT(tmp_2_z),
				SUM_FLOAT4_TO_FLOAT(tmp_2_w));
			sum_3_f4 += make_float4(SUM_FLOAT4_TO_FLOAT(tmp_3_x),
				SUM_FLOAT4_TO_FLOAT(tmp_3_y),
				SUM_FLOAT4_TO_FLOAT(tmp_3_z),
				SUM_FLOAT4_TO_FLOAT(tmp_3_w));
		}

		sums_local_0_f4[TID_X] = sum_0_f4;
		sums_local_1_f4[TID_X] = sum_1_f4;
		sums_local_2_f4[TID_X] = sum_2_f4;
		sums_local_3_f4[TID_X] = sum_3_f4;
		__syncthreads();
		final_quadreduction(sums_local_0_f4, sums_local_1_f4, sums_local_2_f4, sums_local_3_f4, TID_X)
		if (TID_X == 0) {
			int idx = i >> 2;
			switch (kernel_type) {
			case LINEAR:
				sum_0_f4 = *sums_local_0_f4;
				sum_1_f4 = *sums_local_0_f4;
				sum_2_f4 = *sums_local_0_f4;
				sum_3_f4 = *sums_local_0_f4;
				break;
			case POLY:
				sum_0_f4 = powf4(gamma * *sums_local_0_f4 + coef0, degree);
				sum_1_f4 = powf4(gamma * *sums_local_1_f4 + coef0, degree);
				sum_2_f4 = powf4(gamma * *sums_local_2_f4 + coef0, degree);
				sum_3_f4 = powf4(gamma * *sums_local_3_f4 + coef0, degree);
				break;
			case SIGMOID:
				sum_0_f4 = tanhf4(gamma * *sums_local_0_f4 + coef0);
				sum_1_f4 = tanhf4(gamma * *sums_local_1_f4 + coef0);
				sum_2_f4 = tanhf4(gamma * *sums_local_2_f4 + coef0);
				sum_3_f4 = tanhf4(gamma * *sums_local_3_f4 + coef0);
				break;
			case RBF:
				sum_0_f4 = expf4(-gamma * *sums_local_0_f4);
				sum_1_f4 = expf4(-gamma * *sums_local_1_f4);
				sum_2_f4 = expf4(-gamma * *sums_local_2_f4);
				sum_3_f4 = expf4(-gamma * *sums_local_3_f4);
				break;
			}
			result_f4[idx]                       = y_row->x * y_i4[idx] * sum_0_f4;
			result_f4[idx + pitch_result_f4]     = y_row->y * y_i4[idx] * sum_1_f4;
			result_f4[idx + pitch_result_f4 * 2] = y_row->z * y_i4[idx] * sum_2_f4;
			result_f4[idx + pitch_result_f4 * 3] = y_row->w * y_i4[idx] * sum_3_f4;
		}
	}
}

#endif /* __CUDASVM_KERNELS_H */
