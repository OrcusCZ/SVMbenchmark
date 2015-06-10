#ifndef __CUDASVM_KERNELS_SCALE_H
#define __CUDASVM_KERNELS_SCALE_H

#include "cudaSVM_kernels.h"

//<<<numBlocks, NUM_THREADS>>> 
//alphaStatus[gid] is set to y(i) at the begining 
//goal si to set F to -y and alphaStatus ti y * (1,2,3) = LOW, HIGH, FREE
__global__ void InitFandStatus_s(int * alphaStatus, double * F, int2 * R, int N, int row_cached) {
	int gid = BID * blockDim.x + TID_X;
	if (gid < N) {
		int y = alphaStatus[gid];
		F[gid] = (double) -y;
	} else {
		F[gid] = 0.0;
		alphaStatus[gid] = 0;
	}
	if (gid < row_cached) {
		R[gid] = make_int2(gid, 0);
	} else {
		R[gid] = make_int2(-1, -1);
	}
} //InitFandStatus


__global__ void UpdateR_s (int2 * R, int2 * result, int2 * extreme) {
	__shared__ int mins[NUM_THREADS];
	__shared__ int idxs[NUM_THREADS];

	int gid = blockDim.x * BID + TID_X;
	int minValue;
	int minIdx;

	if (TID_X == 0) {
		int extremeIdx = extreme->y;
		int2 cachePos = R[extremeIdx];
		if (cachePos.y == -1) {
			*mins = -1;
		} else {
			*mins = cachePos.x;
		}
	}
	__syncthreads();

	if (*mins < 0) {
		minValue = INT_MAX;
		minIdx = gid;
#pragma unroll
		for (int k = 0; k < UNROLL_X; k++, gid += NUM_THREADS) {
			minVandI(R[gid].y, minValue, gid, minIdx);
		}

		mins[TID_X] = minValue;
		idxs[TID_X] = minIdx;
		__syncthreads();
	 
		if (TID_X < WARP_SIZE) {
#pragma unroll
			for (int k = TID_X + WARP_SIZE; k < NUM_THREADS; k += WARP_SIZE) {
				minVandI(mins[k], minValue, idxs[k], minIdx);
			}
			mins[TID_X] = minValue;
			idxs[TID_X] = minIdx;
		}
		__syncthreads();

		if (TID_X == 0) {
#pragma unroll
			for (int k = 1; k < WARP_SIZE; k++) {
				minVandI(mins[k], minValue, idxs[k], minIdx);
			}
			//store results as float&int to int2
			result[BID] = make_int2(minValue, minIdx);
		}
	} else {
		if (TID_X == 0) {
			result[BID] = make_int2(-1, -1);
		}
	}
}

template <unsigned int kernel_type>
__global__ void Reduce_s_finish(int2 * R, int2 * y, int2 * rawResult, int * result, float * Q,
								float * X, int * cacheSwaps, int * classY, int width, float coef0,
								float degree, float gamma, size_t pitch, size_t pitchX, int numBlocks,
								int iter) {
	__shared__ float4 sums[NUM_THREADS_Y * (NUM_THREADS_X + 1)];
	
	int k;
	int2 i2;
	int tid = TID;
	int minValue;
	int minIdx;
	// Get index of extreme.
	int index = rawResult->y;
	int * mins = (int *) sums;
	int * idxs = mins + NUM_THREADS;

	if (tid == 0) {
		*mins = y->x;
	}
	__syncthreads();

	if (*mins > -1) {
		if (numBlocks < NUM_THREADS_REDUCE && tid >= numBlocks) {
			minValue = INT_MAX;
			minIdx = -1;
		} else {
			minValue = y[tid].x;
			minIdx = y[tid].y;
		}

		for (k = tid + NUM_THREADS_REDUCE; k < numBlocks; k += NUM_THREADS_REDUCE) {
			i2 = y[k];
			minVandI(i2.x, minValue, i2.y, minIdx);
		}

		mins[tid] = minValue;
		idxs[tid] = minIdx;
		__syncthreads();

		if (tid == 0) {
			int too_old_idx;
#pragma unroll
			for (k = 1; k < NUM_THREADS_REDUCE; k++) {
				minVandI(mins[k], minValue, idxs[k], minIdx);
			}
			// Current meanings of variables:
			// minIdx - index in R; corresponding index in cache (R[minIdx].x)
			//          will be rewritten.
			too_old_idx = R[minIdx].x;

			if (BID == 0) {
				// Store result to global memory and update redirection table accordingly
				i2 = make_int2(too_old_idx, iter);

				// Create new record in redirection table.
				R[index] = i2;

				// Store new index in cache.
				*result = i2.x;

				// Increment the number of writes to cache.
				(*cacheSwaps)++;

				// Erase information about row, which is going to be rewriten, in table R.
				R[minIdx].y = -1;
			}
			// Store index of value which should be rewritten in cache.
			*mins = too_old_idx;
		}
		__syncthreads();
		// Copy the row from full matrix to cache
		float4 * Q_f4 = (float4 *) (Q + (*mins) * pitch); // Row in cache.
		//float4 * Q4_full = (float4 *) (Q_full + index * pitch); // Row in original matrix.
		unsigned int row_idx = BID * NUM_THREADS_Y + TID_Y;
		// The not filled part of matrix X is filled with zeros, so we don't need to be worry about
		// multiplying zeros there.
		int width_f4 = (width + 3) >> 2;
		int pitchX_f4 = pitchX >> 2;
		int pitchX_f4_2 = pitchX >> 1;
		int pitchX_f4_3 = pitchX_f4_2 + pitchX_f4;
		float4 sum_f4 = F4_0;
		int4 classY_i4 = *(((int4 *) classY) + row_idx);
		float4 * sums_local_f4 = sums + TID_Y * (NUM_THREADS_X + 1);
		float4 * matrix_f4 = ((float4 *) X) + row_idx * pitchX;
		float4 * vector = (float4 *) (X + index * pitchX); // Row in original matrix.
		
		// Calculate part of the row of matrix Q - RBF variant
		// Do the multiplication
		// every block does the multiplication for NUM_THREADS_Y lines of matrix
		for (k = TID_X; k < width_f4; k += NUM_THREADS_X) {
			float4 tmp_x;
			float4 tmp_y;
			float4 tmp_z;
			float4 tmp_w;
			float4 vector_tmp = vector[k];
			
			if (kernel_type == RBF) {
				tmp_x = matrix_f4[k] - vector_tmp;
				tmp_y = matrix_f4[pitchX_f4 + k] - vector_tmp;
				tmp_z = matrix_f4[pitchX_f4_2 + k] - vector_tmp;
				tmp_w = matrix_f4[pitchX_f4_3 + k] - vector_tmp;
				tmp_x *= tmp_x;
				tmp_y *= tmp_y;
				tmp_z *= tmp_z;
				tmp_w *= tmp_w;
			} else {
				tmp_x = matrix_f4[k] * vector_tmp;
				tmp_y = matrix_f4[pitchX_f4 + k] * vector_tmp;
				tmp_z = matrix_f4[pitchX_f4_2 + k] * vector_tmp;
				tmp_w = matrix_f4[pitchX_f4_3 + k] * vector_tmp;
			}


			sum_f4 += make_float4(SUM_FLOAT4_TO_FLOAT(tmp_x),
				SUM_FLOAT4_TO_FLOAT(tmp_y),
				SUM_FLOAT4_TO_FLOAT(tmp_z),
				SUM_FLOAT4_TO_FLOAT(tmp_w));
		}
		sums_local_f4[TID_X] = sum_f4;
		__syncthreads();

		final_reduction_8(sums_local_f4, TID_X)
		if (TID_X == 0) {
			switch (kernel_type) {
			case LINEAR:
				sum_f4 = *sums_local_f4;
				break;
			case POLY:
				sum_f4 = powf4(gamma * (*sums_local_f4) + coef0, degree);
				break;
			case SIGMOID:
				sum_f4 = tanhf4(gamma * (*sums_local_f4) + coef0);
				break;
			case RBF:
				sum_f4 = expf4(-gamma * (*sums_local_f4));
				break;
			}
			Q_f4[row_idx] = classY[index] * classY_i4 * sum_f4;
		}
	} else {
		if (tid == 0) {
			// Update iteration when this row in cache was used.
			R[index].y = iter;

			*result = R[index].x;
		}
	}
}

__global__ void SelectJ_s_stage1 (int * alphaStatus, double * F, float * Q, 
							  size_t pitch, float2 * maxI_, float2 * result, int numIt) {
	
	__shared__ float maxs[NUM_THREADS];
	__shared__ int idxs[NUM_THREADS];

	int k;
	int gid = numIt * blockDim.x * BID + TID_X;
	float maxValue = NEG_INF;
	int maxIdx = gid;
	float2 maxI = * maxI_;
	float y_i = sign(alphaStatus[__float_as_int(maxI.y)]); //load y_i

	// Shift matrix Q
	Q += __float_as_int(maxI_[3].x) * pitch;

	for (; numIt > 0; numIt -= UNROLL_X) {
#pragma unroll
		for (k = 0; k < UNROLL_X; k++, gid += NUM_THREADS) {
			int as = alphaStatus[gid];
			float f = (float) F[gid];
			float q = Q[gid];
			float eta1 = 1;
			float eta2 = 1;
			eta1 = 2 - 2 * q;
			eta2 = 2 + 2 * y_i * q;
			bool b1 = (as > 1);
			bool b2 = (as == -1 || as == -3);
			float eta = max(TAU, b1 * eta1 + b2 * eta2);
			float grad_diff = maxI.x + f;
			float obj = grad_diff * grad_diff / eta;

			if ((b1 || b2) && (grad_diff > 0)) {
				maxFandI(obj, maxValue, gid, maxIdx);
			}
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

//"single thread" <<<1, 32>>> 
__global__ void TakeStep_s (int * IandJ, double * alpha, int * alphaStatus, double * F,
						  float * Q, size_t pitch, double C, double eps, double * results) {
	int i1 = IandJ[1];
	int i2 = IandJ[3];
	int i1_cache = IandJ[6];
	
	//load stuff:
	double a1 = alpha[i1];
	double a2 = alpha[i2];
	double y1 = sign(alphaStatus[i1]);
	double y2 = sign(alphaStatus[i2]);
	double F1 = F[i1];
	double F2 = F[i2];
	double q = Q[i1_cache * pitch + i2];
	
	//calculate new alphas
	double H, L;
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
__global__ void UpdateF_s (int2 * IandJ, double * AlphaDiff, int * AlphaStatus, double * F,
						 float * Q, size_t pitch, int height) {
	int gid = blockDim.x * BID + TID_X;
	
	if (gid < height) {
		int2 i = *IandJ;
		int i1_cache = i.x;
		int i2_cache = i.y;

		double Fi = F[gid];
		double y = sign(AlphaStatus[gid]);
		double da1 = AlphaDiff[0];
		double da2 = AlphaDiff[1];
		float Q1 = Q[i1_cache * pitch + gid];
		float Q2 = Q[i2_cache * pitch + gid];

		F[gid] = Fi + y * (Q1 * da1 + Q2 * da2);
	}

} //UpdateF

// <<<numBlocks, NUM_THREADS>>> 
__global__ void SearchForMinGap_stage1_s (int * alphaStatus, double * F, float2 * result, int numIt) {
	__shared__ float maxs[NUM_THREADS];
	__shared__ int idxs[NUM_THREADS];

	int gid = numIt * blockDim.x * BID + TID_X;
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
		result[BID] = out;
	}

} //SearchForMinGap_stage1

#endif  /* __CUDASVM_KERNELS_SCALE_H */
