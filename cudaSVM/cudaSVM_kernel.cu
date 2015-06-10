#include "cudaSVM_kernels.h"
#include "cudaSVM_kernels_scale.h"
#include "cudasvm_train.h"
#include <helper_cuda.h>

int perform_training(float * X, int * labels, double *alphas, int N_, int D, double C,
				  double eps, float coef0, float degree, float gamma, double * rho, int kernel_type) {
	cudaError_t error = cudaErrorNotSupported;
	size_t Q_rows_buf;
	size_t devQPitch;
	size_t devQPitchInFloats;
	size_t devXPitch;
	size_t devXPitchInFloats;
	size_t freeMemory, totalMemory;
	int N;
	int numBlocks;
	int numBlocksWarp;
	int numBlocksReduce;
	int iter = 0;
	dim3 blockGrid(NUM_THREADS_X, NUM_THREADS_Y, 1);
	dim3 blockGridInit(NUM_THREADS_X_INIT, NUM_THREADS_Y_INIT, 1);
	dim3 numBlocksQInit(1, 1, 1);

	double * devAlpha = NULL;
	int * devAlphaStatus = NULL;
	int * devY = NULL;
	double * devAlphaDiff = NULL;
	double * devF = NULL;
	int2 * R = NULL; // Redirection table.
	int cacheSwaps = 0;
	float * devQ = NULL;
	float * devX = NULL;

	float2 *devReduceTmp1;
	float2 *devReduceTmp2;

	// Calculate dimensions and grids of blocks to be run at GPU.
	N = ALIGN_UP(N_, NUM_THREADS);
	numBlocks = N / NUM_THREADS;
	numBlocksWarp = ALIGN_UP(numBlocks, WARP_SIZE);
	numBlocksReduce = (N_ + WARP_SIZE - 1) / WARP_SIZE;

	// Allocations in GPU.
	// 1) Alpha - vector of length N
	checkCudaErrors(cudaMalloc((void **) & devAlpha, N * sizeof(double)));
	checkCudaErrors(cudaMemset(devAlpha, 0, N * sizeof(double)));
	// 2a) Alpha status - vector of length N
	checkCudaErrors(cudaMalloc((void**) & devAlphaStatus, N * sizeof(int)));
	checkCudaErrors(cudaMemset(devAlphaStatus + N_, 0, (N - N_) * sizeof(int)));
	checkCudaErrors(cudaMemcpy(devAlphaStatus, labels, N_ * sizeof(int), cudaMemcpyHostToDevice));
	// 2b) Vector y
	checkCudaErrors(cudaMalloc((void**) & devY, N * sizeof(int)));
	checkCudaErrors(cudaMemcpy(devY, devAlphaStatus, N * sizeof(int), cudaMemcpyDeviceToDevice));
	int w = 1;
	CUDA_DEBUG_FPRINT_BIN(devY, int, w, 4, N_, "Y_data.bin");
	// 3) F - vector of length N
	checkCudaErrors(cudaMalloc((void**) & devF, N * sizeof(double)));
	// 4) X - matrix of dimensions N x D
	checkCudaErrors(cudaMallocPitch((void **) & devX, & devXPitch, D * sizeof(float), N));
	checkCudaErrors(cudaMemset(devX, 0, devXPitch * N));
	checkCudaErrors(cudaMemcpy2D(devX, devXPitch, X, D * sizeof(float), D * sizeof(float), N_,
		cudaMemcpyHostToDevice));
	CUDA_DEBUG_FPRINT_BIN(devX, float, D, devXPitch, N_, "X_data.bin");
	// 5) Alpha difference - 2 floating numbers
	checkCudaErrors(cudaMalloc((void**) & devAlphaDiff, 2 * sizeof(double)));
	// 6) Temporary array number 1 - size depending on number of rows (N) of matrix X
	checkCudaErrors(cudaMalloc((void **) & devReduceTmp1,
		(numBlocksWarp < numBlocksReduce ? numBlocksReduce : numBlocksWarp) * sizeof(float2)));
	// 7) Temporary array number 2
	checkCudaErrors(cudaMalloc((void **) & devReduceTmp2, 5 * sizeof(float2)));
	
	// 8) At last, perform an allocation of matrix Q = try to use as much memory as possible.
	// In the best case allocate whole matrix, otherwise allocate only a cache.
	// We need to be sure other arrays are successfully on the GPU before we fill the memory up.
	// 8a) Try to allocate the whole matrix at once
#if (FORCE_CACHE == OFF)
	error = cudaMallocPitch((void **) & devQ, & devQPitch, N * sizeof(float), N);
#endif

	// 8b) Check whether the allocation of whole matrix succeeded.
	if ((FORCE_CACHE == OFF) && (error == cudaSuccess)) {
		Q_rows_buf = N_;
	} else {
		// 8c) Q is too large and does not fit in GPU memory.
#if (FORCE_CACHE == OFF)
		cudaGetLastError(); // Erase last allocation error.
		//checkCudaErrors(cudaFree(devQ));
		devQ = NULL;
#endif
		// 8d) Allocate redirecting table.
		checkCudaErrors(cudaMalloc((void **) & R, N * sizeof(int2)));
		// R[n].x - index in buffer Q, where is stored the n-th row of actual matrix Q.
		// R[n].y - the last iteration when the n-th row of matrix Q was used.
#if (FORCE_CACHE == OFF) // force using cache
		// Get amount of remaining memory on GPU.
		cudaMemGetInfo(&freeMemory, &totalMemory);

		printf("Free:  %d B\nTotal: %d B\n", freeMemory, totalMemory);
		
		// Calculate how many rows can be fit into the free space.
		Q_rows_buf = (freeMemory / ((size_t) (N * sizeof(float) * 16))) * 15;
		Q_rows_buf = ALIGN_DOWN(Q_rows_buf, WARP_SIZE);

		// Try to allocate memory. If the size is still too big, then reduce the number of rows by 32
		// and try again.
		do {
			error = cudaMallocPitch((void **) & devQ, & devQPitch, N * sizeof(float), Q_rows_buf);
			if (error == cudaSuccess) {
				if ((error = cudaGetLastError()) == cudaSuccess) {
					// Allocation succeeded.
					break;
				}
			}
			Q_rows_buf -= WARP_SIZE;
			//checkCudaErrors(cudaFree(devQ));
			devQ = NULL;
		} while (Q_rows_buf > 1);
#else
		Q_rows_buf = ALIGN_DOWN(N_ >> 2, WARP_SIZE);
		checkCudaErrors(cudaMallocPitch((void **) & devQ, & devQPitch, N * sizeof(float), Q_rows_buf));
#endif
	}
	printf("Caching %d out of %d rows (%d Bytes) of matrix Q.\n", Q_rows_buf, N_,
		   Q_rows_buf * N_ * sizeof(float));

	devQPitchInFloats = devQPitch / sizeof(float);
	devXPitchInFloats = devXPitch / sizeof(float);
	cudaMemset(devQ, 0, devQPitch * Q_rows_buf);
	numBlocksQInit.x = (Q_rows_buf + NUM_THREADS_X_INIT - 1) / (NUM_THREADS_X_INIT);

	// Initially fill the matrix Q
	switch (kernel_type) {
	case POLY:
		Q_init <POLY><<<numBlocksQInit, blockGridInit>>> ((float4 *) devX, (int4 *) devY, D, N_,
			devXPitchInFloats, (float4 *) devQ, devQPitchInFloats, coef0, degree, gamma);
		break;
	case RBF:
		Q_init <RBF><<<numBlocksQInit, blockGridInit>>> ((float4 *) devX, (int4 *) devY, D, N_,
			devXPitchInFloats, (float4 *) devQ, devQPitchInFloats, coef0, degree, gamma);
		break;
	case SIGMOID:
		Q_init <SIGMOID><<<numBlocksQInit, blockGridInit>>> ((float4 *) devX, (int4 *) devY, D, N_,
			devXPitchInFloats, (float4 *) devQ, devQPitchInFloats, coef0, degree, gamma);
		break;
	case LINEAR:
		Q_init <LINEAR><<<numBlocksQInit, blockGridInit>>> ((float4 *) devX, (int4 *) devY, D, N_,
			devXPitchInFloats, (float4 *) devQ, devQPitchInFloats, coef0, degree, gamma);
	}
	CUDA_CHECK_KERNEL

	//CUDA_DEBUG_FPRINT_BIN(devQ, float, N_, devQPitch, Q_rows_buf, "Q_init.bin")

	if ((FORCE_CACHE == OFF) && (N_ == Q_rows_buf)) {
		InitFandStatus <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF, N_);
	} else {
		InitFandStatus_s <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF, R, N_, (int) Q_rows_buf);
	}

	InitReduceBuffer<<<1, numBlocksWarp>>>(devReduceTmp1, (int2 *) devReduceTmp2);

	if ((FORCE_CACHE == OFF) && (N_ == Q_rows_buf)) {
		// Whole matrix Q is in GPU memory.
		while(true) {
			iter++;

			// Find I.
			SelectI_stage1 <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF, devReduceTmp1, 1);
			Reduce_finish <<<1, WARP_SIZE>>> (devReduceTmp1, devReduceTmp2, numBlocksWarp);
		
			// Find J denpending on I.
			SelectJ_stage1 <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF, devQ, devQPitchInFloats,
														 devReduceTmp2, devReduceTmp1, 1);
			Reduce_finish <<<1, WARP_SIZE>>> (devReduceTmp1, devReduceTmp2 + 1, numBlocksWarp);
		

			// Calculate new Alpha values fo I and J.
			TakeStep <<<1, 1>>> ((int *) devReduceTmp2, devAlpha, devAlphaStatus, devF, devQ,
										 devQPitchInFloats, C, eps, devAlphaDiff);

			// Update vector F depending on changes of alpha_I and alpha_J.
			UpdateF <<<numBlocks, NUM_THREADS>>> ((int *) devReduceTmp2, devAlphaDiff, devAlphaStatus, devF,
												  devQ, devQPitchInFloats, N_);

			// Check stopping condition every 16th cycle.
			// If maximul allowen number of iteration were reached, then store current rho
			//   and break the trainig process.
			if((iter & 0x7) == 0 || (iter > MAX_ITERS)) {
				float FdiffMax[5];
				//test stopping condition
				SearchForMinGap_stage1 <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF,
																	 devReduceTmp1, 1);
				Reduce_finish <<<1, WARP_SIZE>>> (devReduceTmp1, devReduceTmp2 + 2, numBlocksWarp);

				// Copy result to RAM.
				checkCudaErrors(cudaMemcpy(FdiffMax, devReduceTmp2, sizeof(float) * 5,
										  cudaMemcpyDeviceToHost));

				// If difference is below threshold, then end the training process.
				if ((FdiffMax[0] + FdiffMax[4] < eps) || (iter > MAX_ITERS)) {
					printf("b_low = %f; b_up = %f\n", FdiffMax[0], FdiffMax[4]);
					*rho = (FdiffMax[4] - FdiffMax[0]) / 2.0;
					break;
				}
			}
		} //while
	} else {
		// Matrix Q is only cache in GPU memory.
		while(true) {
			iter++;
		
			// ---------------------------------
			// -------Search for I phase--------
			// ---------------------------------
			SelectI_stage1 <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF, devReduceTmp1, 1);
			Reduce_finish <<<1, WARP_SIZE>>> (devReduceTmp1, devReduceTmp2, numBlocksWarp);

			// Read from GPU what index to optimize.
			// Note: "((float *) devReduceTmp2) + 1" reads parameter 'y' from the FIRST cell of
			//       float2 array.

			// Check whether I-th row of matrix Q is loaded.
			// And if the row is not presented in cache.
			// Search for most suitable place (the record which has not been used for the longest time).
			UpdateR_s <<<numBlocks, NUM_THREADS>>> (R, (int2 *) devReduceTmp1, (int2 *) devReduceTmp2);
			switch (kernel_type) {
			case LINEAR:
				Reduce_s_finish <LINEAR><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) devReduceTmp2, (int *) (devReduceTmp2 + 3), devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			case POLY:
				Reduce_s_finish <POLY><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) devReduceTmp2, (int *) (devReduceTmp2 + 3), devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			case SIGMOID:
				Reduce_s_finish <SIGMOID><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) devReduceTmp2, (int *) (devReduceTmp2 + 3), devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			case RBF:
				Reduce_s_finish <RBF><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) devReduceTmp2, (int *) (devReduceTmp2 + 3), devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			}
			// ---------------------------------
			// -------Search for J phase--------
			// ---------------------------------
			SelectJ_s_stage1 <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF,
				devQ, devQPitchInFloats, devReduceTmp2, devReduceTmp1, 1);
			Reduce_finish <<<1, WARP_SIZE>>> (devReduceTmp1, devReduceTmp2 + 1, numBlocksWarp);

			// Note: "((float *) (devReduceTmp2 + 1)) + 1" reads parameter 'y' from the SECOND cell
			//       of float2 array.

			// Check whether J-th row of matrix Q is loaded.
			// Row is not presented in cache.
			// Search for most suitable place (the same algorithm as before).
			UpdateR_s <<<numBlocks, NUM_THREADS>>> (R, (int2 *) devReduceTmp1,
				(int2 *) (devReduceTmp2 + 1));
			switch (kernel_type) {
			case LINEAR:
				Reduce_s_finish <LINEAR><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) (devReduceTmp2 + 1), ((int *) (devReduceTmp2 + 3)) + 1, devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			case POLY:
				Reduce_s_finish <POLY><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) (devReduceTmp2 + 1), ((int *) (devReduceTmp2 + 3)) + 1, devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			case SIGMOID:
				Reduce_s_finish <SIGMOID><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) (devReduceTmp2 + 1), ((int *) (devReduceTmp2 + 3)) + 1, devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			case RBF:
				Reduce_s_finish <RBF><<<numBlocksReduce, blockGrid>>> (R, (int2 *) devReduceTmp1,
					(int2 *) (devReduceTmp2 + 1), ((int *) (devReduceTmp2 + 3)) + 1, devQ, devX,
					(int *) (devReduceTmp2 + 4), devY, D, coef0, degree, gamma,
					devQPitchInFloats, devXPitchInFloats, numBlocksWarp, iter);
				break;
			}
			//CUDA_DEBUG_PRINT(devReduceTmp2, int, 10)
			//CUDA_DEBUG_PRINT(R + 13196, float, 6)
			//CUDA_DEBUG_PRINT(devQ + 10078 * devQPitchInFloats, float, 6)
			//CUDA_DEBUG_PRINT(devQ + 10046 * devQPitchInFloats, float, 6)
			// ---------------------------------
			// -------Update alphas phase-------
			// ---------------------------------
			TakeStep_s <<<1, WARP_SIZE>>> ((int *) (devReduceTmp2), devAlpha, devAlphaStatus,
				devF, devQ, devQPitchInFloats, C, eps, devAlphaDiff);
			// ---------------------------------
			// -------Update F phase------------
			// ---------------------------------
			UpdateF_s <<<numBlocks, NUM_THREADS>>> ((int2 *) (devReduceTmp2 + 3), devAlphaDiff,
				devAlphaStatus, devF, devQ, devQPitchInFloats, N_);
			// Check stopping condition every 64th cycle.
			// If maximul allowen number of iteration were reached, then store current rho
			//   and break the trainig process.
			if((iter & 0x3F) == 0 || (iter > MAX_ITERS)) {
				float FdiffMax[5];
				//test stopping condition
				SearchForMinGap_stage1 <<<numBlocks, NUM_THREADS>>> (devAlphaStatus, devF,
																	 devReduceTmp1, 1);

				Reduce_finish <<<1, WARP_SIZE>>> (devReduceTmp1, devReduceTmp2 + 2, numBlocksWarp);

				checkCudaErrors(cudaMemcpy(FdiffMax, devReduceTmp2, sizeof(float) * 5,
										  cudaMemcpyDeviceToHost));

				if ((FdiffMax[0] + FdiffMax[4] < eps) || (iter > MAX_ITERS)) {
					*rho = (FdiffMax[4] - FdiffMax[0]) / 2.0;
					break;
				}
			}
		} //while
	}

	cudaThreadSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Kernel launch failed (%s)\n", cudaGetErrorString(error));
		return error;
	}


	// Load vector alpha from GPU to RAM.
	checkCudaErrors(cudaMemcpy(alphas, devAlpha, sizeof(double) * N_, cudaMemcpyDeviceToHost));
	// Load the number of writes from GPU memory.
	checkCudaErrors(cudaMemcpy(&cacheSwaps, devReduceTmp2 + 4, sizeof(int),
		cudaMemcpyDeviceToHost));
	printf("\nNumber of iterations: %d\nCache swapped %d times.\n\n", iter, cacheSwaps);

	cudaFree(devQ);
	cudaFree(devAlpha);
	cudaFree(devAlphaStatus);
	cudaFree(devF);
	cudaFree(devAlphaDiff);
	cudaFree(devReduceTmp1);
	cudaFree(devReduceTmp2);
	cudaFree(devX);
	cudaFree(devY);

	return error;
}
