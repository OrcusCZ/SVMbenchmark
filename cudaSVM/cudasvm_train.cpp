#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector_types.h>

#include "cudasvm_train.h"
#include "utils.h"
//#include "stopwatch.h"

void cusvm_destroy_model(cusvm_model * &cumodel) {
	cudaError_t err;

	if (!cumodel) {
		return;
	}

	/* GPU memory. */
	if (cumodel->_dvectors != NULL) {
		err = cudaFree(cumodel->_dvectors);

		if (err != cudaSuccess) {
			printf("cudaFree failed to free cumodel->_dvectors [%X]!\nError ignored!\n", err);
			/*exit(err);*/
		}

		cumodel->_dvectors = NULL;
	}
	if (cumodel->_dalphas != NULL) {
		err = cudaFree(cumodel->_dalphas);

		if (err != cudaSuccess) {
			printf("cudaFree failed to free cumodel->_dalphas [%X]!\nError ignored!\n", err);
			/*exit(err);*/
		}

		cumodel->_dalphas = NULL;
	}

	free(cumodel);
	cumodel = NULL;
}

void cusvm_destroy_data(cusvm_prob * &problem) {
	if (!problem) {
		return;
	}

	/* CPU memory. */
	CUDA_SAFE_MFREE_HOST(problem->data);
	CUDA_SAFE_MFREE_HOST(problem->labels);
	MEM_SAFE_FREE(problem->result);
	MEM_SAFE_FREE(problem->b);

	/* GPU memory. */
	CUDA_SAFE_MFREE(problem->_ddata);
	CUDA_SAFE_MFREE(problem->_dlabels);
	CUDA_SAFE_MFREE(problem->_dresult);
	CUDA_SAFE_MFREE(problem->_db);

	/* Variables. */
	problem->width = 0;
	problem->nof_vectors = 0;
	problem->pitch = 0;

	free(problem);
	problem = NULL;
}

void malloc_host(void** vector, size_t size) {
	cudaError_t err;

	err = cudaMallocHost(vector, size);

	if (err != cudaSuccess) {
		printf("Unable to allocate memory on host! Exiting...\n");
		exit(err);
	}
}

void free_host(void* vector) {
	cudaError_t err;

	err = cudaFreeHost(vector);

	if (err != cudaSuccess) {
		printf("Unable to free memory on host.\n");
	}
}

void cusvm_create_data(cusvm_prob *prob, float *test) {
	cudaError_t err;
	//StopWatch w1, w2;

	/* Allocate memory */
	err = cudaMalloc((void **) &(prob->_ddata),
		prob->nof_vectors * prob->width * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc failed to allocate prob->_ddata [%X]!\nExiting...\n", err);
		exit(err);
	}

	err = cudaMalloc((void **) &(prob->_dresult), prob->nof_vectors * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc() failed to allocate memory prob->_dresult [%X]!\nExiting...\n", err);
		exit(err);
	}

	/* Copy data to GPU. */
	err = cudaMemcpy((void *) prob->_ddata, test,
		prob->nof_vectors * prob->width * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy() for prob->_ddata failed [%X]!\nExiting...\n", err);
		exit(err);
	}
}

/**
  * Transfer data from RAM to GPU.
  *
  * @param size Size of memory in Bytes.
  */
void transfer_to_gpu(void ** device_ptr, void * host_ptr, size_t size) {
	CUDA_SAFE_MALLOC(device_ptr, size)
	CUDA_SAFE_MEMCPY(*device_ptr, host_ptr, size, cudaMemcpyHostToDevice)
}

/**
  * Transfer data from RAM to GPU as 2D array.
  *
  * @param pitch The width of data in GPU memory in Bytes.
  * @param width The width of data in GPU memory in Bytes.
  */
void transfer_to_gpu_2d(void ** device_ptr, void * host_ptr, size_t * pitch,
						size_t width, size_t height) {
	CUDA_SAFE_MALLOC_2D(device_ptr, pitch, width, height)
	CUDA_SAFE_MEMCPY_2D(*device_ptr, host_ptr, *pitch, width, height, cudaMemcpyHostToDevice)
}

void cusvm_create_model(cusvm_model * model, float *alphas, float *vectors) {
	size_t host_pitch;
	cudaError_t err;

	/*if ((err = cudaMallocHost((void **) &cumodel->fp_vectors, nof_vectors * vector_len * sizeof(float))) != cudaSuccess) {
		printf("Memory allocation for cumodel->fp_vectors failed [%X]!\nExiting...\n", err);

		exit(err);
	}
	if ((err = cudaMallocHost((void **) &cumodel->fp_alphas, nof_vectors * sizeof(float))) != cudaSuccess) {
		printf("Memory allocation for cumodel->fp_alphas failed [%X]!\nExiting...\n", err);

		exit(err);
	}*/

	/* Allocate GPU memory. */
	if ((err = cudaMallocPitch((void **) &(model->_dvectors), (size_t *) &(model->pitch), 
		sizeof(float) * model->m_vector_len, model->m_nof_vectors)) != cudaSuccess) {
		printf("Memory allocation for model->_dvectors failed [%X]!\nExiting...\n", err);

		exit(err);
	}
	if ((err = cudaMalloc((void **) &(model->_dalphas), model->m_nof_vectors * sizeof(float))) != cudaSuccess) {
		printf("Memory allocation for model->_dalphas failed [%X]!\nExiting...\n", err);

		exit(err);
	}

	/* Copy data to GPU. */
	host_pitch = model->m_vector_len * sizeof(float);
	err = cudaMemcpy2D(model->_dvectors, model->pitch, vectors, host_pitch,
		host_pitch, model->m_nof_vectors, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy2D() for model->_dvectors failed [%X]!\nExiting...\n", err);
		exit(err);
	}
	
	err = cudaMemcpy(model->_dalphas, alphas,
		model->m_nof_vectors * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy() for model->_dalphas failed [%X]!\nExiting...\n", err);
		exit(err);
	}
}

int initiate_training(struct cusvm_prob * prob) {
	unsigned int i, len = prob->nof_vectors;
	float * tmp_array = NULL;

	/* Initiate alphas to zero. */
	MEM_SAFE_ALLOC(tmp_array, float, len)
	for (i = 0; i < len; i++) {
		tmp_array[i] = 0.0F;
	}
	CUDA_SAFE_MALLOC((void **) & (prob->_dresult), len * sizeof(float))
	CUDA_SAFE_MEMCPY((void *) prob->_dresult, tmp_array, len * sizeof(float), cudaMemcpyHostToDevice)

	return SUCCESS;
}

int cudasvm_train(struct cusvm_prob * prob, struct svm_params * params) {
	int ret_val = FAILURE;

	/* Allocate output array. */
	MEM_SAFE_ALLOC(prob->result, double, prob->nof_vectors)

	ret_val = perform_training(prob->data, prob->labels, prob->result, prob->nof_vectors, prob->width,
		params->C, params->eps, (float) params->coef0, (float) params->degree, (float) params->gamma,
		& (params->rho), params->kernel_type);

	return ret_val;
}
