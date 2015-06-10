#ifndef _SHARED_KERNEL_H_
#define _SHARED_KERNEL_H_

#include <stdio.h>

#define TPB 128
#define LOADX 32
#define LOADY 32





/**
 * Extract the vector to be computed from the training set
 * @param d_xdata input training set
 * @param d_kernelrow will store the extracted point
 * @param Irow index of the vector of the training set that is considered
 * @param ntraining number of samples in the training set
 * @param nfeatures number of features in each training sample
 */

template <unsigned int blockSize, bool isNtrainingPow2>
__global__  void ExtractKernelRow( 			float* d_xdata,
											float* d_kernelrow,
											int Irow,
											int ntraining,
											int nfeatures)
{

	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;


	while (i < nfeatures)
	{
		d_kernelrow[i]= d_xdata[ Irow + (i)* ntraining];

		if (isNtrainingPow2 || i + blockSize < ntraining)
		{

			d_kernelrow[i + blockSize]= d_xdata[ Irow + (i + blockSize)* ntraining];
		}

		i += gridSize;
	}

	__syncthreads();

}

/**
 * Set the result of the kernel evaluation in the cache matrix
 * @param d_kdata cache that will keep the result
 * @param d_dottraindata product of the input sample with itself
 * @param d_kernelrow input vector to be considered
 * @param Irow index of the d_dotraindata to be considered
 * @param Icache index of the row in the cache that the result will occupy
 * @param ntraining number of samples in the training set
 * @param beta parameter for the RBF kernel
 * @param a if using polynomial or sigmoid kernel the value of a x_i x_j
 * @param b if using polynomial or sigmoid kernel the value of b
 * @param d if using polynomial kernel
 * @param kernelcode type of kernel to compute
 */

template <unsigned int blockSize, bool isNtrainingPow2>
__global__  void SetKernelDot(	 			float* d_kdata,
											float* d_dottraindata,
											float* d_kernelrow,
											int Irow,
											int Icache,
											int ntraining,
											float beta,
											float a,
											float b,
											float d,
											int kernelcode)
{

	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;


	while (i < ntraining)
	{
		if(kernelcode==0)
		{
			float val= 2* beta* d_kernelrow [i] - beta* (d_dottraindata[i] + d_dottraindata[Irow]);
			d_kdata[Icache*ntraining + i] = expf(val);

			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				val= 2* beta* d_kernelrow [i + blockSize] - beta* (d_dottraindata[i + blockSize] + d_dottraindata[Irow]);
				d_kdata[Icache*ntraining + i + blockSize] = expf(val);
			}
		}
		else if (kernelcode==1)
		{
			d_kdata[Icache*ntraining + i]= d_kernelrow [i];

			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				d_kdata[Icache*ntraining + i + blockSize] = d_kernelrow [i + blockSize];
			}

		}
		else if (kernelcode==2)
		{
			d_kdata[Icache*ntraining + i]= powf(a*d_kernelrow [i]+b,d);

			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				d_kdata[Icache*ntraining + i + blockSize] = powf(a*d_kernelrow [i+blockSize]+b,d);
			}
		}
		else if (kernelcode==3)
		{
			d_kdata[Icache*ntraining + i]= tanhf(a*d_kernelrow [i]+b);

			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				d_kdata[Icache*ntraining + i + blockSize] = tanhf(a*d_kernelrow [i+blockSize]+b);
			}
		}

		i += gridSize;
	}

	__syncthreads();


}


#endif // _SHARED_KERNEL_H_
