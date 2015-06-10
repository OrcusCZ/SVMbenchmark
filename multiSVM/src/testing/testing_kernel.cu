#ifndef _TESTING_KERNEL_H_
#define _TESTING_KERNEL_H_

#include <stdio.h>


/**
 * Generate binary labels of the training data
 * @param d_ltraindata device pointer to multiclass labels
 * @param d_rdata device pointer to the binary matrix that encodes the output code
 * @param d_ytraindata device pointer to the array with binary labels
 * @param ntraining number of training samples in the training set
 * @param ntasks number of binary tasks to be solved
 */

template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void generatebinarylabels(	int* d_ltraindata,
												int* d_rdata,
												int* d_ytraindata,
												int ntraining,
												int ntasks)
{

	const unsigned int bidy = blockIdx.y;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	const unsigned int gridSize = blockSize*2*gridDim.x;

	while (i < ntraining)
	{
		int label= d_ltraindata[i];
		d_ytraindata[bidy*ntraining + i]= d_rdata[(label-1)*ntasks + bidy];

		if (isNtrainingPow2 || i + blockSize < ntraining)
		{
			label= d_ltraindata[i + blockSize];
			d_ytraindata[bidy*ntraining + i + blockSize]= d_rdata[(label-1)*ntasks + bidy];
		}
		i += gridSize;
	}
	__syncthreads();
}

/**
 * Performs the dot product of one array with itself
 * @param d_x device pointer to the input array data
 * @param d_dot device pointer to the output array data
 * @param nfeatures number of features in the samples of the input array
 * @param n length of the input array
 */

template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void makeselfdot(		float* d_x,
										float* d_dot,
										int nfeatures,
										int n)
{
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	const unsigned int gridSize = blockSize*2*gridDim.x;

	while (i < n)
	{
		float val=0;

		for(int j=0; j<nfeatures; j++)
		{
			val+=d_x[j*n + i]*d_x[j*n+i];

		}

		d_dot[i]=val;


		if (isNtrainingPow2 || i + blockSize < n)
		{
			val=0;

			for(int j=0; j<nfeatures; j++)
			{
				val+=d_x[j*n + i+blockSize]*d_x[j*n+i+blockSize];

			}

			d_dot[i+blockSize]=val;

		}

		i += gridSize;
	}
	__syncthreads();
}



/**
 * Performs a reduction to sum the RDF values
 * @param d_ytraindata device pointer to the array with binary labels
 * @param d_atraindata device pointer to the array with the alphas
 * @param d_reduction device pointer to the reduced values
 * @param d_dottestdata device pointer to the dot product of the test data array with itself
 * @param d_dottraindata device pointer to the dot product of the train data array with itself
 * @param d_kdata device pointer to the matrix that stores rows of the gram matrix
 * @param ntraining number of training samples in the training set
 * @param ntestsize number of testing samples considered
 * @param ntestlid number of the partition of the testing set considered
 * @param beta parameter of the Radial Basis Fuction kernel
 */

template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void reductionrbf(			int* d_ytraindata,
												float* d_atraindata,
												float* d_reduction,
												float* d_dottestdata,
												float* d_dottraindata,
												float* d_kdata,
												int ntraining,
												int ntestsize,
												int ntestlid,
												float beta,
												float a,
												float b,
												float d,
												int kernelcode)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bidx = blockIdx.x;
	const unsigned int bidy = blockIdx.y;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	__shared__ float reduction [blockSize];

	//Each thread loads one element
	reduction[tid]= 0;

	while (i < ntraining)
	{
		if(kernelcode==0)
		{
			float val= 2*beta* d_kdata[ntestlid*ntraining +i] - beta* (d_dottraindata[i] + d_dottestdata[ntestlid]);
			val= expf(val);

			reduction[tid] +=	d_atraindata[bidy*ntraining +i]*
								d_ytraindata[bidy*ntraining +i]*
								val;


			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				val= 2*beta* d_kdata[ntestlid*ntraining +(i+blockSize)] - beta* (d_dottraindata[i+blockSize] + d_dottestdata[ntestlid]);
				val= expf(val);

				reduction[tid] +=	d_atraindata[bidy*ntraining +i+blockSize]*
												d_ytraindata[bidy*ntraining +i+blockSize]*
												val;

			}
		}
		else if (kernelcode==1)
		{
			float val=  d_kdata[ntestlid*ntraining +i];

			reduction[tid] +=	d_atraindata[bidy*ntraining +i]*
								d_ytraindata[bidy*ntraining +i]*
								val;


			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				val=  d_kdata[ntestlid*ntraining +(i+blockSize)];

				reduction[tid] +=	d_atraindata[bidy*ntraining +i+blockSize]*
									d_ytraindata[bidy*ntraining +i+blockSize]*
									val;

			}
		}
		else if (kernelcode==2)
		{
			float val= powf(a* d_kdata[ntestlid*ntraining +i]+b,d);

			reduction[tid] +=	d_atraindata[bidy*ntraining +i]*
								d_ytraindata[bidy*ntraining +i]*
								val;


			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				val= powf(a* d_kdata[ntestlid*ntraining +(i+blockSize)]+b,d);

				reduction[tid] +=	d_atraindata[bidy*ntraining +i+blockSize]*
									d_ytraindata[bidy*ntraining +i+blockSize]*
									val;
			}

		}
		else if (kernelcode==3)
		{

			float val= tanhf(a* d_kdata[ntestlid*ntraining +i]+b);

			reduction[tid] +=	d_atraindata[bidy*ntraining +i]*
								d_ytraindata[bidy*ntraining +i]*
								val;


			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				val= tanhf(a* d_kdata[ntestlid*ntraining +(i+blockSize)]+b);

				reduction[tid] +=	d_atraindata[bidy*ntraining +i+blockSize]*
									d_ytraindata[bidy*ntraining +i+blockSize]*
									val;
			}
		}

		i += gridSize;
	 }

	__syncthreads();


	if(blockSize>=512)	{if(tid<256){reduction[tid] += reduction[tid + 256];}__syncthreads();}
	if(blockSize>=256)	{if(tid<128){reduction[tid] += reduction[tid + 128];}__syncthreads();}
	if(blockSize>=128)  {if(tid<64)	{reduction[tid] += reduction[tid + 64];}__syncthreads();}
	if(tid<32){	if(blockSize >= 64)	{reduction[tid] += reduction[tid + 32];}
				if(blockSize >= 32)	{reduction[tid] += reduction[tid + 16];}
				if(blockSize >= 16)	{reduction[tid] += reduction[tid + 8];}
				if(blockSize >= 8)	{reduction[tid] += reduction[tid + 4];}
				if(blockSize >= 4)	{reduction[tid] += reduction[tid + 2];}
				if(blockSize >= 2)	{reduction[tid] += reduction[tid + 1];}	}

	if(tid==0)
	{
		d_reduction[bidy * gridDim.x + bidx]=reduction[tid];
	}

}


#endif // _TESTING_KERNEL_H_
