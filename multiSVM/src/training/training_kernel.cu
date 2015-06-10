
#ifndef _TRAINING_KERNEL_H_
#define _TRAINING_KERNEL_H_

#include <stdio.h>

/**
 * Set initial values of the binary labels and alphas
 * @param d_ltraindata device pointer to multiclass labels
 * @param d_rdata device pointer to the binary matrix that encodes the output code
 * @param d_ytraindata device pointer to the array with binary labels
 * @param d_atraindata device pointer to the array with the alphas
 * @param d_fdata device pointer to the intermediate values of f
 * @param ntraining number of training samples in the training set
 * @param ntasks number of binary tasks to be solved
 * @param d_active device pointer to the binary array that indicates the status of the task
 */
template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void initializetraining(	int* d_ltraindata,
											int* d_rdata,
											int* d_ytraindata,
											float* d_atraindata,
											float* d_fdata,
											int ntraining,
											int ntasks,
											int* d_active)
{
	const unsigned int j = blockIdx.y;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	const unsigned int gridSize = blockSize*2*gridDim.x;

	const unsigned int bidy= d_active[j];

	while (i < ntraining)
	{
		int label= d_ltraindata[i];
		d_ytraindata[bidy*ntraining + i]= d_rdata[(label-1)*ntasks + bidy];
		d_atraindata[bidy* ntraining + i]=0.0f;
		d_fdata[bidy* ntraining + i]= -1.0* (float)( d_ytraindata[bidy*ntraining + i]);

		if (isNtrainingPow2 || i + blockSize < ntraining)
		{
			label= d_ltraindata[i + blockSize];
			d_ytraindata[bidy*ntraining + i + blockSize]= d_rdata[(label-1)*ntasks + bidy];
			d_atraindata[bidy* ntraining + i + blockSize]=0.0f;
			d_fdata[bidy* ntraining + i + blockSize]= -1.0* (float)( d_ytraindata[bidy*ntraining + i + blockSize]);

		}
		i += gridSize;
	}
	__syncthreads();

}

/**
 * Calculate the new values of the chosen duple of alphas
 * @param d_xtraindata device pointer to the training set
 * @param d_kdata device pointer to the cached rows of the gram matrix
 * @param d_ytraindata device pointer to the array with binary labels
 * @param d_atraindata device pointer to the array with the alphas
 * @param d_anewtraindata device pointer to the new duple of alphas (alpha 1 and alpha 2)
 * @param d_aoldtraindata device pointer to the old duple of alphas (alpha 1 and alpha 2)
 * @param d_fdata device pointer to the intermediate values of f
 * @param d_Iup_global device pointer to the Iup indexes for each binary task
 * @param d_Ilow_global device pointer to the Ilow indexes for each binary task
 * @param d_Iup_cache device pointer to the location of the Iup index in the kernel cache
 * @param d_Ilow_cache device pointer to the location of the Ilow index in the kernel cache
 * @param d_done device pointer containing the status of each binary task
 * @param ntraining number of training samples in the training set
 * @param nfeatures number of features in each of the training samples
 * @param ntasks number of binary tasks to be solved
 * @param d_C device pointer to the regularization parameter for each binary task
 */
__global__ static void calculatealphas(		float* d_xtraindata,
											float* d_kdata,
											int* d_ytraindata,
											float* d_atraindata,
											float* d_anewtraindata,
											float* d_aoldtraindata,
											float* d_fdata,
											int* d_Iup_global,
											int* d_Ilow_global,
											int* d_Iup_cache,
											int* d_Ilow_cache,
											int* d_done,
											int ntraining,
											int nfeatures,
											int ntasks,
											float* d_C)
{

	const unsigned int tid = threadIdx.x;
	const unsigned int bidy = blockIdx.x;
	const float eps= 0.000001;

	//Check if the task has converged
	if(d_done[tid]==0)
	{

		int blockYAlpha=(int) ceil((float)(ntasks)/(float)(TPB));

		if((bidy != blockYAlpha-1) || ((bidy == blockYAlpha-1) && tid < (ntasks - bidy*TPB)))
		{
			//Calculate alpha_2 and alpha _1

			float C= d_C[tid];

			int g_Iup=d_Iup_global[tid];
			int g_Ilow=d_Ilow_global[tid];

			int y_2= d_ytraindata[tid*ntraining + g_Iup];
			int y_1= d_ytraindata[tid*ntraining + g_Ilow];

			float alpha_2_old= d_atraindata[tid*ntraining + g_Iup];
			float alpha_1_old= d_atraindata[tid*ntraining + g_Ilow];

			d_aoldtraindata[tid*2]= alpha_1_old;
			d_aoldtraindata[tid*2 +1]= alpha_2_old;


			float f_2_old= d_fdata[tid*ntraining + g_Iup];
			float f_1_old= d_fdata[tid*ntraining + g_Ilow];

			int s= y_1 * y_2;


			float gamma=0.0f;
			float L=0.0f;
			float H=0.0f;

			if(y_1 == y_2)
			{
				gamma= alpha_1_old + alpha_2_old;
			}
			else
			{
				gamma= alpha_1_old - alpha_2_old;
			}

			if(s==1)
			{
				L= max( 0.0f, gamma -C);
				H= min (C,gamma);
			}
			else
			{
				L= max(0.0f, -gamma);
				H= min(C, C -gamma);
			}


			if(H<=L)
			{
				d_done[tid]=1;

			}


			float K12= d_kdata [d_Ilow_cache[tid]*ntraining + g_Iup];
			float K11= d_kdata [d_Ilow_cache[tid]*ntraining + g_Ilow];
			float K22= d_kdata [d_Iup_cache[tid]*ntraining + g_Iup];

			float nu= 2*K12 - K11 -K22;


			float alpha_2_new=0.0f;
			float alpha_1_new=0.0f;

			if(nu < 0)
			{
				alpha_2_new= alpha_2_old - (y_2*(f_1_old - f_2_old)/nu);
				if(alpha_2_new <L)
				{
					alpha_2_new=L;
				}
				else if (alpha_2_new>H)
				{
					alpha_2_new=H;
				}
			}
			else
			{
				float slope= y_2 *(f_1_old - f_2_old);
				float change= slope * (H-L);
				if(fabs(change)>0.0f)
				{
					if(slope>0.0f)
					{
						alpha_2_new= H;
					}
					else
					{
						alpha_2_new= L;
					}
				}
				else
				{
					alpha_2_new= alpha_2_old;
				}

				if( alpha_2_new > C - eps * C)
				{
					alpha_2_new=C;
				}
				else if (alpha_2_new < eps * C)
				{
					alpha_2_new=0.0f;
				}
			}

			if( fabs( alpha_2_new - alpha_2_old) < eps * ( alpha_2_new + alpha_2_old + eps))
			{
				d_done[tid]=1;
			}

			if(s==1)
			{
				alpha_1_new= gamma - alpha_2_new;
			}
			else
			{
				alpha_1_new= gamma + alpha_2_new;
			}

			if( alpha_1_new > C - eps * C)
			{
				alpha_1_new=C;
			}
			else if (alpha_1_new < eps * C)
			{
				alpha_1_new=0.0f;
			}

			d_anewtraindata[tid*2]= alpha_1_new;
			d_anewtraindata[tid*2 +1]= alpha_2_new;
			//printf("a1 = %.3f a2 = %.3f\n", alpha_1_new, alpha_2_new); //LLLLLLLLLLLLLL

		}

	}
	__syncthreads();
}

/**
 * Calculate the new values of the chosen duple of alphas
 * @param d_xtraindata device pointer to the training set
 * @param d_kdata device pointer to the cached rows of the gram matrix
 * @param d_ytraindata device pointer to the array with binary labels
 * @param d_atraindata device pointer to the array with the alphas
 * @param d_anewtraindata device pointer to the new duple of alphas (alpha 1 and alpha 2)
 * @param d_aoldtraindata device pointer to the old duple of alphas (alpha 1 and alpha 2)
 * @param d_fdata device pointer to the intermediate values of f
 * @param d_Iup_global device pointer to the Iup indexes for each binary task
 * @param d_Ilow_global device pointer to the Ilow indexes for each binary task
 * @param d_Iup_cache device pointer to the location of the Iup index in the kernel cache
 * @param d_Ilow_cache device pointer to the location of the Ilow index in the kernel cache
 * @param d_done device pointer containing the status of each binary task
 * @param ntraining number of training samples in the training set
 * @param nfeatures number of features in each of the training samples
 * @param activeTasks number of non converged tasks
 * @param ntasks number of binary tasks to be solved
 * @param d_C device pointer to the regularization parameter for each binary task
 */
template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void updateparams(		float* d_xtraindata,
											float* d_kdata,
											int* d_ytraindata,
											float* d_atraindata,
											float* d_anewtraindata,
											float* d_aoldtraindata,
											float* d_fdata,
											int* d_Iup_global,
											int* d_Ilow_global,
											int* d_Iup_cache,
											int* d_Ilow_cache,
											int* d_done,
											int* d_active,
											int ntraining,
											int nfeatures,
											int ntasks,
											int activeTasks,
											float* d_C)
{

	const unsigned int tid = threadIdx.x;
	const unsigned int bidx = blockIdx.x;
	unsigned int j = blockIdx.y;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;


	int bidy= d_active[j];

	if(d_done[bidy]==0)
	{
		int g_Iup=d_Iup_global[bidy];
		int g_Ilow=d_Ilow_global[bidy];

		float alpha_1_new= d_anewtraindata[bidy*2];
		float alpha_2_new= d_anewtraindata[bidy*2+1];

		float alpha_1_old= d_aoldtraindata[bidy*2];
		float alpha_2_old= d_aoldtraindata[bidy*2 +1];

		int y_2= d_ytraindata[bidy*ntraining + g_Iup];
		int y_1= d_ytraindata[bidy*ntraining + g_Ilow];

		while (i < ntraining)
		{

			float f_i_old= d_fdata[bidy* ntraining +i];

			float K1i= d_kdata [d_Ilow_cache [bidy] * ntraining +i];
			float K2i= d_kdata [d_Iup_cache [bidy] * ntraining + i];

			float f_i_new= f_i_old + (alpha_1_new - alpha_1_old)*y_1* K1i + (alpha_2_new - alpha_2_old)*y_2* K2i;

			d_fdata[bidy*ntraining +  i]= f_i_new;


			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				f_i_old= d_fdata[bidy* ntraining +i + blockSize];
				K1i= d_kdata [d_Ilow_cache [bidy] * ntraining + i + blockSize];
				K2i= d_kdata [d_Iup_cache [bidy] * ntraining + i + blockSize];

				f_i_new= f_i_old + (alpha_1_new - alpha_1_old)*y_1* K1i + (alpha_2_new - alpha_2_old)*y_2* K2i;

				d_fdata[bidy*ntraining +  i + blockSize]= f_i_new;
			}

			 i += gridSize;
		}

		if(bidx==0 && tid==0)
		{
			d_atraindata[bidy*ntraining + g_Ilow]= d_anewtraindata[bidy*2];
			d_atraindata[bidy*ntraining + g_Iup]= d_anewtraindata[bidy*2+1];
		}


	}
}


#endif // _TRAINING_KERNEL_H_
