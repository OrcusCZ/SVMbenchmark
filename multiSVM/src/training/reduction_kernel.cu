	
#ifndef _REDUCTION_KERNEL_H_
#define _REDUCTION_KERNEL_H_

#include <stdio.h>


/**
 * Performs an optimized local reduction step to find Iup, Ilow, Bup and Blow
 * @param d_ytraindata device pointer to the array of binary labels
 * @param d_atraindata device pointer to the array of alphas
 * @param d_fdata device pointer to the array of fs
 * @param d_bup device pointer to the local bup values
 * @param d_blow device pointer to the local blow values
 * @param d_Iup device pointer to the local Iup values
 * @param d_Ilow device pointer to the local Ilow values
 * @param d_done_device pointer to the array with the status of each binary task
 * @param d_active device pointer to the array with active binary tasks
 * @param ntraining number of training samples in the training set
 * @param ntasks number of binary tasks to be solved
 * @param activeTasks number of active tasks
 * @param d_C device pointer to the array of regularization parameters
 */
template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void reduction(			int* d_ytraindata,
											float* d_atraindata,
											float* d_fdata,
											float* d_bup,
											float* d_blow,
											int* d_Iup,
											int* d_Ilow,
											int* d_done,
											int* d_active,
											int ntraining,
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

	if(d_done[bidy]== 0)
	{
		float C= d_C[bidy];

		__shared__ float minreduction [blockSize];
		__shared__ float maxreduction [blockSize];
		__shared__ int minreductionid [blockSize];
		__shared__ int maxreductionid [blockSize];

		////Each thread loads one element
		//minreduction[tid]= (float)FLT_MAX;
		//maxreduction[tid]= -1.0* (float)FLT_MAX;

		//minreductionid[tid]= i;
		//maxreductionid[tid]= i;

		//while (i < ntraining)
		//{

		//		float alpha_i;
		//		int y_i= d_ytraindata[bidy* ntraining + i];
		//		float minval=(float)FLT_MAX;
		//		float maxval= -1.0* (float)FLT_MAX;
		//		float f_i= d_fdata[bidy* ntraining +i];


		//		if(y_i !=0)
		//		{
		//			alpha_i= d_atraindata[bidy* ntraining + i];

		//			if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==0) || (y_i== -1 && alpha_i==C) )
		//			{
		//				minval=f_i;
		//			}

		//			if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==C) || (y_i== -1 && alpha_i==0))
		//			{
		//				maxval= f_i;
		//			}


		//			if(minreduction[tid] > minval)
		//			{
		//				minreduction[tid]= minval;
		//				minreductionid[tid]= i;
		//			}

		//			if(maxreduction[tid] < maxval)
		//			{
		//				maxreduction[tid]= maxval;
		//				maxreductionid[tid]= i;
		//			}

		//		}



		//		if (isNtrainingPow2 || i + blockSize < ntraining)
		//		{
		//			y_i= d_ytraindata[bidy* ntraining + i + blockSize];
		//			minval=(float)FLT_MAX;
		//			maxval= -1.0* (float)FLT_MAX;
		//			f_i=d_fdata[bidy* ntraining +i + blockSize];


		//			if(y_i != 0)
		//			{
		//				alpha_i= d_atraindata[bidy* ntraining + i + blockSize];

		//				if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==0) || (y_i== -1 && alpha_i==C) )
		//				{
		//					minval=f_i;


		//				}

		//				if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==C) || (y_i== -1 && alpha_i==0))
		//				{
		//					maxval= f_i;


		//				}


		//				if(minreduction[tid] > minval)
		//				{
		//					minreduction[tid]= minval;
		//					minreductionid[tid]= i + blockSize;
		//				}

		//				if(maxreduction[tid] < maxval)
		//				{
		//					maxreduction[tid]= maxval;
		//					maxreductionid[tid]= i + blockSize;
		//				}
		//			}

		//		}

		//	 i += gridSize;
		// }

		//Each thread loads one element
		float mn = (float)FLT_MAX;
		float mx = -1.0* (float)FLT_MAX;

		int mnid = i;
		int mxid = i;

		while (i < ntraining)
		{

				float alpha_i;
				int y_i= d_ytraindata[bidy* ntraining + i];
				float minval=(float)FLT_MAX;
				float maxval= -1.0* (float)FLT_MAX;
				float f_i= d_fdata[bidy* ntraining +i];


				if(y_i !=0)
				{
					alpha_i= d_atraindata[bidy* ntraining + i];

					if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==0) || (y_i== -1 && alpha_i==C) )
					{
						minval=f_i;
					}

					if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==C) || (y_i== -1 && alpha_i==0))
					{
						maxval= f_i;
					}


					if(mn == minval) mnid = min(mnid, i);
					if(mn > minval)
					{
						mn = minval;
						mnid = i;
					}
					
					if(mx == maxval) mxid = min(mxid, i);
					if(mx < maxval)
					{
						mx = maxval;
						mxid = i;
					}

				}



				if (isNtrainingPow2 || i + blockSize < ntraining)
				{
					y_i= d_ytraindata[bidy* ntraining + i + blockSize];
					minval=(float)FLT_MAX;
					maxval= -1.0* (float)FLT_MAX;
					f_i=d_fdata[bidy* ntraining +i + blockSize];


					if(y_i != 0)
					{
						alpha_i= d_atraindata[bidy* ntraining + i + blockSize];

						if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==0) || (y_i== -1 && alpha_i==C) )
						{
							minval=f_i;


						}

						if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==C) || (y_i== -1 && alpha_i==0))
						{
							maxval= f_i;


						}
						//if(i < 2) printf("i = %d (%d), mx=%.1f (%d) mn=%.1f (%d) f=%.1f minval=%e minval=%e\n", i, y_i, mx, mxid, mn, mnid, f_i, minval, maxval); //LLLLLLLLLLLLLLLLLLLLLL

						if(mn == minval) mnid = min(mnid, i+blockSize);
						if(mn > minval)
						{
							mn = minval;
							mnid = i + blockSize;
						}

						if(mx == maxval) mxid = min(mxid, i+blockSize);
						if(mx < maxval)
						{
							mx = maxval;
							mxid = i + blockSize;
						}
					}

				}

			 i += gridSize;
		 }
		minreduction[tid] = mn;
		minreductionid[tid] = mnid;
		maxreduction[tid] = mx;
		maxreductionid[tid] = mxid;

		__syncthreads();


		if(blockSize>=512)
		{
			if(tid<256)
			{
				if(minreduction[tid] == minreduction[tid+256]) minreductionid[tid] = min(minreductionid[tid], minreductionid[tid +256]);
				if(minreduction[tid] > minreduction[tid+256])
				{
					minreduction[tid]= minreduction[tid +256];
					minreductionid[tid]= minreductionid[tid +256];
				}

				if(maxreduction[tid] == maxreduction[tid+256]) maxreductionid[tid] = min(maxreductionid[tid], maxreductionid[tid +256]);
				if(maxreduction[tid] < maxreduction[tid+256])
				{
					maxreduction[tid]= maxreduction[tid +256];
					maxreductionid[tid]= maxreductionid[tid +256];
				}
			}
			__syncthreads();
		}

		if(blockSize>=256)
		{
			if(tid<128)
			{
				if(minreduction[tid] == minreduction[tid+128]) minreductionid[tid] = min(minreductionid[tid], minreductionid[tid +128]);
				if(minreduction[tid] > minreduction[tid+128])
				{
					minreduction[tid]= minreduction[tid +128];
					minreductionid[tid]= minreductionid[tid +128];
				}

				if(maxreduction[tid] == maxreduction[tid+128]) maxreductionid[tid] = min(maxreductionid[tid], maxreductionid[tid +128]);
				if(maxreduction[tid] < maxreduction[tid+128])
				{
					maxreduction[tid]= maxreduction[tid +128];
					maxreductionid[tid]= maxreductionid[tid +128];
				}
			}
			__syncthreads();
		}

		if(blockSize>=128)
		{
			if(tid<64)
			{
				if(minreduction[tid] == minreduction[tid+64]) minreductionid[tid] = min(minreductionid[tid], minreductionid[tid +64]);
				if(minreduction[tid] > minreduction[tid+64])
				{
					minreduction[tid]= minreduction[tid +64];
					minreductionid[tid]= minreductionid[tid +64];
				}

				if(maxreduction[tid] == maxreduction[tid+64]) maxreductionid[tid] = min(maxreductionid[tid], maxreductionid[tid +64]);
				if(maxreduction[tid] < maxreduction[tid+64])
				{
					maxreduction[tid]= maxreduction[tid +64];
					maxreductionid[tid]= maxreductionid[tid +64];
				}
			}
			__syncthreads();
		}

		if(tid<32)
		{
			volatile float *_minreduction = minreduction;
			volatile float *_maxreduction = maxreduction;
			volatile int *_minreductionid = minreductionid;
			volatile int *_maxreductionid = maxreductionid;

			//32
			if(blockSize >= 64)
			{
				if(_minreduction[tid] == _minreduction[tid+32]) _minreductionid[tid] = min(_minreductionid[tid], _minreductionid[tid +32]);
				if(_minreduction[tid] > _minreduction[tid+32])
				{
					_minreduction[tid]= _minreduction[tid +32];
					_minreductionid[tid]= _minreductionid[tid +32];
				}

				if(_maxreduction[tid] == _maxreduction[tid+32]) _maxreductionid[tid] = min(_maxreductionid[tid], _maxreductionid[tid +32]);
				if(_maxreduction[tid] < _maxreduction[tid+32])
				{
					_maxreduction[tid]= _maxreduction[tid +32];
					_maxreductionid[tid]= _maxreductionid[tid +32];
				}
			}
			//16
			if(blockSize >= 32)
			{
				if(_minreduction[tid] == _minreduction[tid+16]) _minreductionid[tid] = min(_minreductionid[tid], _minreductionid[tid +16]);
				if(_minreduction[tid] > _minreduction[tid+16])
				{
					_minreduction[tid]= _minreduction[tid +16];
					_minreductionid[tid]= _minreductionid[tid +16];
				}

				if(_maxreduction[tid] == _maxreduction[tid+16]) _maxreductionid[tid] = min(_maxreductionid[tid], _maxreductionid[tid +16]);
				if(_maxreduction[tid] < _maxreduction[tid+16])
				{
					_maxreduction[tid]= _maxreduction[tid +16];
					_maxreductionid[tid]= _maxreductionid[tid +16];
				}
			}
			//8
			if(blockSize >= 16)
			{
				if(_minreduction[tid] == _minreduction[tid+8]) _minreductionid[tid] = min(_minreductionid[tid], _minreductionid[tid +8]);
				if(_minreduction[tid] > _minreduction[tid+8])
				{
					_minreduction[tid]= _minreduction[tid +8];
					_minreductionid[tid]= _minreductionid[tid +8];
				}

				if(_maxreduction[tid] == _maxreduction[tid+8]) _maxreductionid[tid] = min(_maxreductionid[tid], _maxreductionid[tid +8]);
				if(_maxreduction[tid] < _maxreduction[tid+8])
				{
					_maxreduction[tid]= _maxreduction[tid +8];
					_maxreductionid[tid]= _maxreductionid[tid +8];
				}
			}
			//4
			if(blockSize >= 8)
			{
				if(_minreduction[tid] == _minreduction[tid+4]) _minreductionid[tid] = min(_minreductionid[tid], _minreductionid[tid +4]);
				if(_minreduction[tid] > _minreduction[tid+4])
				{
					_minreduction[tid]= _minreduction[tid +4];
					_minreductionid[tid]= _minreductionid[tid +4];
				}

				if(_maxreduction[tid] == _maxreduction[tid+4]) _maxreductionid[tid] = min(_maxreductionid[tid], _maxreductionid[tid +4]);
				if(_maxreduction[tid] < _maxreduction[tid+4])
				{
					_maxreduction[tid]= _maxreduction[tid +4];
					_maxreductionid[tid]= _maxreductionid[tid +4];
				}
			}
			//2
			if(blockSize >= 4)
			{
				if(_minreduction[tid] == _minreduction[tid+2]) _minreductionid[tid] = min(_minreductionid[tid], _minreductionid[tid +2]);
				if(_minreduction[tid] > _minreduction[tid+2])
				{
					_minreduction[tid]= _minreduction[tid +2];
					_minreductionid[tid]= _minreductionid[tid +2];
				}

				if(_maxreduction[tid] == _maxreduction[tid+2]) _maxreductionid[tid] = min(_maxreductionid[tid], _maxreductionid[tid +2]);
				if(_maxreduction[tid] < _maxreduction[tid+2])
				{
					_maxreduction[tid]= _maxreduction[tid +2];
					_maxreductionid[tid]= _maxreductionid[tid +2];
				}
			}

			//1
			if(blockSize >= 2)
			{
				if(_minreduction[tid] == _minreduction[tid+1]) _minreductionid[tid] = min(_minreductionid[tid], _minreductionid[tid +1]);
				if(_minreduction[tid] > _minreduction[tid+1])
				{
					_minreduction[tid]= _minreduction[tid +1];
					_minreductionid[tid]= _minreductionid[tid +1];
				}

				if(_maxreduction[tid] == _maxreduction[tid+1]) _maxreductionid[tid] = min(_maxreductionid[tid], _maxreductionid[tid +1]);
				if(_maxreduction[tid] < _maxreduction[tid+1])
				{
					_maxreduction[tid]= _maxreduction[tid +1];
					_maxreductionid[tid]= _maxreductionid[tid +1];
				}
			}
		}
		__syncthreads();

		if(tid==0)
		{
			d_bup[bidy * gridDim.x + bidx]=minreduction[tid];
			d_blow[bidy * gridDim.x + bidx]=maxreduction[tid];
			d_Iup[bidy * gridDim.x + bidx]= minreductionid[tid];
			d_Ilow[bidy * gridDim.x + bidx]= maxreductionid[tid];

		}

	}

}


/**
 * Performs an optimized local global reduction to find global Iup, Ilow, Bup and Blow
 * @param d_bup device pointer to the local bup values
 * @param d_blow device pointer to the local blow values
 * @param d_Iup device pointer to the local Iup values
 * @param d_Ilow device pointer to the local Ilow values
 * @param d_done_device pointer to the array with the status of each binary task
 * @param d_active device pointer to the array with active binary tasks
 * @param n number of local blockwise reduction results that need to be globally reduced
 * @param activeTasks number of active tasks
 */
template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void globalreduction(		float* d_bup,
												float* d_blow,
												int* d_Iup,
												int* d_Ilow,
												int* d_done,
												int* d_active,
												int n,
												int activeTasks)
{
		const unsigned int tid = threadIdx.x;
		const unsigned int bidx = blockIdx.x;
		unsigned int j = blockIdx.y;
		unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
		unsigned int gridSize = blockSize*2*gridDim.x;

		int bidy= d_active[j];

		//Check if the task has converged
		if(d_done[bidy]== 0)
		{

			__shared__ float minreduction [blockSize];
			__shared__ float maxreduction [blockSize];
			__shared__ int minreductionid [blockSize];
			__shared__ int maxreductionid [blockSize];

			//Each thread loads one element
			minreduction[tid]= (float)FLT_MAX;
			maxreduction[tid]= (float)FLT_MIN;
			minreductionid[tid]= i;
			maxreductionid[tid]= i;

			while (i < n)
			{

				float minval=d_bup[bidy* n + i];
				float maxval= d_blow[bidy* n + i];

				if(minreduction[tid] > minval)
				{
					minreduction[tid]= minval;
					minreductionid[tid]= d_Iup[bidy* n + i];
				}

				if(maxreduction[tid] < maxval)
				{
					maxreduction[tid]= maxval;
					maxreductionid[tid]= d_Ilow[bidy* n + i];
				}

				if (isNtrainingPow2 || i + blockSize < n)
				{
					minval=d_bup[bidy* n + i+blockSize];
					maxval= d_blow[bidy* n + i+blockSize];

					if(minreduction[tid] > minval)
					{
						minreduction[tid]= minval;
						minreductionid[tid]= d_Iup[bidy* n + i+blockSize];
					}

					if(maxreduction[tid] < maxval)
					{
						maxreduction[tid]= maxval;
						maxreductionid[tid]= d_Ilow[bidy* n + i+blockSize];
					}
				}

				i += gridSize;

			}


			__syncthreads();


			if(blockSize>=512)
			{
				if(tid<256)
				{
					if(minreduction[tid] > minreduction[tid+256])
					{
						minreduction[tid]= minreduction[tid +256];
						minreductionid[tid]= minreductionid[tid +256];
					}

					if(maxreduction[tid] < maxreduction[tid+256])
					{
						maxreduction[tid]= maxreduction[tid +256];
						maxreductionid[tid]= maxreductionid[tid +256];
					}
				}
				__syncthreads();
			}

			if(blockSize>=256)
			{
				if(tid<128)
				{
					if(minreduction[tid] > minreduction[tid+128])
					{
						minreduction[tid]= minreduction[tid +128];
						minreductionid[tid]= minreductionid[tid +128];
					}

					if(maxreduction[tid] < maxreduction[tid+128])
					{
						maxreduction[tid]= maxreduction[tid +128];
						maxreductionid[tid]= maxreductionid[tid +128];
					}
				}
				__syncthreads();
			}

			if(blockSize>=128)
			{
				if(tid<64)
				{
					if(minreduction[tid] > minreduction[tid+64])
					{
						minreduction[tid]= minreduction[tid +64];
						minreductionid[tid]= minreductionid[tid +64];
					}

					if(maxreduction[tid] < maxreduction[tid+64])
					{
						maxreduction[tid]= maxreduction[tid +64];
						maxreductionid[tid]= maxreductionid[tid +64];
					}
				}
				__syncthreads();
			}

			if(tid<32)
			{
				volatile float *_minreduction = minreduction;
				volatile float *_maxreduction = maxreduction;
				volatile int *_minreductionid = minreductionid;
				volatile int *_maxreductionid = maxreductionid;

				//32
				if(blockSize >= 64)
				{

					if(_minreduction[tid] > _minreduction[tid+32])
					{
						_minreduction[tid]= _minreduction[tid +32];
						_minreductionid[tid]= _minreductionid[tid +32];
					}

					if(_maxreduction[tid] < _maxreduction[tid+32])
					{
						_maxreduction[tid]= _maxreduction[tid +32];
						_maxreductionid[tid]= _maxreductionid[tid +32];
					}
				}
				//16
				if(blockSize >= 32)
				{
					if(_minreduction[tid] > _minreduction[tid+16])
					{
						_minreduction[tid]= _minreduction[tid +16];
						_minreductionid[tid]= _minreductionid[tid +16];
					}

					if(_maxreduction[tid] < _maxreduction[tid+16])
					{
						_maxreduction[tid]= _maxreduction[tid +16];
						_maxreductionid[tid]= _maxreductionid[tid +16];
					}
				}
				//8
				if(blockSize >= 16)
				{
					if(_minreduction[tid] > _minreduction[tid+8])
					{
						_minreduction[tid]= _minreduction[tid +8];
						_minreductionid[tid]= _minreductionid[tid +8];
					}

					if(_maxreduction[tid] < _maxreduction[tid+8])
					{
						_maxreduction[tid]= _maxreduction[tid +8];
						_maxreductionid[tid]= _maxreductionid[tid +8];
					}
				}
				//4
				if(blockSize >= 8)
				{
					if(_minreduction[tid] > _minreduction[tid+4])
					{
						_minreduction[tid]= _minreduction[tid +4];
						_minreductionid[tid]= _minreductionid[tid +4];
					}

					if(_maxreduction[tid] < _maxreduction[tid+4])
					{
						_maxreduction[tid]= _maxreduction[tid +4];
						_maxreductionid[tid]= _maxreductionid[tid +4];
					}
				}
				//2
				if(blockSize >= 4)
				{
					if(_minreduction[tid] > _minreduction[tid+2])
					{
						_minreduction[tid]= _minreduction[tid +2];
						_minreductionid[tid]= _minreductionid[tid +2];
					}

					if(_maxreduction[tid] < _maxreduction[tid+2])
					{
						_maxreduction[tid]= _maxreduction[tid +2];
						_maxreductionid[tid]= _maxreductionid[tid +2];
					}
				}

				//1
				if(blockSize >= 2)
				{
					if(_minreduction[tid] > _minreduction[tid+1])
					{
						_minreduction[tid]= _minreduction[tid +1];
						_minreductionid[tid]= _minreductionid[tid +1];
					}

					if(_maxreduction[tid] < _maxreduction[tid+1])
					{
						_maxreduction[tid]= _maxreduction[tid +1];
						_maxreductionid[tid]= _maxreductionid[tid +1];
					}
				}
			}
			__syncthreads();

			if(tid==0)
			{
				d_bup[bidy * gridDim.x + bidx]=minreduction[tid];
				d_blow[bidy * gridDim.x + bidx]=maxreduction[tid];
				d_Iup[bidy * gridDim.x + bidx]= minreductionid[tid];
				d_Ilow[bidy * gridDim.x + bidx]= maxreductionid[tid];

			}

		}
		
}

#endif // _REDUCTION_KERNEL_H_
