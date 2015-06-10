#include "../../include/reduction.h"
#include "../../include/utilities.h"

/**
 * Performs an optimized reduction step to find Iup, Ilow, Bup and Blow
 * @param d_ytraindata device pointer to the array of binary labels
 * @param d_atraindata device pointer to the array of alphas
 * @param d_fdata device pointer to the array of fs
 * @param h_bup host pointer to the local bup values
 * @param h_blow host pointer to the local blow values
 * @param h_Iup host pointer to the local Iup values
 * @param h_Ilow host pointer to the local Ilow values
 * @param d_bup device pointer to the local bup values
 * @param d_blow device pointer to the local blow values
 * @param d_Iup device pointer to the local Iup values
 * @param d_Ilow device pointer to the local Ilow values
 * @param h_bup_global host pointer to the global bup values
 * @param h_blow_global host pointer to the global blow values
 * @param h_Iup_global host pointer to the global Iup values
 * @param h_Ilow_global host pointer to the global Ilow values
 * @param d_bup_global device pointer to the global bup values
 * @param d_blow_global device pointer to the global blow values
 * @param d_Iup_global device pointer to the global Iup values
 * @param d_Ilow_global device pointer to the global Ilow values
 * @param h_done host pointer to the array with the status of each binary task
 * @param d_done_device pointer to the array with the status of each binary task
 * @param d_active device pointer to the array with active binary tasks
 * @param numthreads number of threads per block
 * @param numBlockRed number of blocks in the reduction
 * @param ntraining number of training samples in the training set
 * @param ntasks number of binary tasks to be solved
 * @param activeTasks number of active tasks
 * @param d_C device pointer to the array of regularization parameters
 */
void reductionstep(         int* d_ytraindata,
							float* d_atraindata,
							float* d_fdata,
							float* h_bup,
							float* h_blow,
							int* h_Iup,
							int* h_Ilow,
							float* d_bup,
							float* d_blow,
							int* d_Iup,
							int* d_Ilow,
							float* h_bup_global,
							float* h_blow_global,
							int* h_Iup_global,
							int* h_Ilow_global,
							float* d_bup_global,
							float* d_blow_global,
							int* d_Iup_global,
							int* d_Ilow_global,
							int* h_done,
							int* d_done,
							int* d_active,
							int numThreads,
							int numBlocksRed,
							int ntraining,
							int ntasks,
							int activeTasks,
							float* d_C)
{

	int smemSize = 0;
	bool isNtrainingPow2=isPow2(ntraining);

	dim3 dimBlockActiveReduction(numThreads, 1, 1);
	dim3 dimGridActiveReduction(numBlocksRed, activeTasks, 1);

	if(isNtrainingPow2)
	{
		switch (numThreads)
		{
			case 512:
				reduction <512,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 256:
				reduction <256,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 128:
				reduction <128,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 64:
				reduction <64,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 32:
				reduction <32,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 16:
				reduction <16,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  8:
				reduction <8,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  4:
				reduction <4,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  2:
				reduction <2,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  1:
				reduction <1,true><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
		}
	}
	else
	{
		switch (numThreads)
		{
			case 512:
				reduction <512,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 256:
				reduction <256,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 128:
				reduction <128,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 64:
				reduction <64,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 32:
				reduction <32,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case 16:
				reduction <16,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  8:
				reduction <8,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  4:
				reduction <4,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,	d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  2:
				reduction <2,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,	d_active,ntraining,ntasks,activeTasks,d_C); break;
			case  1:
				reduction <1,false><<< dimGridActiveReduction, dimBlockActiveReduction, smemSize >>>(d_ytraindata,d_atraindata,	d_fdata,d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,ntraining,ntasks,activeTasks,d_C); break;
		}
	}



	cudaThreadSynchronize();


	cudaError_t error= cudaGetLastError();

	if(error)
	{
		printf("Errors Reduction!, %s,\n", cudaGetErrorString(error));
		getchar();
	}

	//globalparamsparallel(d_bup,	d_blow,	d_Iup,d_Ilow,h_bup_global,h_blow_global,h_Iup_global,h_Ilow_global,	d_bup_global,d_blow_global,	d_Iup_global,d_Ilow_global,	d_done,	d_active,ntasks,numBlocksRed,activeTasks);

	globalparamsserial(h_bup,h_blow,h_Iup,h_Ilow,d_bup,d_blow, d_Iup,d_Ilow,h_bup_global,h_blow_global,h_Iup_global,h_Ilow_global,d_bup_global, d_blow_global,d_Iup_global,d_Ilow_global,h_done, d_done,ntasks,numBlocksRed);



}
