#include "../../include/globalparams.h"
#include "../../include/utilities.h"


/**
 * Performs the global reduction step in serial code
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
 * @param ntasks number of binary tasks to be solved
 * @param numBlockRed number of blocks in the reduction
 */

void globalparamsserial(	float* h_bup,
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
							int ntasks,
							int numBlocksRed)
{

	int bmem_size= ntasks* numBlocksRed*sizeof(float);
	int imem_size= ntasks* numBlocksRed*sizeof(int);

	cudaMemcpy(h_bup, d_bup, bmem_size ,cudaMemcpyDeviceToHost);
	cudaError_t errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}
	cudaMemcpy(h_blow, d_blow, bmem_size,cudaMemcpyDeviceToHost);
	errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}
	cudaMemcpy(h_Iup, d_Iup, imem_size,cudaMemcpyDeviceToHost);
	errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}
	cudaMemcpy(h_Ilow, d_Ilow, imem_size,cudaMemcpyDeviceToHost);
	errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}

	// Loop to find global params
	for (int i=0; i< ntasks; i++)
	{
		if(h_done[i]==0)
		{
			float g_bup= h_bup[i * numBlocksRed];
			int g_Iup= h_Iup[i * numBlocksRed];

			float g_blow= h_blow[i * numBlocksRed];
			int g_Ilow= h_Ilow[i*numBlocksRed];

			for (int j=0; j<numBlocksRed; j++)
			{
				//Find minimum bup and Iup
				if(h_bup[i * numBlocksRed + j] < g_bup)
				{
					g_bup= h_bup[i * numBlocksRed +j];
					g_Iup= h_Iup[i * numBlocksRed +j];
				}
				//Find maximum blow and Ilow
				if(h_blow[i * numBlocksRed + j] > g_blow)
				{
					g_blow= h_blow[i * numBlocksRed +j];
					g_Ilow= h_Ilow[i * numBlocksRed +j];
				}

			}
			h_bup_global[i]= g_bup;
			h_Iup_global[i]= g_Iup;
			h_blow_global[i]= g_blow;
			h_Ilow_global[i]= g_Ilow;


		}
	}

	cudaMemcpy(d_bup_global, h_bup_global, sizeof(float)* ntasks ,cudaMemcpyHostToDevice);
	errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}
	cudaMemcpy(d_Iup_global, h_Iup_global, sizeof(int)* ntasks ,cudaMemcpyHostToDevice);
	errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}
	cudaMemcpy(d_blow_global, h_blow_global, sizeof(float)* ntasks ,cudaMemcpyHostToDevice);
	errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}
	cudaMemcpy(d_Ilow_global, h_Ilow_global, sizeof(int)* ntasks ,cudaMemcpyHostToDevice);

	errormcpy= cudaGetLastError();
	if(errormcpy)
	{
		printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
	}
}

/**
 * Performs the global reduction step in parallel code
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
 * @param ntasks number of binary tasks to be solved
 * @param n number of local reductions that need to be globally reduced
 * @param activeTasks number of non converged binary tasks
 */
void globalparamsparallel(		float* d_bup,
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
								int* d_done,
								int* d_active,
								int ntasks,
								int n,
								int activeTasks)
{

	int numThreads = (n < MAXTHREADS*2) ? nextPow2((n + 1)/ 2) : MAXTHREADS;
	int numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);
	int numBlocksRed = min(MAXBLOCKS, numBlocks);

	int smemSize = 0;
	bool isNtrainingPow2=isPow2(n);

	int bglobalmem_size= ntasks* numBlocksRed*sizeof(float);
	int iglobalmem_size= ntasks* numBlocksRed*sizeof(int);

	while( true)
	{
		dim3 dimBlockReduction(numThreads, 1, 1);
		dim3 dimGridReduction(numBlocksRed, activeTasks, 1);


		if(isNtrainingPow2)
		{
			switch (numThreads)
			{
				case 512:
					globalreduction <512,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 256:
					globalreduction <256,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 128:
					globalreduction <128,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 64:
					globalreduction <64,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 32:
					globalreduction <32,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 16:
					globalreduction <16,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  8:
					globalreduction <8,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  4:
					globalreduction <4,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  2:
					globalreduction <2,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  1:
					globalreduction <1,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,	d_Ilow,	d_done,d_active,n, activeTasks); break;
			}
		}
		else
		{
			switch (numThreads)
			{
				case 512:
					globalreduction <512,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 256:
					globalreduction <256,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 128:
					globalreduction <128,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 64:
					globalreduction <64,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 32:
					globalreduction <32,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case 16:
					globalreduction <16,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  8:
					globalreduction <8,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  4:
					globalreduction <4,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  2:
					globalreduction <2,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
				case  1:
					globalreduction <1,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_bup,d_blow,d_Iup,d_Ilow,	d_done,d_active,n, activeTasks); break;
			}
		}

		cudaThreadSynchronize();

		if(numBlocksRed==1)
		{
			//printf("here");
			cudaMemcpy(d_bup_global, d_bup, bglobalmem_size ,cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_blow_global, d_blow, bglobalmem_size ,cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_Iup_global, d_Iup, iglobalmem_size ,cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_Ilow_global, d_Ilow, iglobalmem_size ,cudaMemcpyDeviceToDevice);

			cudaError_t errormcpy= cudaGetLastError();
			if(errormcpy)
			{
				printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
			}

			cudaMemcpy(h_bup_global, d_bup_global, bglobalmem_size ,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_blow_global, d_blow_global, bglobalmem_size ,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Iup_global, d_Iup_global, iglobalmem_size ,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Ilow_global, d_Ilow_global, iglobalmem_size ,cudaMemcpyDeviceToHost);

			errormcpy= cudaGetLastError();
			if(errormcpy)
			{
				printf("Errors making memory copy:, %s,\n", cudaGetErrorString(errormcpy));
			}


			break;
		}
		else
		{
			n=numBlocksRed;

			numThreads = (n < MAXTHREADS*2) ? nextPow2((n + 1)/ 2) : MAXTHREADS;
			numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);
			numBlocksRed = min(MAXBLOCKS, numBlocks);

			bglobalmem_size= ntasks* numBlocksRed*sizeof(float);
			iglobalmem_size= ntasks* numBlocksRed*sizeof(int);

		}
	}
}
