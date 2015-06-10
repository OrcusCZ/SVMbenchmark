#include "../../include/kernelevaluation.h"
#include "../../include/utilities.h"




/**
 * Evaluate a row of the gram matrix
 * @param d_xtraindata device pointer to the training set
 * @param d_dottraindata device pointer to the array containing the dot product of the row with itself
 * @param d_kernelrow device pointer that will store the array extracted from d_xtraindata.
 * @param d_kerneldot device pointer that will store the result of the kernel evaluation
 * @param d_kdata device pointer to the matrix that stores the cached values
 * @param gid index that points to the point in d_xtraindata to be  calculated
 * @param cacheid index that points to the location in cache that will keep the results
 * @param ntraining number of training samples in the training set
 * @param nfeatures number of features in the training samples
 * @param beta value of the parameter of the RBF kernel
 * @param a if using polynomial or sigmoid kernel the value of a x_i x_j
 * @param b if using polynomial or sigmoid kernel the value of b
 * @param d if using polynomial kernel
 * @param kernelcode code that indicates the kernel type to run
 */
void kerneleval ( 	float* d_xtraindata,
					float* d_dottraindata,
					float* d_kernelrow,
					float* d_kerneldot,
					float* d_kdata,
					int gid,
					int cacheid,
					int ntraining,
					int nfeatures,
					float beta,
					float a,
					float b,
					float d,
					int kernelcode)
{

	int numThreads = (nfeatures < MAXTHREADS*2) ? nextPow2((nfeatures + 1)/ 2) : MAXTHREADS;
	int numBlocks = (nfeatures + (numThreads * 2 - 1)) / (numThreads * 2);
	int numBlocksRed = min(MAXBLOCKS, numBlocks);

	dim3 dimBlockKernelRow(numThreads, 1, 1);
	dim3 dimGridKernelRow(numBlocksRed, 1, 1);

	int smemSize = 0;
	bool isNtrainingPow2=isPow2(nfeatures);


	if(isNtrainingPow2)
	{
		switch (numThreads)
		{
			case 512:
				ExtractKernelRow <512,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 256:
				ExtractKernelRow <256,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 128:
				ExtractKernelRow <128,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 64:
				ExtractKernelRow <64,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 32:
				ExtractKernelRow <32,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 16:
				ExtractKernelRow <16,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  8:
				ExtractKernelRow <8,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  4:
				ExtractKernelRow <4,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  2:
				ExtractKernelRow <2,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  1:
				ExtractKernelRow <1,true><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
		}
	}
	else
	{
		switch (numThreads)
		{
			case 512:
				ExtractKernelRow <512,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 256:
				ExtractKernelRow <256,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 128:
				ExtractKernelRow <128,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 64:
				ExtractKernelRow <64,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 32:
				ExtractKernelRow <32,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case 16:
				ExtractKernelRow <16,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  8:
				ExtractKernelRow <8,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  4:
				ExtractKernelRow <4,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  2:
				ExtractKernelRow <2,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
			case  1:
				ExtractKernelRow <1,false><<< dimGridKernelRow, dimBlockKernelRow, smemSize >>>(d_xtraindata,d_kernelrow,	gid,ntraining,nfeatures); break;
		}
	}

	cudaThreadSynchronize();

	cublasSgemv ('n',ntraining,	nfeatures,1,d_xtraindata,ntraining,d_kernelrow,	1,0,d_kerneldot,1);
	cublasStatus status;
	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
	}



	numThreads = (ntraining < MAXTHREADS*2) ? nextPow2((ntraining + 1)/ 2) : MAXTHREADS;
	numBlocks = (ntraining + (numThreads * 2 - 1)) / (numThreads * 2);
	numBlocksRed = min(MAXBLOCKS, numBlocks);

	dim3 dimBlockReduction(numThreads, 1, 1);
	dim3 dimGridReduction(numBlocksRed, 1, 1);

	smemSize = 0;
	isNtrainingPow2=isPow2(ntraining);

	if(isNtrainingPow2)
	{
		switch (numThreads)
		{
			case 512:
				SetKernelDot <512,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 256:
				SetKernelDot <256,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 128:
				SetKernelDot <128,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 64:
				SetKernelDot <64,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 32:
				SetKernelDot <32,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 16:
				SetKernelDot <16,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  8:
				SetKernelDot <8,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  4:
				SetKernelDot <4,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  2:
				SetKernelDot <2,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  1:
				SetKernelDot <1,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
		}
	}
	else
	{
		switch (numThreads)
		{
			case 512:
				SetKernelDot <512,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 256:
				SetKernelDot <256,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 128:
				SetKernelDot <128,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 64:
				SetKernelDot <64,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 32:
				SetKernelDot <32,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case 16:
				SetKernelDot <16,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  8:
				SetKernelDot <8,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  4:
				SetKernelDot <4,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  2:
				SetKernelDot <2,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
			case  1:
				SetKernelDot <1,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_kdata,d_dottraindata,d_kerneldot,gid,cacheid,ntraining,beta,a,b,d,kernelcode); break;
		}
	}

	cudaThreadSynchronize();

}



