#include "../../include/parseinputs.h"

/**
 * Parses data from file
 * @param inputfilename pointer to the char array that contains the file name
 * @param h_xdata host pointer to the array that will store the training set
 * @param h_ldata host pointer to the array that will store the labels of the training set
 * @param nsamples number of samples in the training set
 * @param nfeatures number of features per sample in the training set
 * @param nclasses number of classes
 */
int parsedata (const char* inputfilename, float* h_xdata, int * h_ldata, int nsamples, int nfeatures, int nclasses)
{
	FILE* inputFilePointer = fopen(inputfilename, "r");
	if (inputFilePointer == 0)
	{
		printf("File not found\n");
		return 0;
	}

	char* stringBuffer = (char*)malloc(65536);

	for(int i = 0; i < nsamples; i++)
	{
		char c;
		int index=0;
		char* bufferPointer = stringBuffer;

		do
		{
			c = fgetc(inputFilePointer);
			if((c== ',') || (c == '\n'))
			{
				if (index < nfeatures)
				{
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					h_xdata[i*nfeatures + index]= value;
					index++;
				}
				else
				{
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);
					h_ldata[i]=value;
					index=0;
				}
				bufferPointer = stringBuffer;
			}
			else
			{
				*(bufferPointer) = c;
				bufferPointer++;
			}

		}
		while (c != '\n');

	}
	free(stringBuffer);
	fclose(inputFilePointer);
	return 1;
}

/**
 * Parses data from file containing the output code
 * @param inputfilename pointer to the char array that contains the file name
 * @param h_rdata host pointer to the array that contains the mapping between classes and binary labels
 * @param nclasses number of classes
 * @param ntasks number of tasks in the input file
 */
int parsecode (const char* inputfilename, int * h_rdata, int nclasses, int ntasks)
{
	FILE* inputFilePointer = fopen(inputfilename, "r");
	if (inputFilePointer == 0)
	{
		printf("File not found\n");
		return 0;
	}

	char* stringBuffer = (char*)malloc(65536);

	for(int i = 0; i < nclasses; i++)
	{
		char c;
		int index=0;
		char* bufferPointer = stringBuffer;

		do
		{
			c = fgetc(inputFilePointer);
			if((c== ',') || (c == '\n'))
			{
				if(index<ntasks)
				{
					//code found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);
					h_rdata[i*ntasks + index]=value;
					index++;
				}
				else
				{
					index=0;
				}

				bufferPointer = stringBuffer;
			}
			else
			{
				*(bufferPointer) = c;
				bufferPointer++;
			}

		}
		while (c != '\n');

	}
	free(stringBuffer);
	fclose(inputFilePointer);
	return 1;
}


/**
 * Parses data from file in the libsvm format
 * @param inputfilename pointer to the char array that contains the file name
 * @param h_xdata host pointer to the array that will store the training set
 * @param h_ldata host pointer to the array that will store the labels of the training set
 * @param nsamples number of samples in the training set
 * @param nfeatures number of features per sample in the training set
 * @param nclasses number of classes
 */
int parsedatalibsvm (const char* inputfilename, float* h_xdata, int * h_ldata, int nsamples, int nfeatures, int nclasses)
{
	FILE* inputFilePointer = fopen(inputfilename, "r");
	if (inputFilePointer == 0)
	{
		printf("File not found\n");
		return 0;
	}

	char* stringBuffer = (char*)malloc(65536);

	for(int i = 0; i < nsamples; i++)
	{
		char c;
		int pos=0;
		char* bufferPointer = stringBuffer;

		do
		{
			c = fgetc(inputFilePointer);

			if((c== ' ') || (c == '\n'))
			{
				if(pos==0)
				{
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);
					h_ldata[i]=(value > 0)? value : 2; //force binary classification task to idx 1 and 2
					pos++;
				}
				else
				{
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					h_xdata[i*nfeatures + pos-1]= value;
				}
				bufferPointer = stringBuffer;
			}
			else if(c== ':')
			{
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos= value;
				bufferPointer = stringBuffer;
			}
			else
			{
				*(bufferPointer) = c;
				bufferPointer++;
			}

		}
		while (c != '\n');

	}
	free(stringBuffer);
	fclose(inputFilePointer);
	return 1;
}


/**
 * Prints data to the screen
 * @param h_xdata host pointer to the array that will store the training set
 * @param h_ldata host pointer to the array that will store the labels of the training set
 * @param n number of samples in the training set
 * @param nfeatures number of features per sample in the training set
 */
void printdata(float * h_xdata, int* h_ldata,  int n, int nfeatures)
{

	for (int i=0; i<n; i++)
	{
		printf("Sample %i: ", i);

		for (int j=0; j<nfeatures; j++)
		{
			printf(" %f,", h_xdata[j*n + i]);
		}
		printf(" Label:  %i\n", h_ldata[i]);
	}

}

/**
 * Prints code to the screen
 * @param h_ldata host pointer to the array that will store the labels of the training set
 * @param nclasses number of classes in the multiclass problem
 * @param ntasks number of tasks
 */
void printcode(int* h_rdata, int nclasses, int ntasks)
{

	for (int i=0; i<nclasses; i++)
	{
		printf("Class %i: ", i);
		for (int j=0; j<ntasks; j++)
		{
			printf(" %i,", h_rdata[i*ntasks + j]);
		}
		printf("\n");
	}

}

/**
 * Generates the All-vs-All code
 * @param h_rdata host pointer to the generated output matrix
 * @param nclasses number of classes in the multiclass problem
 * @param ntasks number of tasks
 */
void generateavacode (int* h_rdata, int nclasses, int ntasks)
{
	int start= nclasses -1;
	int section=0;
	int neg=section+1;
	int k=0;

	for (int j=0; j<ntasks; j++)
	{
		for (int i=0; i<nclasses; i++)
		{
			if( i==section)
			{
				h_rdata[i*ntasks + j]=1;
			}
			else if( i== neg)
			{
				h_rdata[i*ntasks + j]=-1;
			}
			else
			{
				h_rdata[i*ntasks + j]=0;
			}
		}
		neg++;
		k++;

		if(k==start)
		{
			start--;
			section++;
			neg=section+1;
			k=0;
		}
	}
}

/**
 * Generates the One-vs-All
 * @param h_rdata host pointer to the generated output matrix
 * @param nclasses number of classes in the multiclass problem
 * @param ntasks number of tasks
 */
void generateovacode (int* h_rdata, int nclasses, int ntasks)
{
	for (int j=0; j<ntasks; j++)
	{
		for (int i=0; i<nclasses; i++)
		{
			if(j==i)
			{
				h_rdata[i*ntasks + j]=1;
			}
			else
			{
				h_rdata[i*ntasks + j]=-1;
			}
		}
	}
}

/**
 * Generates the Even-vs-Odd
 * @param h_rdata host pointer to the generated output matrix
 * @param nclasses number of classes in the multiclass problem
 * @param ntasks number of tasks
 */
void generateevenoddcode (int* h_rdata, int nclasses, int ntasks)
{
	for (int j=0; j<ntasks; j++)
	{
		for (int i=0; i<nclasses; i++)
		{
			if(i%2==1)
			{
				h_rdata[i*ntasks + j]=1;
			}
			else
			{
				h_rdata[i*ntasks + j]=-1;
			}

		}
	}
}


