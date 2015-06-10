#include "svmIO.h"



std::pair<std::string, Parameters> matchedParameters[] = {
  std::pair<std::string, Parameters>("gamma",       p_gamma),
  std::pair<std::string, Parameters>("coef",        p_coef0),
  std::pair<std::string, Parameters>("degree",      p_degree),
  std::pair<std::string, Parameters>("rho",         p_rho),
  std::pair<std::string, Parameters>("kernel_type", p_kernel_type),
  std::pair<std::string, Parameters>("svm_type",    p_svm_type),
  std::pair<std::string, Parameters>("nr_class",    p_nr_class),
  std::pair<std::string, Parameters>("total_sv",    p_total_sv),
  std::pair<std::string, Parameters>("label",       p_label),
  std::pair<std::string, Parameters>("nr_sv",       p_nr_sv),
  std::pair<std::string, Parameters>("SV",          p_SV)
};

// Map to associate the strings with the enum values
std::map<std::string, Parameters>
mapStringToParameters(matchedParameters,
                      matchedParameters +
                      sizeof(matchedParameters)/sizeof(matchedParameters[0]));


float readFloat(FILE* input) {
  char currentCharacter;
  char floatBuffer[50];
  char* bufferPointer = floatBuffer;
  do {
    currentCharacter = fgetc(input);
    *(bufferPointer) = currentCharacter;
    bufferPointer++;
  } while ((currentCharacter >= '0' && currentCharacter <= '9') ||
	   (currentCharacter == '.') || (currentCharacter == '-') ||
	   (currentCharacter == 'e') || (currentCharacter == 'E'));
  bufferPointer--;
  *(bufferPointer) = 0;
  float result;
  sscanf(floatBuffer, "%f", &result);
	ungetc(currentCharacter, input);
  return result;
}

int readModel(const char* filename, float** alpha, float** supportVectors, int* nSVOut, int* nDimOut, Kernel_params* kp, float* p_class1Label, float* p_class2Label) {
	
  FILE* input = fopen(filename, "r");
  if (input == 0) {
    printf("Model File not found\n");
    return 0;
  }
  std::string line_description;

  char currentLine[80];
	int parsingParameters = 1;
	while (parsingParameters > 0) {
		fgets(currentLine, 80, input);
		line_description.clear();
		int j = 0;
		while(isalpha(currentLine[j]) || currentLine[j]=='_')
		{
			line_description.push_back(currentLine[j]);
			j++;
		}
		
		int foundParameter = 0;
		switch (mapStringToParameters[line_description]) {
			case p_gamma:
				//printf("Reading gamma - %s\n",line_description.c_str() );
				sscanf(&currentLine[line_description.length()+1], "%f", &(kp->gamma));
				foundParameter = 1;
				printf("Found gamma = %f\n",kp->gamma);
				break;
			case p_total_sv:
				sscanf(&currentLine[line_description.length()+1], "%d", nSVOut);
				foundParameter = 1;
				break;
			case p_rho:
				sscanf(&currentLine[line_description.length()+1], "%f", &(kp->b));
				kp->b = -kp->b;
				foundParameter = 1;
				break;
			case p_degree:
				sscanf(&currentLine[line_description.length()+1], "%d", &(kp->degree));
				foundParameter = 1;
				printf("Found degree = %d\n", kp->degree);
				break;
			case p_coef0:
				//printf("Reading coef0 - %s\n",line_description.c_str() );
				sscanf(&currentLine[line_description.length()+1], "%f", &(kp->coef0));
				foundParameter = 1;
				printf("Found coef0=%f\n",kp->coef0);
				break;
			case p_kernel_type:
				kp->kernel_type.insert(0,currentLine+line_description.length()+1); 
				foundParameter = 1;
				break;
			case p_label:
        int intLabel1, intLabel2;
				sscanf(&currentLine[line_description.length()+1], "%d %d", &intLabel1, &intLabel2);
        *p_class1Label = (float)intLabel1;
        *p_class2Label = (float)intLabel2;
				foundParameter=1;
				break;
			case p_SV:
				parsingParameters = 0;
			default:
				foundParameter = 1;
				break;
		}
		if (foundParameter == 0) {
			printf("Malformed model: %s\n", currentLine);
			return 0;
		}
	}

  int nSV = *nSVOut;
  float* localAlpha = (float*)malloc(sizeof(float)*nSV);
  float* currentSV = (float*)malloc(sizeof(float)*65536);//Don't know dimension
  //guessing it will be less than 65536.  Otherwise, we'll fail
  //This should be rewritten
 
  localAlpha[0] = readFloat(input);
  int nDimension = 0;
  char currentCharacter = 0;
  while (currentCharacter != '\n') {
    currentCharacter = fgetc(input);
    if (currentCharacter == ':') {
      float currentCoordinate = readFloat(input);
      currentSV[nDimension] = currentCoordinate;
      nDimension++;
    }
  }
  float* localSV = (float*)malloc(sizeof(float)*nSV*nDimension);
	for(int dim = 0; dim < nDimension; dim++) {
		localSV[dim*nSV] = currentSV[dim]; 
	}
  free(currentSV);
  
  for(int sv = 1; sv < nSV; sv++) {
    localAlpha[sv] = readFloat(input);
    for(int dim = 0; dim < nDimension; dim++) {
      do {
				currentCharacter = fgetc(input);
			} while(currentCharacter != ':');
			localSV[nSV*dim + sv] = readFloat(input);
    }
		do {
			currentCharacter = fgetc(input);
		} while (((currentCharacter < '0') || (currentCharacter > '9')) && (currentCharacter != '.') && (currentCharacter != '-') && (currentCharacter != 'e') && (currentCharacter != 'E') && (currentCharacter >= 0));
		ungetc(currentCharacter, input);
		
  }

  *alpha = localAlpha;
  *supportVectors = localSV;
  *nSVOut = nSV;
  *nDimOut = nDimension;
  return 1;
}

int readSvm(const char* filename, float** p_data, float** p_labels, int* p_npoints, int* p_dimension, float** p_transposed_data) {
	FILE* inputFilePointer = fopen(filename, "r");
	if (inputFilePointer == 0) {
		printf("File not found\n");
		return 0;
	}
	int npoints = 0;
	int dimension = 0;
	char c;
	char firstLine = 1;
	do {
		c = fgetc(inputFilePointer);
		switch(c) {
		case '\n':
			npoints++;
			firstLine = 0;
			break;
		case ':':
			if (firstLine > 0) {
				dimension++;
			}
		default:
			;
		}
			
	} while (c != EOF);
	rewind(inputFilePointer);
	*(p_npoints) = npoints;
	*(p_dimension) = dimension;

	
	
	float* data = (float*)malloc(sizeof(float)*npoints*dimension);
	float* labels = (float*)malloc(sizeof(float)*npoints);
	*(p_data) = data;
	*(p_labels) = labels;
  float* transposed_data = NULL;
  if (p_transposed_data != NULL) {
    transposed_data = (float*)malloc(sizeof(float)*npoints*dimension);
    *(p_transposed_data) = transposed_data;
  }

	char* stringBuffer = (char*)malloc(65536);
	
	for(int i = 0; i < npoints; i++) {
		char* bufferPointer = stringBuffer;
		char validCharacter = 1;
		int currentDim = 0;
		int parsingLabel = 1;

		do {
			c = fgetc(inputFilePointer);
			if (validCharacter > 0) {
				if ((c == ' ') || (c == '\n')) {
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					if (parsingLabel > 0) {
						labels[i] = value;
						parsingLabel = 0;
					} else {
						data[currentDim*npoints + i] = value;
            if (p_transposed_data != NULL) {
              transposed_data[i*dimension + currentDim] = value;
            }
						currentDim++;
					}
					validCharacter = 0;
					bufferPointer = stringBuffer;
				} else {
					*(bufferPointer) = c;
					bufferPointer++;
				}
			}
			if (c == ':') {
				validCharacter = 1;
			}
		} while (c != '\n');
	}
				
	free(stringBuffer);
	fclose(inputFilePointer);
	return 1;
}

void printModel(const char* outputFile, Kernel_params kp, float* alpha, float* labels, float* data, int nPoints, int nDimension, float epsilon) { 

	printf("Output File: %s\n", outputFile);
	FILE* outputFilePointer = fopen(outputFile, "w");
	if (outputFilePointer == NULL) {
		printf("Can't write %s\n", outputFile);
		exit(1);
	}

	int nSV = 0;
	int pSV = 0;
	for(int i = 0; i < nPoints; i++) {
		if (alpha[i] > epsilon) {
			if (labels[i] > 0) {
				pSV++;
			} else {
				nSV++;
			}
		}
	}

  bool printGamma = false;
  bool printCoef0 = false;
  bool printDegree = false;
  const char* kernelType = kp.kernel_type.c_str();
  if (strncmp(kernelType, "polynomial", 10) == 0) {
    printGamma = true;
    printCoef0 = true;
    printDegree = true;
  } else if (strncmp(kernelType, "rbf", 3) == 0) {
    printGamma = true;
  } else if (strncmp(kernelType, "sigmoid", 7) == 0) {
    printGamma = true;
    printCoef0 = true;
  }
	
	fprintf(outputFilePointer, "svm_type c_svc\n");
	fprintf(outputFilePointer, "kernel_type %s\n", kp.kernel_type.c_str());
  if (printDegree) {
    fprintf(outputFilePointer, "degree %i\n", kp.degree);
  }
  if (printGamma) {
    fprintf(outputFilePointer, "gamma %f\n", kp.gamma);
  }
  if (printCoef0) {
    fprintf(outputFilePointer, "coef0 %f\n", kp.coef0);
  }
	fprintf(outputFilePointer, "nr_class 2\n");
	fprintf(outputFilePointer, "total_sv %d\n", nSV + pSV);
	fprintf(outputFilePointer, "rho %.10f\n", kp.b);
	fprintf(outputFilePointer, "label 1 -1\n");
	fprintf(outputFilePointer, "nr_sv %d %d\n", pSV, nSV);
	fprintf(outputFilePointer, "SV\n");
	for (int i = 0; i < nPoints; i++) {
		if (alpha[i] > epsilon) {
			fprintf(outputFilePointer, "%.10f ", labels[i]*alpha[i]);
			for (int j = 0; j < nDimension; j++) {
				fprintf(outputFilePointer, "%d:%.10f ", j+1, data[j*nPoints + i]);
			}
			fprintf(outputFilePointer, "\n");
		}
	}
	fclose(outputFilePointer);
}

void printClassification(const char* outputFile, float* result, int nPoints) {
  FILE* outputFilePointer = fopen(outputFile, "w");
	if (outputFilePointer == NULL) {
		printf("Can't write %s\n", outputFile);
		exit(1);
	}
  for (int i = 0; i < nPoints; i++) {
    fprintf(outputFilePointer, "%f\n", result[i]);
  }
  fclose(outputFilePointer);
}
