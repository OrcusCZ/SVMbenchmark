//#include <sys/time.h>   
#include <gettimeofday.h>
#include <stdio.h>   
#include <math.h>
#include <string.h>
#include <cutil.h>
#include <cuda.h>
#include <getopt.h>
#include <stdlib.h>

#include "svmCommon.h"
#include "../common/svmIO.h"
#include "../common/framework.h"
#include "kernelType.h"


void performTraining(float* data, int nPoints, int nDimension, float* labels, float** p_alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData);

void printHelp() {
  printf("Usage: svmTrain [options] trainingData.svm\n");
  printf("Options:\n");
  printf("\t-o outputFilename\t Location of output file\n");
  printf("Kernel types:\n");
  printf("\t--gaussian\tGaussian or RBF kernel (default): Phi(x, y; gamma) = exp{-gamma*||x-y||^2}\n");
  printf("\t--linear\tLinear kernel: Phi(x, y) = x . y\n");
  printf("\t--polynomial\tPolynomial kernel: Phi(x, y; a, r, d) = (ax . y + r)^d\n");
  printf("\t--sigmoid\tSigmoid kernel: Phi(x, y; a, r) = tanh(ax . y + r)\n");
  printf("Parameters:\n");
  printf("\t-c, --cost\tSVM training cost C (default = 10)\n");
  printf("\t-g\tGamma for Gaussian kernel (default = 1/l)\n");
  printf("\t-a\tParameter a for Polynomial and Sigmoid kernels (default = 1/l)\n");
  printf("\t-r\tParameter r for Polynomial and Sigmoid kernels (default = 1)\n");
  printf("\t-d\tParameter d for Polynomial kernel (default = 3)\n");
  printf("Convergence parameters:\n");
  printf("\t--tolerance, -t\tTermination criterion tolerance (default = 0.001)\n");
  printf("\t--epsilon, -e\tSupport vector threshold (default = 1e-5)\n");
  printf("Internal options:\n");
  printf("\t--heuristic, -h\tWorking selection heuristic:\n");
  printf("\t\t0: First order\n");
  printf("\t\t1: Second order\n");
  printf("\t\t2: Random (either first or second order)\n");
  printf("\t\t3: Adaptive (default)\n");
  
}


static int kType = GAUSSIAN;

int main( const int argc, const char** argv)  { 
  int currentOption;
  float parameterA = -0.125f;
  float parameterB = 1.0f;
  float parameterC = 3.0f;

  bool parameterASet = false;
  bool parameterBSet = false;
  bool parameterCSet = false;
  
  
  SelectionHeuristic heuristicMethod = ADAPTIVE;
  float cost = 10.0f;
  
  float tolerance = 1e-3f;
  float epsilon = 1e-5f;
  char* outputFilename = NULL;
  while (1) {
    static struct option longOptions[] = {
      {"gaussian", no_argument, &kType, GAUSSIAN},
      {"polynomial", no_argument, &kType, POLYNOMIAL},
      {"sigmoid", no_argument, &kType, SIGMOID},
      {"linear", no_argument, &kType, LINEAR},
      {"cost", required_argument, 0, 'c'},
      {"heuristic", required_argument, 0, 'h'},
      {"tolerance", required_argument, 0, 't'},
      {"epsilon", required_argument, 0, 'e'},
      {"output", required_argument, 0, 'o'},
      {"version", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'f'}
    };
    int optionIndex = 0;
    currentOption = getopt_long(argc, (char *const*)argv, "c:h:t:e:o:a:r:d:g:v:f", longOptions, &optionIndex);
    if (currentOption == -1) {
      break;
    }
    int method = 3;
    switch (currentOption) {
    case 0:
      break;
    case 'v':
      printf("GPUSVM version %1.1f\n", VERSION);
      return(0);
    case 'f':
      printHelp();
      return(0);
    case 'c':
      sscanf(optarg, "%f", &cost);
      break;
    case 'h':
      sscanf(optarg, "%i", &method);
      switch (method) {
      case 0:
        heuristicMethod = FIRSTORDER;
        break;
      case 1:
        heuristicMethod = SECONDORDER;
        break;
      case 2:
        heuristicMethod = RANDOM;
        break;
      case 3:
        heuristicMethod = ADAPTIVE;
        break;
      }
      break;
    case 't':
      sscanf(optarg, "%f", &tolerance);
      break;
    case 'e':
      sscanf(optarg, "%e", &epsilon);
      break;
    case 'o':
      outputFilename = (char*)malloc(strlen(optarg));
      strcpy(outputFilename, optarg);
      break;
    case 'a':
      sscanf(optarg, "%f", &parameterA);
      parameterASet = true;
      break;
    case 'r':
      sscanf(optarg, "%f", &parameterB);
      parameterBSet = true;
      break;
    case 'd':
      sscanf(optarg, "%f", &parameterC);
      parameterCSet = true;
      break;
    case 'g':
      sscanf(optarg, "%f", &parameterA);
      parameterA = -parameterA;
      parameterASet = true;
      break;
    case '?':
      break;
    default:
      abort();
      break;
    }
  }

  if (optind != argc - 1) {
    printHelp();
    return(0);
	}

  const char* trainingFilename = argv[optind];
  
  if (outputFilename == NULL) {
    int inputNameLength = strlen(trainingFilename);
    outputFilename = (char*)malloc(sizeof(char)*(inputNameLength + 5));
    strncpy(outputFilename, trainingFilename, inputNameLength + 4);
    char* period = strrchr(outputFilename, '.');
    if (period == NULL) {
      period = outputFilename + inputNameLength;
    }
    strncpy(period, ".mdl\0", 5);
  }
  
	
	int nPoints;
	int nDimension;
	float* data;
	float* transposedData;
	float* labels;
	readSvm(trainingFilename, &data, &labels, &nPoints, &nDimension, &transposedData);
	printf("Input data found: %d points, %d dimension\n", nPoints, nDimension);
  
  float* alpha;
  Kernel_params kp;
  if (kType == LINEAR) {
    printf("Linear kernel\n");
    //kp.kernel_type = "linear";
  } else if (kType == POLYNOMIAL) {
    if (!(parameterCSet)) {
      parameterC = 3.0f;
    }
    if (!(parameterASet)) {
      parameterA = 1.0/nPoints;
    }
    if (!(parameterBSet)) {
      parameterB = 0.0f;
    }
    //printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
    if ((parameterA <= 0) || (parameterB < 0) || (parameterC < 1.0)) {
      printf("Invalid parameters\n");
      exit(1);
    }
    kp.kernel_type = "polynomial";
    kp.gamma = parameterA;
    kp.coef0 = parameterB;
    kp.degree = (int)parameterC;
  } else if (kType == GAUSSIAN) {
    if (!(parameterASet)) {
      parameterA = 1.0/nPoints;
    } else {
      parameterA = -parameterA;
    }
    //printf("Gaussian kernel: gamma = %f\n", parameterA);
    if (parameterA < 0) {
      printf("Invalid parameters\n");
      exit(1);
    }
    kp.kernel_type = "rbf";
    kp.gamma = parameterA;
  } else if (kType == SIGMOID) {
    if (!(parameterASet)) {
      parameterA = 1.0/nPoints;
    }
    if (!(parameterBSet)) {
      parameterB = 0.0f;
    }
    //printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
    if ((parameterA <= 0) || (parameterB < 0)) {
      printf("Invalid Parameters\n");
      exit(1);
    }
    kp.kernel_type = "sigmoid";
    kp.gamma = parameterA;
    kp.coef0 = parameterB;
  }

	struct timeval start;
  gettimeofday(&start, 0);
  performTraining(data, nPoints, nDimension, labels, &alpha, &kp, cost, heuristicMethod, epsilon, tolerance, transposedData);

  struct timeval finish;
	gettimeofday(&finish, 0);
  float trainingTime = (float)(finish.tv_sec - start.tv_sec) + ((float)(finish.tv_usec - start.tv_usec)) * 1e-6;
	
	printf("Training time : %f seconds\n", trainingTime);
	printModel(outputFilename, kp, alpha, labels, data, nPoints, nDimension, epsilon);
	
}

