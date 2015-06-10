#ifndef SVMIO
#define SVMIO

#include <stdio.h>
#include <stdlib.h>
/* Includes, stl */
#include <utility>
#include <map>
#include <string.h>
#include <iostream>
#include "svmCommon.h"


enum Parameters { p_gamma, p_coef0, p_degree, p_rho, p_kernel_type, p_svm_type, p_nr_class, p_total_sv, p_label, p_nr_sv, p_SV };


float readFloat(FILE* input);

int readModel(const char* filename, float** alpha, float** supportVectors, int* nSVOut, int* nDimOut, Kernel_params* kp, float* p_class1Label, float* p_class2Label);

int readSvm(const char* filename, float** p_data, float** p_labels, int* p_npoints, int* p_dimension, float** p_transposed_data = 0);

void printModel(const char* outputFile, Kernel_params kp, float* alpha, float* labels, float* data, int nPoints, int nDimension, float epsilon);

void printClassification(const char* outputFile, float* result, int nPoints);

#endif
