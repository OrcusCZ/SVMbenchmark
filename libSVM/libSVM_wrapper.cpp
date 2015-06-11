//#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <stdLib.h>
#include <string.h>
#include <cctype>

using namespace std;

#include "utils.h"
#include "libSVM_wrapper.h"
#include "libSVM_utils.h"

LibSvmData::LibSvmData() {
	printf("Using LibSVM...\n\n");
	prob = NULL;
}

LibSvmData::~LibSvmData() {
	Delete();
}

int LibSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
	
	svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = 1;
	req_data_format.vectAlignment = 1;
	req_data_format.transposed = false;
	req_data_format.labelsInFloat = false;
	req_data_format.supported_types = SUPPORTED_FORMAT_DENSE | SUPPORTED_FORMAT_CSR | SUPPORTED_FORMAT_BINARY;

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, req_data_format));

	if(this->type == DENSE) ConvertFromDenseData();
	else REPORT_ERROR("Unsuported format in LibSVM wrapper");

	return SUCCESS;
}

int LibSvmData::Delete() {
	if (prob == NULL) {
		return SUCCESS;
	}

	if (prob->x != NULL) {
		if (*(prob->x) != NULL) {
			free(*(prob->x));
		}
		free(prob->x);
		prob->x = NULL;
	}
	if (prob->y != NULL) {
		free(prob->y);
		prob->y = NULL;
	}

	free(prob);
	prob = NULL;

	return SUCCESS;
}

struct svm_problem * LibSvmData::GetProb() {
	return prob;
}

void LibSvmData::ConvertFromDenseData() {
	if(numVects == 0) REPORT_ERROR("No data loaded");
	Delete();
	MEM_SAFE_ALLOC(prob, struct svm_problem, 1);
	prob->l = numVects;
	
	//compute number of non-zeros:
	size_t elements = 0;
	for(unsigned int j=0; j < numVects; j++) {
		for(unsigned int i=0; i < dimVects; i++) {
			if(data_raw[j * dimVects + i] != 0.0f) elements++;
		}
		elements++; //terminal node index = -1
	}

	prob->y = Malloc(double,prob->l);
	prob->x = Malloc(struct svm_node *,prob->l);
	struct svm_node * x_space = Malloc(struct svm_node,elements);
	
	unsigned int n = 0;
	for(unsigned int j=0; j < numVects; j++) {
		prob->x[j] = &x_space[n];
		prob->y[j] = vector_labels[j];
		for(unsigned int i=0; i < dimVects; i++) {
			if(data_raw[j * dimVects + i] != 0.0f) {
				x_space[n].index = i;
				x_space[n].value = data_raw[j * dimVects + i];
				n++;
			}
		}
		x_space[n].index = -1;
		x_space[n].value = 0;
		n++;
	}
	
	//free dense data to save memory
	free(data_raw);
	data_raw = NULL;
}


/////////////////////////////////////////////////////////////
//MODEL
LibSvmModel::LibSvmModel() {
	model = NULL;
	params = NULL;
	alphas = NULL;
}

LibSvmModel::~LibSvmModel() {
	Delete();
}

int LibSvmModel::Delete() {
	if (model != NULL) {
		delete model;
		model = NULL;
	}
	
	if (params != NULL) {
		free(params);
		params = NULL;
	}

	if (alphas != NULL) {
		free(alphas);
		alphas = NULL;
	}

	return SUCCESS;
}

int LibSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {
	unsigned int numSVs;
	svm_parameter * libsvm_params;
	LibSvmData *data = (LibSvmData *) _data;
	params = _params;
	
	if (data == NULL) {
		return FAILURE;
	}
	numSVs = data->GetNumVects();

	if (alphas != NULL) {
		free(alphas);
	}
	alphas = (float *) malloc(numSVs * sizeof(float));

	ConvertParameters(params, libsvm_params);
	if (libsvm_params->gamma == 0) {
		libsvm_params->gamma = 1.0 / data->GetDimVects();
	}

	model = svm_train(data->GetProb(), libsvm_params);

	return SUCCESS;
}

void LibSvmModel::ConvertParameters(struct svm_params * par_src, struct svm_parameter * &par_dst) {
	par_dst = (struct svm_parameter *) malloc(sizeof(struct svm_parameter));

	par_dst->C = par_src->C;
	par_dst->eps = par_src->eps;	/* Stoping criteria */
	par_dst->kernel_type = par_src->kernel_type;
	par_dst->degree = par_src->degree;
	par_dst->gamma = par_src->gamma;	/* If this value is not set (= 0), than it is set to 1.0 / num_features. */
	par_dst->coef0 = par_src->coef0;
	par_dst->nu = par_src->nu;
	par_dst->p = par_src->p;		/* Regression parameter epsilon */

	/* Default LibSVM values. */
	par_dst->svm_type = C_SVC;
	par_dst->cache_size = 3072; /* This was originally 100. */
	par_dst->probability = 0;
	par_dst->shrinking = 1;
	par_dst->nr_weight = 0;
	par_dst->weight_label = NULL;
	par_dst->weight = NULL;
}

int LibSvmModel::StoreModel(char * model_file_name, SVM_MODEL_FILE_TYPE type) {
	if(type != M_LIBSVM_TXT) REPORT_ERROR("LIBSVM_TXT format only is supported to store the model");
	if (svm_save_model(model_file_name, model) == 0) {
		return SUCCESS;
	} else {
		return FAILURE;
	}
}

