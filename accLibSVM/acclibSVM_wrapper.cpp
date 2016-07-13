//ACCLibSVM modifications

//#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cctype>

using namespace std;

#include "utils.h"
#include "acclibSVM_wrapper.h"
#include "libSVM_utils.h"

#ifdef USE_SSE
#ifdef USE_AVX
#define ACCLIBSVM_DIM_ALIGNMENT 8
#else
#define ACCLIBSVM_DIM_ALIGNMENT 4
#endif
#else
#define ACCLIBSVM_DIM_ALIGNMENT 1
#endif



AccLibSvmData::AccLibSvmData() {
	//printf("Using LibSVM...\n\n");
	prob = NULL;
	dimOffset=0;
}

AccLibSvmData::~AccLibSvmData() {
	Delete();
}

int AccLibSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
	
	svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = ACCLIBSVM_DIM_ALIGNMENT; //dim alignment for SSE/AVX intrinsics
	req_data_format.vectAlignment = 1;
	req_data_format.transposed = false;
	req_data_format.labelsInFloat = false;
	req_data_format.supported_types = SUPPORTED_FORMAT_DENSE | SUPPORTED_FORMAT_CSR;
	//req_data_format.supported_types = SUPPORTED_FORMAT_DENSE;

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

	switch(this->type) {
		case DENSE:
			ConvertFromDenseData();
			break;
		case SPARSE:
			ConvertFromCSRData();
			break;
		default:
			REPORT_ERROR("Unsuported format in LibSVM wrapper");
	}

	//LLLLLLLLLLLLLLLLLLLLLLL
	/*FILE *fid = fopen("xxx_data.txt", "w");
	unsigned int n = 0;
	for(int i = 0; i < prob->l; i++) {
		fprintf(fid, "%+.0f", prob->y[i]);
		svm_node *p = prob->x[i];
		while(p->index >= 0) {
			fprintf(fid, " %d:%.0f", p->index+1, p->value);
			p++;
		}
		fprintf(fid, "\n"); 
	}
	fclose(fid);*/
	//EEEEEEEEEEEEEEEEEEEEEEE

	return SUCCESS;
}

int AccLibSvmData::Delete() {
	if (prob == NULL) {
		return SUCCESS;
	}

	if (prob->x != NULL) {
		//prob->x->values //reference only, don't delete here
		//prob->x->ind //reference only, don't delete here
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

struct acclibsvm_problem * AccLibSvmData::GetProb() {
	return prob;
}

void AccLibSvmData::ConvertFromDenseData() {
	if(numVects == 0) REPORT_ERROR("No data loaded");
	Delete();
	MEM_SAFE_ALLOC(prob, struct acclibsvm_problem, 1);
	prob->l = numVects;
	
	prob->y = Malloc(double, prob->l);
	prob->x = Malloc(struct acclibsvm_node, prob->l);
	
	unsigned int n = 0;
	for(unsigned int j=0; j < numVects; j++) {
		prob->y[j] = this->vector_labels[j];
		prob->x[j].num = dimVects_aligned;
		prob->x[j].values = data_dense + j * dimVects_aligned;
		prob->x[j].ind = NULL;
	}
	dimOffset=1;
	
} //AccLibSvmData::ConvertFromDenseData

void AccLibSvmData::ConvertFromCSRData() {
	if(numVects == 0 || data_csr == NULL) REPORT_ERROR("No data loaded");
	Delete();
	MEM_SAFE_ALLOC(prob, struct acclibsvm_problem, 1);
	prob->l = numVects;
	
	prob->y = Malloc(double,prob->l);
	prob->x = Malloc(struct acclibsvm_node,prob->l);
	
	unsigned int n = 0;
	for(unsigned int j=0; j < numVects; j++) {
		prob->y[j] = this->vector_labels[j];
		prob->x[j].num = data_csr->rowOffsets[j+1] - data_csr->rowOffsets[j];
		prob->x[j].values = data_csr->values + data_csr->rowOffsets[j];
		prob->x[j].ind = data_csr->colInd + data_csr->rowOffsets[j];
		if(prob->x[j].ind[0] == 0) dimOffset=1; //for libsvm compatible output model, dim index from one
	}
	
} //AccLibSvmData::ConvertFromCSRData


/////////////////////////////////////////////////////////////
//MODEL
AccLibSvmModel::AccLibSvmModel() {
	model = NULL;
	params = NULL;
	alphas = NULL;
	dimOffset = 0;
}

AccLibSvmModel::~AccLibSvmModel() {
	Delete();
}

int AccLibSvmModel::Delete() {
	if (model != NULL) {
		delete model;
		model = NULL;
	}
	
	//if (params != NULL) {
	//	free(params);
	//	params = NULL;
	//}

	if (alphas != NULL) {
		free(alphas);
		alphas = NULL;
	}

	return SUCCESS;
}

int AccLibSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {
	unsigned int numSVs;
	libsvm::svm_parameter * libsvm_params;
	AccLibSvmData *data = (AccLibSvmData *) _data;
	params = _params;
	dimOffset = data->GetDimOffset();
	
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

	model = acclibsvm_train(data->GetProb(), libsvm_params);

	return SUCCESS;
}

void AccLibSvmModel::ConvertParameters(struct svm_params * par_src, struct libsvm::svm_parameter * &par_dst) {
	par_dst = (struct libsvm::svm_parameter *) malloc(sizeof(struct libsvm::svm_parameter));

	par_dst->C = par_src->C;
	par_dst->eps = par_src->eps;	/* Stoping criteria */
	par_dst->kernel_type = par_src->kernel_type;
	par_dst->degree = par_src->degree;
	par_dst->gamma = par_src->gamma;	/* If this value is not set (= 0), than it is set to 1.0 / num_features. */
	par_dst->coef0 = par_src->coef0;
	par_dst->nu = par_src->nu;
	par_dst->p = par_src->p;		/* Regression parameter epsilon */

	/* Default LibSVM values. */
	par_dst->svm_type = libsvm::C_SVC;
	par_dst->cache_size = 3072; /* This was originally 100. */
	par_dst->probability = 0;
	par_dst->shrinking = 1;
	par_dst->nr_weight = 0;
	par_dst->weight_label = NULL;
	par_dst->weight = NULL;
}

int AccLibSvmModel::StoreModel(char * model_file_name, SVM_MODEL_FILE_TYPE type) {
	if(type != M_LIBSVM_TXT) REPORT_ERROR("LIBSVM_TXT format only is supported to store the model");
	if (acclibsvm_save_model(model_file_name, model, dimOffset) == 0) {
		return SUCCESS;
	} else {
		return FAILURE;
	}
}

