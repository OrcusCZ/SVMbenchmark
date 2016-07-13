//GPU LibSVM modeifications

//#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cctype>

using namespace std;

#include "utils.h"
#include "GPUlibSVM_wrapper.h"
#include "libSVM_utils.h"

extern int g_cache_size;

GPULibSvmData::GPULibSvmData() {
	//printf("Using LibSVM...\n\n");
	prob = NULL;
}

GPULibSvmData::~GPULibSvmData() {
	Delete();
}

int GPULibSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
	
	svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = 1;
	req_data_format.vectAlignment = 1;
	req_data_format.transposed = false;
	req_data_format.labelsInFloat = false;
	//req_data_format.supported_types = SUPPORTED_FORMAT_DENSE | SUPPORTED_FORMAT_CSR;
	req_data_format.supported_types = SUPPORTED_FORMAT_DENSE;

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

	switch(this->type) {
		case DENSE:
			ConvertFromDenseData();
			break;
		//case SPARSE:
		//	ConvertFromCSRData();
		//	break;
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

int GPULibSvmData::Delete() {
	if (prob == NULL) {
		return SUCCESS;
	}

	if (prob->x != NULL) {
		if (prob->x->values != NULL) {
			free(prob->x->values);
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

struct gpulibsvm_problem * GPULibSvmData::GetProb() {
	return prob;
}

void GPULibSvmData::ConvertFromDenseData() {
	if(numVects == 0) REPORT_ERROR("No data loaded");
	Delete();
	MEM_SAFE_ALLOC(prob, struct gpulibsvm_problem, 1);
	prob->l = numVects;
	
	////compute number of non-zeros:
	//size_t elements = 0;
	//for(unsigned int j=0; j < numVects; j++) {
	//	for(unsigned int i=0; i < dimVects; i++) {
	//		if(data_dense[j * dimVects + i] != 0.0f) elements++;
	//	}
	//	elements++; //terminal node index = -1
	//}

	prob->y = Malloc(double, prob->l);
	prob->x = Malloc(struct gpulibsvm_node, prob->l);
	//struct gpulibsvm_node * x_space = Malloc(struct gpulibsvm_node,elements);
	
	unsigned int n = 0;
	for(unsigned int j=0; j < numVects; j++) {
		prob->y[j] = this->vector_labels[j];
		prob->x[j].dim = dimVects;
		prob->x[j].values = Malloc(double, dimVects);
		for(unsigned int i=0; i < dimVects; i++) prob->x[j].values[i] = data_dense[j * dimVects + i];
	}
	
	//free dense data to save memory
	free(data_dense);
	data_dense = NULL;
} //GPULibSvmData::ConvertFromDenseData

//void GPULibSvmData::ConvertFromCSRData() {
//	if(numVects == 0 || data_csr == NULL) REPORT_ERROR("No data loaded");
//	Delete();
//	MEM_SAFE_ALLOC(prob, struct svm_problem, 1);
//	prob->l = numVects;
//	
//	prob->y = Malloc(double,prob->l);
//	prob->x = Malloc(struct svm_node *,prob->l);
//	struct svm_node * x_space = Malloc(struct svm_node,data_csr->nnz + numVects);
//	
//	unsigned int n = 0;
//	for(unsigned int j=0; j < numVects; j++) {
//		prob->x[j] = &x_space[n];
//		prob->y[j] = vector_labels[j];
//		for(unsigned int i=data_csr->rowOffsets[j]; i < data_csr->rowOffsets[j+1]; i++) {
//				x_space[n].index = data_csr->colInd[i];
//				x_space[n].value = data_csr->values[i];
//				n++;
//		}
//		x_space[n].index = -1;
//		x_space[n].value = 0;
//		n++;
//	}
//	
//	//free sparse data to save memory
//	free(data_csr->values);
//	free(data_csr->colInd);
//	free(data_csr->rowOffsets);
//	delete data_csr;
//	data_csr = NULL;
//} //GPULibSvmData::ConvertFromCSRData


/////////////////////////////////////////////////////////////
//MODEL
GPULibSvmModel::GPULibSvmModel() {
	model = NULL;
	params = NULL;
	alphas = NULL;
}

GPULibSvmModel::~GPULibSvmModel() {
	Delete();
}

int GPULibSvmModel::Delete() {
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

int GPULibSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {
	unsigned int numSVs;
	libsvm::svm_parameter * libsvm_params;
	GPULibSvmData *data = (GPULibSvmData *) _data;
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

	model = gpulibsvm_train(data->GetProb(), libsvm_params);

	return SUCCESS;
}

void GPULibSvmModel::ConvertParameters(struct svm_params * par_src, struct libsvm::svm_parameter * &par_dst) {
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
	par_dst->cache_size = g_cache_size > 0 ? g_cache_size : 3072; /* This was originally 100. */
	par_dst->probability = 0;
	par_dst->shrinking = 1;
	par_dst->nr_weight = 0;
	par_dst->weight_label = NULL;
	par_dst->weight = NULL;
}

int GPULibSvmModel::StoreModel(char * model_file_name, SVM_MODEL_FILE_TYPE type) {
	if(type != M_LIBSVM_TXT) REPORT_ERROR("LIBSVM_TXT format only is supported to store the model");
	if (gpulibsvm_save_model(model_file_name, model) == 0) {
		return SUCCESS;
	} else {
		return FAILURE;
	}
}

