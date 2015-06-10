//#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <stdLib.h>
#include <cctype>

using namespace std;

#include "utils.h"
#include "cudaSVM_wrapper.h"
#include "cudasvm_train.h"

CudaSvmData::CudaSvmData() {
	select_device(-1, 13, 0);
	prob = NULL;
}

CudaSvmData::~CudaSvmData() {
	Delete();
}

int CudaSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
	FILE *fid;

	Delete();

	float * labels_tmp = NULL;

	FILE_SAFE_OPEN(fid, filename, "r")

	MEM_SAFE_ALLOC(prob, struct cusvm_prob, 1)
	prob->data = NULL;
	prob->labels = NULL;
	prob->result = NULL;
	prob->b = NULL;
	prob->_ddata = NULL;
	prob->_dlabels = NULL;
	prob->_dresult = NULL;
	prob->_db = NULL;

	/* Read data from file. */
	//load_data_dense(fid, labels_tmp, prob->data, prob->nof_vectors, prob->width,
	//	LOAD_FLAG_ALL_WC | LOAD_FLAG_FILE_BUFFER);

	//DEBUG: not finished
	return -1;

	malloc_host_WC((void **) & (prob->labels), prob->nof_vectors * sizeof(int));

	for (unsigned int i = 0; i < prob->nof_vectors; i++) {
		prob->labels[i] = (int) labels_tmp[i];
	}

	free_host(labels_tmp);
	/* Close streams. */
	fclose(fid);

	return SUCCESS;
}

int CudaSvmData::Delete() {
	if (prob != NULL) {
		cusvm_destroy_data(prob);
	}

	return SUCCESS;
}

int CudaSvmData::GetClassLabel(unsigned int i) {
	if (prob != NULL && prob->labels != NULL && i < prob->nof_vectors) {
		return (int) prob->labels[i];
	} else {
		return 0;
	}
}

float *CudaSvmData::GetSVs() {
	if (prob != NULL && prob->_ddata != NULL) {
		return prob->_ddata;
	} else {
		return NULL;
	}
}

unsigned int CudaSvmData::GetNumVects() {
	return prob->nof_vectors;
}

struct cusvm_prob *CudaSvmData::GetDataStruct() {
	return prob;
}

/////////////////////////////////////////////////////////////
//MODEL
CudaSvmModel::CudaSvmModel() {
	select_device(-1, 13, 0);
	model = NULL;
	results = NULL;
}

CudaSvmModel::~CudaSvmModel() {
	Delete();
}

int CudaSvmModel::Delete() {
	if (model != NULL) {
		cusvm_destroy_model(model);
		model = NULL;
	}

	if (results != NULL) {
		free(results);
		results = NULL;
	}

	return SUCCESS;
}

#include <cuda_runtime.h>
int CudaSvmModel::Train(SvmData * _data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {
	unsigned int i,
		j,
		k;
	CudaSvmData *data = (CudaSvmData *) _data;
	struct cusvm_prob * training_set = data->GetDataStruct();

	params = _params;

	// Check whether gamma has been set via command line argument.
	if (params->gamma == 0.0) {
		params->gamma = 1.0 / training_set->width;
	}

	/* Perform training. */
	cudasvm_train(training_set, params);

	/* Prepare model structure */
	MEM_SAFE_ALLOC(model, struct cusvm_model, 1)

	model->alpha = NULL;
	model->vector = NULL;
	model->_dalphas = NULL;
	model->_dvectors = NULL;

	model->kernel_param.coef0 = params->coef0;
	model->kernel_param.degree = params->degree;
	model->kernel_param.gamma = params->gamma;
	model->kernel_param.kernel_type = params->kernel_type;
	model->kernel_param.rho = params->rho;
	model->kernel_param.svm_type = C_SVC;

	params->nsv_class1 = 0;
	params->nsv_class2 = 0;

	model->m_vector_len = training_set->width;
	model->m_nof_vectors = 0;

	/* Count number of support vectors. */
	for (i = 0; i < training_set->nof_vectors; i++) {
		if (training_set->result[i] > params->eps) {
			model->m_nof_vectors++;
		}
	}

	/* Allocate support vectors and vector alpha. */
	MEM_SAFE_ALLOC(model->vector, float, model->m_nof_vectors * model->m_vector_len)
	MEM_SAFE_ALLOC(model->alpha, float, model->m_nof_vectors)

	/* Fill vector alpha and support vectors */
	for (i = 0, j = 0; i < training_set->nof_vectors; i++) {
		if (training_set->result[i] > params->eps) {
			for (k = 0; k < model->m_vector_len; k++) {
				model->vector[j * model->m_vector_len + k] = training_set->data[i * training_set->width + k];
			}
			model->alpha[j] = (float) (training_set->result[i] * training_set->labels[i]);
			if (training_set->labels[i] < 0.0F) {
				params->nsv_class1++;
			} else {
				params->nsv_class2++;
			}
			++j;
		}
	}

	return SUCCESS;
}

int CudaSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) {
	//unsigned int i,
	//	j,
	//	width,
	//	height;
	//float alpha,
	//	value,
	//	alpha_mult;
	//FILE *fid;
	//struct cusvm_prob *dataStruct = ((CudaSvmData *) data)->GetDataStruct();

	//if (model == NULL) {
	//	return FAILURE;
	//}

	//FILE_SAFE_OPEN(fid, model_file_name, "w")

	//height = model->m_nof_vectors;
	//width = model->m_vector_len;

	///* Print header. */
	//fprintf(fid, "svm_type c_svc\nkernel_type %s\n", "rbf");
	//switch (params->kernel_type) {
	//case POLY:
	//	fprintf(fid, "degree %d\n", params->degree);
	//case SIGMOID:
	//	fprintf(fid, "coef0 %g\n", params->coef0);
	//case RBF:
	//	fprintf(fid, "gamma %g\n", params->gamma);
	//	break;
	//}
	//fprintf(fid, "nr_class 2\ntotal_sv %d\n", height);

	///* Print labels and counts. */
	//if (model->alpha[0] < 0.f) { // If the first label is negative
	//	fprintf(fid, "rho %g\nlabel -1 1\nnr_sv %d %d\nSV\n",
	//		-params->rho, params->nsv_class1, params->nsv_class2);
	//	alpha_mult = -1.f;
	//} else {
	//	fprintf(fid, "rho %g\nlabel 1 -1\nnr_sv %d %d\nSV\n",
	//		params->rho, params->nsv_class2, params->nsv_class1);
	//	alpha_mult = 1.f;
	//}

	///* Print Support Vectors */
	//for (i = 0; i < height; i++) {
	//	alpha = alpha_mult * model->alpha[i];
	//	fprintf(fid, "%g ", alpha);

	//	for (j = 0; j < width; j++) {
	//		value = model->vector[i * width + j];
	//		if (value != 0.0F) {
	//			if (value == 1.0F) {
	//				fprintf(fid, "%d:1 ", j + 1);
	//			} else {
	//				fprintf(fid, "%d:%g ", j + 1, value);
	//			}
	//		}
	//	}

	//	fprintf(fid, "\n");
	//}

	//fclose(fid);

	return SUCCESS;
}