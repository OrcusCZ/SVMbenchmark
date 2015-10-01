//#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <stdLib.h>
#include <cctype>

using namespace std;

#include "utils.h"
#include "gpuSVM_wrapper.h"
#include "svmTrain.h"
#include "svmIO.h"

GpuSvmData::GpuSvmData() {
	select_device(-1, 0, 0);
	//prob = NULL;
}

//GpuSvmData::~GpuSvmData() {
//	Delete();
//}

int GpuSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
	Delete();

	//MEM_SAFE_ALLOC(prob, struct gpusvm_data, 1);
	//prob->labels = NULL;
	//prob->data = NULL;
	//prob->data_transposed = NULL;

	/* Read data from file. */
	svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = 1;
	req_data_format.vectAlignment = 1;
	req_data_format.transposed = true;
	req_data_format.labelsInFloat = true;
	req_data_format.supported_types = SUPPORTED_FORMAT_DENSE;

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

	//load_data_dense(fid, prob->labels, prob->data, height, width, LOAD_FLAG_ALL_WC | LOAD_FLAG_TRANSPOSE | LOAD_FLAG_FILE_BUFFER);

	/* Store dimensions. */
	//prob->total_nPoints = this->numVects;
	//prob->dataDimension = this->dimVects;

	return SUCCESS;
}

//int GpuSvmData::Delete() {
//	//if (prob != NULL) {
//	//	MEM_SAFE_FREE(prob->data)
//	//	MEM_SAFE_FREE(prob->data_transposed)
//	//	MEM_SAFE_FREE(prob->labels)
//	//	free(prob);
//	//	prob = NULL;
//	//}
//
//	return SUCCESS;
//}

//unsigned int GpuSvmData::GetNumVects() {
//	return (unsigned int) prob->total_nPoints;
//}

//struct gpusvm_data *GpuSvmData::GetDataStruct() {
//	return prob;
//}

/////////////////////////////////////////////////////////////
//MODEL
GpuSvmModel::GpuSvmModel() {
	select_device(-1, 0, 0);
	//model = NULL;
}

//GpuSvmModel::~GpuSvmModel() {
//	Delete();
//}

//int GpuSvmModel::Delete() {
//	//if (model != NULL) {
//	//	MEM_SAFE_FREE(model->supportVectors)
//	//	MEM_SAFE_FREE(model->alpha)
//
//	//	free(model);
//	//	model = NULL;
//	//}
//
//	return SUCCESS;
//}

int GpuSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {

	data = (GpuSvmData *) _data;
	//struct gpusvm_data *dataStruct;
	params = _params;
	
	if (data == NULL || params == NULL) {
		return FAILURE;
	}

	//dataStruct = data->GetDataStruct();

	if (params->gamma == 0) {
		params->gamma = 1.0 / data->GetDimVects();
	}

	memset(&kp, 0, sizeof(struct Kernel_params));
	kp.coef0 = (float) params->coef0;
	kp.degree = params->degree;
	kp.gamma = (float) params->gamma;
	switch (params->kernel_type) {
	case LINEAR:
		kp.kernel_type = LINEAR_STRING;
		break;
	case POLY:
		kp.kernel_type = POLY_STRING;
		break;
	case RBF:
		kp.kernel_type = RBF_STRING;
		break;
	case SIGMOID:
		kp.kernel_type = SIGMOID_STRING;
		break;
	}

	//alphas allocated during the training (float*) malloc(nPoints*sizeof(floats))
	performTraining(data->GetDataDensePointer(), data->GetNumVects(), data->GetDimVects(),
		(float *)data->GetVectorLabelsPointer(), &alphas, &kp, (float) params->C, ADAPTIVE,
		(float) params->p, (float) params->eps, 0);

	params->rho = kp.b;

	SAFE_CALL(CalculateSupperVectorCounts());

	//ExtractSupportVectors(dataStruct);

	return SUCCESS;
}

int GpuSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) {
	return StoreModelGeneric(model_file_name, type);
}

//int GpuSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) {
	//unsigned int i,
	//	j,
	//	width,
	//	height;
	//float alpha,
	//	alpha_mult,
	//	value;
	//FILE *fid;
	//struct gpusvm_data *dataStruct = ((GpuSvmData *) data)->GetDataStruct();

	//FILE_SAFE_OPEN(fid, model_file_name, "w")

	//height = model->nSV;
	//width = model->nDimension;

	///* Print header. */
	//fprintf(fid, "svm_type c_svc\nkernel_type %s\n", kp.kernel_type.c_str());
	//switch (params->kernel_type) {
	//case POLY:
	//	fprintf(fid, "degree %d\n", params->degree);
	//case SIGMOID:
	//	fprintf(fid, "coef0 %g\n", params->coef0);
	//case RBF:
	//	fprintf(fid, "gamma %g\n", params->gamma);
	//	break;
	//}
	//fprintf(fid, "nr_class 2\ntotal_sv %d\n", model->nSV);

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
	//		value = model->supportVectors[i * width + j];
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

//	return SUCCESS;
//}

//void GpuSvmModel::ExtractSupportVectors(struct gpusvm_data * data) {
//	unsigned int i,
//		j,
//		k,
//		width,
//		height;
//	float alpha;
//
//	MEM_SAFE_ALLOC(model, struct gpusvm_model, 1)
//
//	model->alpha = NULL;
//	model->supportVectors = NULL;
//
//	params->nsv_class1 = 0;
//	params->nsv_class2 = 0;
//
//	height = data->total_nPoints;
//
//	/* Count support vectors for each class. */
//	for (i = 0, j = 0; i < height; i++) {
//		alpha = alphas[i] *= data->labels[i];
//		if (alpha != 0.0) {
//			if (alpha < 0.0) {
//				params->nsv_class1++;  /* class label -1 */
//			} else {
//				params->nsv_class2++;  /* class label +1 */
//			}
//		}
//	}
//
//	width = data->dataDimension;
//
//	model->nSV = params->nsv_class1 + params->nsv_class2;
//	model->nDimension = width;
//
//	MEM_SAFE_ALLOC(model->alpha, float, model->nSV)
//	MEM_SAFE_ALLOC(model->supportVectors, float, model->nSV * width)
//
//	/* Store support vectors and alphas. */
//	for (i = 0, j = 0; i < height; i++) {
//		if (alphas[i] != 0.0) {
//			for (k = 0; k < width; k++) {
//				/* Transpose training data. */
//				model->supportVectors[j * width + k] = data->data[k * height + i];
//			}
//			model->alpha[j++] = alphas[i];
//		}
//	}
//}
