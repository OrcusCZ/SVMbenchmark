#ifndef _GPUSVM_WRAPPER_H_
#define _GPUSVM_WRAPPER_H_

#include "svm_template.h"
#include "svmTrain.h"

//class SvmData;
//class SvmModel;

//struct gpusvm_data {
//	int total_nPoints;
//	int dataDimension;
//	float *labels;
//	float *data;
//	float *data_transposed;
//};

struct gpusvm_model {
	int nSV;
	int nDimension;
	float *alpha;
	float *supportVectors;
	float class1Label;
	float class2Label;
};

class GpuSvmData : public SvmData {
private:
protected:
	//struct gpusvm_data *prob;
public:
	GpuSvmData();
	//~GpuSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
	//int Delete();
	//struct gpusvm_data *GetDataStruct();
};

class GpuSvmModel : public SvmModel {
private:
	struct Kernel_params kp;
	//void ExtractSupportVectors(struct gpusvm_data * data);
protected:
	//struct gpusvm_model* model;
public:
	GpuSvmModel();
	//~GpuSvmModel();

	int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
	//int Delete();
};

#endif //_GPUSVM_WRAPPER_H_