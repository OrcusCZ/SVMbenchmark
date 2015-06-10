#ifndef _MULTISVM_WRAPPER_H_
#define _MULTISVM_WRAPPER_H_

#include "svm_template.h"
//#include "svmTrain.h"

//class SvmData;
//class SvmModel;

//struct multisvm_data {
//	int total_nPoints;
//	int dataDimension;
//	float *labels;
//	float *data;
//	float *data_transposed;
//};
//
//struct multisvm_model {
//	int nSV;
//	int nDimension;
//	float *alpha;
//	float *supportVectors;
//	float class1Label;
//	float class2Label;
//};

class MultiSvmData : public SvmData {
private:
protected:
public:
	MultiSvmData();
	//~MultiSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
};

class MultiSvmModel : public SvmModel {
private:
protected:
public:
	MultiSvmModel();
	//~MultiSvmModel();

	int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
};

#endif //_MULTISVM_WRAPPER_H_