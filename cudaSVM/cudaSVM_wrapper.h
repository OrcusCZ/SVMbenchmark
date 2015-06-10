#ifndef _CUDASVM_WRAPPER_H_
#define _CUDASVM_WRAPPER_H_

#include "svm_template.h"

class SvmData;
class SvmModel;

class CudaSvmData : public SvmData {
private:
protected:
	struct cusvm_prob * prob;
public:
	CudaSvmData();
	~CudaSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
	int Delete();
	unsigned int GetNumVects();
	int GetClassLabel(unsigned int i);
	float * GetSVs();
	struct cusvm_prob * GetDataStruct();
};

class CudaSvmModel : public SvmModel {
private:
	int * results;
protected:
	struct cusvm_model * model;
	struct svm_params * params;
public:
	CudaSvmModel();
	~CudaSvmModel();

	int Train(SvmData * data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
	int Delete();
};

#endif //_CUDASVM_WRAPPER_H_