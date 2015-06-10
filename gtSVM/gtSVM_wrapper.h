#ifndef _GTSVM_WRAPPER_H_
#define _GTSVM_WRAPPER_H_

#include <iostream>
#include <vector>

#include "auto_context.hpp"
#include "svm_template.h"
//#include "svmTrain.h"

class SvmData;
class SvmModel;

class GtSvmData : public SvmData {
private:
protected:
	std::vector<int> labels;
	std::vector<float> values;
	std::vector<size_t> indices;
	std::vector<size_t> offsets;

	int ConvertFromDenseData();
public:
	GtSvmData();
	~GtSvmData();
	int Delete();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);

	friend class GtSvmModel;
};



class GtSvmModel : public SvmModel {
private:
protected:
	AutoContext context;
public:
	GtSvmModel();
	//~GtSvmModel();

	int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
};

#endif //_GTSVM_WRAPPER_H_