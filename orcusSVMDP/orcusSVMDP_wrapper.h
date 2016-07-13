#pragma once

#include "../svm_template.h"

class OrcusSvmDPData : public SvmData
{
public:
    int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
};

class OrcusSvmDPModel : public SvmModel
{
public:
    int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
};
