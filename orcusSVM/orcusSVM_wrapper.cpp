#include "orcusSVM_wrapper.h"

int OrcusSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type)
{
    return 0;
}

int OrcusSvmModel::Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo)
{
    return 0;
}

int OrcusSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type)
{
    return 0;
}