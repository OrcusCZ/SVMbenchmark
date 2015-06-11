#include "orcusSVM_wrapper.h"
#include "OrcusSvm.h"
#include "../utils.h"

int OrcusSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type)
{
    svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = 32;
	req_data_format.vectAlignment = 32;
	req_data_format.transposed = false;
	req_data_format.labelsInFloat = true;
	req_data_format.supported_types = SUPPORTED_FORMAT_DENSE;

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, req_data_format));

    return SUCCESS;
}

int OrcusSvmModel::Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo)
{
    float * alpha = new float[data->GetNumVects()];
    float rho;

    try
    {
        OrcusSvmTrain(alpha, &rho,
            data->GetDataRawPointer(), (const float *)data->GetVectorLabelsPointer(),
            data->GetNumVects(), data->GetNumVectsAligned(),
            data->GetDimVects(), data->GetDimVectsAligned(),
            params->C, params->gamma, params->eps);
    }
    catch (std::exception & e)
    {
        std::cerr << "Exception in OrcusSvm: " << e.what() << std::endl;
    }

    delete[] alpha;
    return SUCCESS;
}

int OrcusSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type)
{
    return SUCCESS;
}