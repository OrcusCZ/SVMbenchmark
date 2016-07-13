#include "orcusSVM1Block_wrapper.h"
#include "OrcusSvm.h"
#include "../utils.h"

extern int g_ws_size;

int OrcusSvm1BlockData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type)
{
    svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = 32;
	req_data_format.vectAlignment = 32;
	req_data_format.transposed = false;
	req_data_format.labelsInFloat = true;
    req_data_format.supported_types = SUPPORTED_FORMAT_DENSE | SUPPORTED_FORMAT_CSR;  //no sparse yet

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

    return SUCCESS;
}

int OrcusSvm1BlockModel::Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo)
{
    this->data = data;
    this->params = params;

    //float * alpha = new float[data->GetNumVects()];
    alphas = (float *)malloc(data->GetNumVects() * sizeof(float));
    float rho;

    try
    {
        bool is_sparse = data->GetDataType() == SVM_DATA_TYPE::SPARSE;
        //OrcusSvm1BlockTrain(alphas, &rho, is_sparse,
            //is_sparse ? (const float *)data->GetDataSparsePointer() : data->GetDataDensePointer(), (const float *)data->GetVectorLabelsPointer(),
            //data->GetNumVects(), data->GetNumVectsAligned(),
            //data->GetDimVects(), data->GetDimVectsAligned(),
            //params->C, params->gamma, params->eps);
        OrcusSvm1B::Data x;
        if (is_sparse)
            x.sparse = (OrcusSVM1B::csr *)data->GetDataSparsePointer();
        else
            x.dense = data->GetDataDensePointer();
        OrcusSvm1B::Train(alphas, &rho, is_sparse, x, (const float *)data->GetVectorLabelsPointer(),
            data->GetNumVects(), data->GetNumVectsAligned(),
            data->GetDimVects(), data->GetDimVectsAligned(),
            params->C, params->gamma, params->eps, g_ws_size);
    }
    catch (std::exception & e)
    {
        std::cerr << "Exception in OrcusSvm: " << e.what() << std::endl;
    }
    params->rho = rho;
    SAFE_CALL(CalculateSupperVectorCounts());

    //delete[] alpha;
    return SUCCESS;
}

int OrcusSvm1BlockModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type)
{
    return StoreModelGeneric(model_file_name, type);
}
