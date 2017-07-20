#include "ohdSVM_wrapper.h"
#include "OHD-SVM/ohdSVM.h"
#include "utils.h"
#include <string>

extern int g_ws_size;
extern std::string g_imp_spec_arg;

int ohdSVMData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type)
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

int ohdSVMModel::Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo)
{
    this->data = data;
    this->params = params;

    alphas = (float *)malloc(data->GetNumVects() * sizeof(float));
	float rho = 0;

    try
    {
		size_t pos = g_imp_spec_arg.find(',');
		if (pos != std::string::npos)
		{
			int sliceSize = atoi(g_imp_spec_arg.c_str());
			int threadsPerRow = atoi(g_imp_spec_arg.c_str() + pos + 1);
			ohdSVM::useEllRT(true, sliceSize, threadsPerRow);
		}

        bool is_sparse = data->GetDataType() == SVM_DATA_TYPE::SPARSE;
		ohdSVM::Data x;
        if (is_sparse)
            x.sparse = (ohdSVM::csr *)data->GetDataSparsePointer();
        else
            x.dense = data->GetDataDensePointer();
		ohdSVM::Train(alphas, &rho, is_sparse, x, (const float *)data->GetVectorLabelsPointer(),
            data->GetNumVects(), data->GetNumVectsAligned(),
            data->GetDimVects(), data->GetDimVectsAligned(),
            params->C, params->gamma, params->eps, g_ws_size);
    }
    catch (std::exception & e)
    {
        std::cerr << "Exception in OHD-SVM: " << e.what() << std::endl;
    }
    params->rho = rho;
    SAFE_CALL(CalculateSupperVectorCounts());

    return SUCCESS;
}

int ohdSVMModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type)
{
    return StoreModelGeneric(model_file_name, type);
}
