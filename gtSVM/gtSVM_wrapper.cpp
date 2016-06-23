#include "utils.h"
#include "gtSVM_wrapper.h"
#include <limits>

#include "auto_context.hpp"

using namespace libsvm;


//from cu file
void trainclassifier (float* h_xtraindata, int* h_ltraindata, int* h_rdata, float* h_atraindata, int ntraining, int nfeatures, int nclasses, int ntasks, float* h_C, float* h_b, float tau, int kernelcode, float beta, float a, float b, float d);

GtSvmData::GtSvmData() {

}

GtSvmData::~GtSvmData() {
    Delete();
    SvmData::Delete();
}

int GtSvmData::Delete() {
    labels.clear();
    values.clear();
    indices.clear();
    offsets.clear();
    return SUCCESS;
}

int GtSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
    Delete();

    svm_memory_dataformat req_data_format;
    req_data_format.allocate_pinned = false;
    req_data_format.allocate_write_combined = false;
    req_data_format.dimAlignment = 1;
    req_data_format.vectAlignment = 1;
    req_data_format.transposed = false;
    req_data_format.labelsInFloat = false;
    req_data_format.supported_types = SUPPORTED_FORMAT_DENSE | SUPPORTED_FORMAT_CSR;

    SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

    switch(this->type) {
        case DENSE:
            ConvertFromDenseData();
            break;
        case SPARSE:
            ConvertFromCSRData();
            break;
        default:
            REPORT_ERROR("Unsuported format in gtSVM wrapper");
    }

    return SUCCESS;
}


int GtSvmData::ConvertFromDenseData() {
    if(numVects == 0) REPORT_ERROR("No data loaded");
    Delete();
    
    size_t n = 0;
    offsets.push_back(n);
    for(unsigned int j=0; j < numVects; j++) {
        labels.push_back(vector_labels[j]);
        for(unsigned int i=0; i < dimVects; i++) {
            float value = data_dense[j * dimVects + i];
            if(value != 0.0f) {
                values.push_back(value);
                indices.push_back(i);
                n++;
            }
        }
        offsets.push_back(n);
    }
    
    return SUCCESS;
}

int GtSvmData::ConvertFromCSRData() {
    if(numVects == 0 || data_csr == NULL) REPORT_ERROR("No data loaded");
    Delete();
    
    offsets.push_back(0);
    for(unsigned int j=0; j < numVects; j++) {
        offsets.push_back(data_csr->rowOffsets[j+1]);
        labels.push_back(vector_labels[j]);
    }
    for(unsigned int i=0; i < data_csr->nnz; i++) {
        values.push_back(data_csr->values[i]);
        indices.push_back(data_csr->colInd[i]);
    }

    return SUCCESS;
}

GtSvmModel::GtSvmModel(bool LargeOrSmallClusters) {
	this->LargeOrSmallClusters = LargeOrSmallClusters;
}

int GtSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {

    data = (GtSvmData *) _data;
    GtSvmData *gtdata = (GtSvmData *) _data;
    params = _params;

    if (data == NULL || params == NULL) {
        return FAILURE;
    }

    unsigned int const rows = data->GetNumVects();
    unsigned int const columns = data->GetDimVects();
    bool const multiclass = false;
    float const regularization = (float) params->C;
    bool const biased = true;

    float kernelParameter1=0;
    float kernelParameter2=0;
    float kernelParameter3=0;

    //kernelcode: 0 (RBF), 1(linear), 2(polynomial), 3(sigmoid)
    GTSVM_Kernel kernel;
    switch (params->kernel_type) {
    case RBF:
        kernel = GTSVM_KERNEL_GAUSSIAN;
        kernelParameter1 = (float)params->gamma;
        break;
    case POLY:
        kernel = GTSVM_KERNEL_POLYNOMIAL;
        kernelParameter1 = (float)params->gamma;
        kernelParameter2 = (float)params->coef0;
        kernelParameter3 = (float)params->degree;
        break;
    case SIGMOID:
        kernel = GTSVM_KERNEL_SIGMOID;
        kernelParameter1 = (float)params->gamma;
        kernelParameter2 = (float)params->coef0;
        break;
    default:
        REPORT_ERROR("GTSVM supports only folowing kernels: gaussian, polynomial and sigmoid");
    }

    if (
        GTSVM_InitializeSparse(
            context,
            &(gtdata->values[ 0 ]),
            &(gtdata->indices[ 0 ]),
            &(gtdata->offsets[ 0 ]),
            GTSVM_TYPE_FLOAT,
            &(gtdata->labels[ 0 ]),
            GTSVM_TYPE_INT32,
            rows,
            columns,
            false,
            multiclass,
            regularization,
            kernel,
            kernelParameter1,
            kernelParameter2,
            kernelParameter3,
            biased,
			LargeOrSmallClusters, // true/false = small/large clusters (4/8 vectors clusters)
            1
        )
    )
    {
        REPORT_ERROR( GTSVM_Error() );
    }

    unsigned int iterations = 1<<20; //max iterations - 1G - almost infinite
    unsigned int repetitions = 1<<8;
    for ( unsigned int ii = 0; ii < iterations; ii += repetitions ) {
        double primal =  std::numeric_limits< double >::infinity();
        double dual   = -std::numeric_limits< double >::infinity();
        if (
            GTSVM_Optimize(
                context,
                &primal,
                &dual,
                repetitions
            )
        )
        {
            REPORT_ERROR( GTSVM_Error() );
        }
        std::cout << "Iteration " << ( ii + 1 ) << '/' << iterations << ", primal = " << primal << ", dual = " << dual << std::endl;
        if ( 2 * ( primal - dual ) < params->eps * ( primal + dual ) )
            break;
    }

    //if (
    //    GTSVM_Shrink(
    //        context,
    //        smallClusters,
    //        activeClusters
    //    )
    //)
    //{
    //    REPORT_ERROR( GTSVM_Error() );
    //}
    
    MEM_SAFE_ALLOC(alphas, float, data->GetNumVects());

    GTSVM_GetBias(context, &(params->rho));
    params->rho = -params->rho;
    GTSVM_GetAlphas(context, alphas, GTSVM_TYPE_FLOAT, true);

    SAFE_CALL(CalculateSupperVectorCounts());


    return SUCCESS;
}

int GtSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) {
    return StoreModelGeneric(model_file_name, type);
}
