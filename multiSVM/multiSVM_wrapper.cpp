#include <utils.h>
#include <multiSVM_wrapper.h>

using namespace libsvm;

//from cu file
void trainclassifier (float* h_xtraindata, int* h_ltraindata, int* h_rdata, float* h_atraindata, int ntraining, int nfeatures, int nclasses, int ntasks, float* h_C, float* h_b, float tau, int kernelcode, float beta, float a, float b, float d);

MultiSvmData::MultiSvmData() {

}

int MultiSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
	Delete();

	svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = 1;
	req_data_format.vectAlignment = 1;
	req_data_format.transposed = true;
	req_data_format.labelsInFloat = false;
	req_data_format.supported_types = SUPPORTED_FORMAT_DENSE;

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

	return SUCCESS;
}

MultiSvmModel::MultiSvmModel() {

}


int MultiSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {

	/**
	 * Trains the multiclass Support Vector Machine
	 * @param h_xtraindata host pointer to the training set
	 * @param h_ltraindata host pointer to the labels of the training set
	 * @param h_rdata host pointer to the binary matrix that encodes the output code
	 * @param h_atraindata host pointer that will contain the values of the alphas
	 * @param ntraining number of training samples in the training set
	 * @param nfeatures number of features in each sample of the training set
	 * @param nclasses number of classes of the multiclass problem
	 * @param ntasks number of binary tasks to be solved
	 * @param h_C host pointer to the regularization parameters for each binary task
	 * @param h_b host pointer to the offset parameter of each binary task
	 * @param tau stopping parameter of the SMO algorithm
	 * @param kernelcode type of kernel to use
	 * @param beta if using RBF kernel, the value of beta
	 * @param a if using polynomial or sigmoid kernel the value of a x_i x_j
	 * @param b if using polynomial or sigmoid kernel the value of b

	trainclassifier			( 	h_xtraindata,
								h_ltraindata,
								h_rdata,
								h_atraindata,
								ntraining,
								nfeatures,
								nclasses,
								ntasks,
								h_C,
								h_b,
								tau,
								kernelcode,
								beta,
								a,
								b,
								d);
	*/

	data = (MultiSvmData *) _data;
	params = _params;

	if (data == NULL || params == NULL) {
		return FAILURE;
	}

	int rdata[2] = {1, -1};
	int nclasses = data->GetNumClasses();
	if(nclasses != 2) REPORT_ERROR("The benchmark supports 2-class problems only")
	int ntasks = 1;
	float C = (float) params->C;
	float rho = 0;
	float tau = (float) params->eps;

	//kernelcode: 0 (RBF), 1(linear), 2(polynomial), 3(sigmoid)
	int kernelcode;
	switch (params->kernel_type) {
	case RBF:
		kernelcode = 0;
		break;
	case LINEAR:
		kernelcode = 1;
		break;
	case POLY:
		kernelcode = 2;
		break;
	case SIGMOID:
		kernelcode = 3;
		break;
	default:
		REPORT_ERROR("Kernel not supported");
	}

	MEM_SAFE_ALLOC(alphas, float, data->GetNumVects());

	//modify labels to 1 and 2 in separate memory:
	int *tmp_labels=NULL;
	MEM_SAFE_ALLOC(tmp_labels, int, data->GetNumVects());
	int *p = data->GetVectorLabelsPointer();
	for(unsigned int i=0; i < data->GetNumVects(); i++) {
		tmp_labels[i] = (p[i] == 1)? 1 : 2;
	}

	trainclassifier			( 	data->GetDataDensePointer(),
								tmp_labels,
								rdata,
								alphas,
								data->GetNumVects(),
								data->GetDimVects(),
								nclasses,
								ntasks,
								&C,
								&rho,
								params->eps,
								kernelcode,
								(float)params->gamma,
								(float)params->coef0,
								(float)params->nu,
								(float)params->p
							);


	params->rho = -rho;
	free(tmp_labels);

	SAFE_CALL(CalculateSupperVectorCounts());


	return SUCCESS;
}

int MultiSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) {
	return StoreModelGeneric(model_file_name, type);
}