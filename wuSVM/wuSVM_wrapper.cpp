#include "utils.h"
#include <numeric>

#include "svm.h"
#include "options.h"
#include "lasp_svm.h"
#include "fileIO.h"
#include "pegasos.h"
#include "wuSVM_wrapper.h"

using namespace lasp;

WuSvmData::WuSvmData() {

}

WuSvmData::~WuSvmData() {
    Delete();
    SvmData::Delete();
}

int WuSvmData::Delete() {
	//myData.numFeatures = 0;
	//myData.multiClass = false;
	//myData.numPoints = 0;

    return SUCCESS;
}

int WuSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
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
			delete this->data_dense;
			this->data_dense = NULL;
            break;
        case SPARSE:
            ConvertFromCSRData();
			delete this->data_csr->values;
			delete this->data_csr->rowOffsets;
			delete this->data_csr->colInd;
			delete this->data_csr;
			this->data_csr = NULL;
            break;
        default:
            REPORT_ERROR("Unsuported format in WuSvm wrapper");
    }

    return SUCCESS;
}


int WuSvmData::ConvertFromDenseData() {
    if(numVects == 0) REPORT_ERROR("No data loaded");
    Delete();
    
	//for (unsigned int k = 0; k < numClasses; k++) myData.orderSeen.push_back(class_labels[k]);
	
    //for(unsigned int j=0; j < numVects; j++) {
		//vector<svm_node> currentNodes;

		//for (unsigned int i = 0; i < dimVects; i++) {
			//svm_node newNode;
			//newNode.index = i + 1;
			//newNode.value = data_dense[j * dimVects_aligned + i];
			//currentNodes.push_back(newNode);
		//}
		//myData.allData[vector_labels[j]].push_back(currentNodes);
    //}
    
	//myData.numPoints = numVects;
	//myData.numFeatures = this->dimVects+1;
	//myData.multiClass = myData.allData.size() > 2;
    //

    myProblem.features = this->dimVects+1;
    myProblem.n = numVects;

    //vector<int> ind(numVects);
    //for (int i = 0; i < numVects; i++)
        //ind[i] = i;
    //random_shuffle(ind.begin(), ind.end());

	for (unsigned int k = 0; k < numClasses; k++) myProblem.classifications.push_back(class_labels[k]);
    myProblem.xS = new double [myProblem.features * myProblem.n];
    myProblem.y = new double [myProblem.n];
    memset(myProblem.xS, 0, myProblem.features * myProblem.n * sizeof(double));
    //for (size_t i = 0; i < dimVects * myProblem.n; i++)
        //myProblem.xS[i] = data_dense[i];
    for (size_t i = 0; i < myProblem.n; i++)
        for (size_t j = 0; j < dimVects; j++)
            myProblem.xS[i * myProblem.features + j] = data_dense[i * dimVects + j];
            //myProblem.xS[i * myProblem.features + j] = data_dense[ind[i] * dimVects + j];
    for (size_t i = 0; i < myProblem.n; i++)
        myProblem.y[i] = vector_labels[i];

    return SUCCESS;
}

int WuSvmData::ConvertFromCSRData() {
    if(numVects == 0 || data_csr == NULL) REPORT_ERROR("No data loaded");
    Delete();
    
	//for (unsigned int k = 0; k < numClasses; k++) myData.orderSeen.push_back(class_labels[k]);

	//for (unsigned int j = 0; j < numVects; j++) {
		//vector<svm_node> currentNodes;

		//for (unsigned int i = data_csr->rowOffsets[j]; i < data_csr->rowOffsets[j+1]; i++) {
			//svm_node newNode;
			//newNode.index = data_csr->colInd[i] + 1;
			//newNode.value = data_csr->values[i];
			//currentNodes.push_back(newNode);
		//}
		//myData.allData[vector_labels[j]].push_back(currentNodes);
	//}

	//myData.numPoints = numVects;
	//myData.numFeatures = this->dimVects + 1;
	//myData.multiClass = myData.allData.size() > 2;

    //return SUCCESS;
    return FAILURE;
}

WuSvmModel::WuSvmModel(bool use_single, bool use_pegasos, bool use_gpu) {
	this->use_single = use_single;
	this->use_pegasos = use_pegasos;
	this->use_gpu = use_gpu;
}

int WuSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {

	WuSvmData *data = (WuSvmData *)_data;

	//prepate params/options:
	opt options;
	ParseParams(_params, options);
    data->myProblem.options = options;

	//int firstClass = data->myData.orderSeen[0];
	//int secondClass = data->myData.orderSeen[1];
	//lasp::svm_problem curProblem = lasp::get_onevsone_subproblem(data->myData, firstClass, secondClass, options);

	if (options.single) {
		if (options.pegasos) {
			lasp::pegasos_svm_host<float>(data->myProblem);
		}
		else {
			lasp::lasp_svm_host<float>(data->myProblem);
		}

	}
	else {
		if (options.pegasos) {
			lasp::pegasos_svm_host<double>(data->myProblem);
		}
		else {
			lasp::lasp_svm_host<double>(data->myProblem);
		}
	}

	vector<lasp::svm_problem> solvedProblems;
	vector<lasp::svm_sparse_data> holdouts;
	solvedProblems.push_back(data->myProblem);

	//myModel = lasp::get_model_from_solved_problems(solvedProblems, holdouts, data->myData.orderSeen);
	myModel = lasp::get_model_from_solved_problems(solvedProblems, holdouts, data->myProblem.classifications);
	//lasp::write_model(myModel, options.modelFile);

    return SUCCESS;
}

int WuSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) {
	if (type != M_LIBSVM_TXT) return FAILURE;
	return lasp::write_model(myModel, model_file_name);
    }


void  WuSvmModel::ParseParams(struct svm_params * _params, struct lasp::opt &options) {

	//sets parameters to defaults
	options.nb_cand = 10;
	options.set_size = 5000;
	options.maxiter = 20;
	options.base_recomp = pow(2, 0.5);
	options.verb = 1;
	options.contigify = true;
	options.maxnewbasis = 800;
	options.candbufsize = 0;
	options.stoppingcriterion = 5e-6;
	options.maxcandbatch = 100;
	options.coef = 0;
	options.degree = 3;
	options.kernel = lasp::RBF;
	options.modelFile = "output.model";
	options.C = 1;
	options.gamma = -999; //we will set this later, but the defaut requires knowledge of the dataset.
	options.plattScale = 0;
	options.usegpu = this->use_gpu;
	options.randomize = false;
	options.single = this->use_single;
	options.pegasos = this->use_pegasos;
	options.usebias = false;
	options.shuffle = true;
	options.smallKernel = true;
	options.bias = 1;
	options.start_size = 100;
	options.maxGPUs = -1;
	options.stopIters = 1;

#ifdef _OPENMP
	//omp_set_num_threads(omp_get_num_procs()); //set by openMP as default
#endif

	//update to actual params:
	options.C = _params->C;
	options.gamma = _params->gamma;
	options.degree = _params->degree;
	options.coef = _params->coef0;

	switch(_params->kernel_type) {
	case libsvm::RBF:
		options.kernel = lasp::RBF;
		break;
	case  libsvm::LINEAR:
		options.kernel = lasp::LINEAR;
		break;
	case  libsvm::SIGMOID:
		options.kernel = lasp::SIGMOID;
		break;
	case  libsvm::POLY:
		options.kernel = lasp::POLYNOMIAL;
		break;
	case  libsvm::PRECOMPUTED:
		options.kernel = lasp::PRECOMPUTED;
		break;
	} //switch
	
} //ParseParams
