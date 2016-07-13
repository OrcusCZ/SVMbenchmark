#ifndef _GPULIBSVM_WRAPPER
#define _GPULIBSVM_WRAPPER

#include "svm_template.h"
#include "gpulibsvm.h"

class SvmData;
class SvmModel;

class GPULibSvmData : public SvmData {
private:
protected:
	struct gpulibsvm_problem *prob;

	void ConvertFromDenseData();
	//void ConvertFromCSRData();
public:
	GPULibSvmData();
	~GPULibSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
	int Delete();
	struct gpulibsvm_problem * GetProb();
};

class GPULibSvmModel : public SvmModel {
private:
	float * alphas;
	void ConvertParameters(struct svm_params * par_src, struct libsvm::svm_parameter * &par_dst);

protected:
	struct svm_params * params;
	struct gpulibsvm_model * model;

public:
	GPULibSvmModel();
	~GPULibSvmModel();

	int Train(SvmData * data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char * model_file_name, SVM_MODEL_FILE_TYPE type);
	int Delete();
};

#endif //_GPULIBSVM_WRAPPER