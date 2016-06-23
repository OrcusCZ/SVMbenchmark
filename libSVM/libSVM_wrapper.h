#ifndef _LIBSVM_WRAPPER
#define _LIBSVM_WRAPPER

#include "svm_template.h"
#include "svm.h"

using namespace libsvm;

class SvmData;
class SvmModel;

class LibSvmData : public SvmData {
private:
protected:
	struct svm_problem *prob;

	void ConvertFromDenseData();
	void ConvertFromCSRData();
public:
	LibSvmData();
	~LibSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
	int Delete();
	struct svm_problem * GetProb();
};

class LibSvmModel : public SvmModel {
private:
	float * alphas;
	void ConvertParameters(struct svm_params * par_src, struct svm_parameter * &par_dst);

protected:
	struct svm_params * params;
	struct svm_model * model;

public:
	LibSvmModel();
	~LibSvmModel();

	int Train(SvmData * data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char * model_file_name, SVM_MODEL_FILE_TYPE type);
	int Delete();
};

#endif //_LIBSVM_WRAPPER