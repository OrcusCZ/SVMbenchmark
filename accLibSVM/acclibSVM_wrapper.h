#ifndef _ACCLIBSVM_WRAPPER
#define _ACCLIBSVM_WRAPPER

#include "svm_template.h"
#include "acclibsvm.h"

class SvmData;
class SvmModel;

class AccLibSvmData : public SvmData {
private:
	int dimOffset;
protected:
	struct acclibsvm_problem *prob;

	void ConvertFromDenseData();
	void ConvertFromCSRData();
public:
	AccLibSvmData();
	~AccLibSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
	int Delete();
	struct acclibsvm_problem * GetProb();
	int GetDimOffset() {return dimOffset;}
};

class AccLibSvmModel : public SvmModel {
private:
	float * alphas;
	void ConvertParameters(struct svm_params * par_src, struct libsvm::svm_parameter * &par_dst);
	int dimOffset;

protected:
	struct svm_params * params;
	struct acclibsvm_model * model;

public:
	AccLibSvmModel();
	~AccLibSvmModel();

	int Train(SvmData * data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char * model_file_name, SVM_MODEL_FILE_TYPE type);
	int Delete();
};

#endif //_ACCLIBSVM_WRAPPER