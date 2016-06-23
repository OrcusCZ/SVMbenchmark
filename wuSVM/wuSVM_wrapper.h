#ifndef _WUSVM_WRAPPER_H_
#define _WUSVM_WRAPPER_H_


#include "lasp_svm.h"
#include "svm_template.h"

class SvmData;
class SvmModel;
struct lasp::svm_sparse_data;
struct lasp::opt;
struct svm_params;

class WuSvmData : public SvmData {
private:
protected:
	//lasp::svm_sparse_data myData;
    lasp::svm_problem myProblem;

	int ConvertFromDenseData();
	int ConvertFromCSRData();

public:
	WuSvmData();
	~WuSvmData();
	int Delete();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);

	friend class WuSvmModel;
};



class WuSvmModel : public SvmModel {
private:
protected:
	lasp::svm_model myModel;
	bool use_single;
	bool use_pegasos;
	bool use_gpu;

	void ParseParams(struct svm_params * _params, struct lasp::opt &options);
public:
	WuSvmModel(bool use_single=false, bool use_pegasos=false, bool use_gpu=true);
	//~WuSvmModel();

	int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
};

#endif //_WUSVM_WRAPPER_H_
