#ifndef _CUSVM_WRAPPER_H_
#define _CUSVM_WRAPPER_H_

#include "svm_template.h"

#define MAX_SUPPORTED_SM 55	/* max CU level with correct # corres calculation */

/* Train */
/**
  * mexalpha     Output alpha values.
  * beta         Calculated SVM threshold (rho).
  * y            Input labels.
  * x            input matrix of training vectors (transposed).
  * C            Training parameter C.
  * kernelwidth  The parameter gamma.
  * eps          Thre regression parameter epsilon.
  * m            Number of rows (# training vectors).
  * n            Number of columns (width).
  * StoppingCrit Stopping criteria.
  */
extern "C" void SVRTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float _kernelwidth, float eps, int m, int n, float StoppingCrit);

extern "C" void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float kernelwidth, int m, int n, float StoppingCrit);


/*paddedm = (m & 0xFFFFFFE0) + ((m & 0x1F) ? 0x20 : 0);*/
/*	int ceiled_pm_ni = (paddedm + NecIterations - 1) / NecIterations;
	int RowsPerIter = (ceiled_pm_ni & 0xFFFFFFE0) + ((ceiled_pm_ni & 0x1F) ? 0x20 : 0);*/

//struct cuSVM_model {
//	unsigned int nof_vectors;
//	unsigned int wof_vectors;
//	float lambda;
//	float beta;
//	float *alphas;
//	float *vectors;
//};

class SvmData;
class SvmModel;

class CuSvmData : public SvmData {
private:
protected:
public:
	CuSvmData();
	//~CuSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
	//int Delete();
};

class CuSvmModel : public SvmModel {
private:
	//float * alphas;
	//void ExtractSupportVectors(struct cuSVM_data * dataStruct);
protected:
	//struct cuSVM_model * model;
	//struct svm_params * params;
public:
	CuSvmModel();
	//~CuSvmModel();

	int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
	//int Delete();
};

#endif //_CUSVM_WRAPPER_H_