#ifndef _SVM_TEMPLATE
#define _SVM_TEMPLATE

#include <stdio.h>

struct svm_params {
	/* Training algorithm parameters: */
	double eps;	/* Stopping criteria. */
	double C;

	/* Kernel parameters: */
	int kernel_type;
	int degree;
	double gamma;
	double coef0;
	double nu;
	double p;

	/* Output parameters: */
	double rho; /* beta */
	unsigned int nsv_class1; /* Number of support vectors for the class with label -1. */
	unsigned int nsv_class2; /* Number of support vectors for the class with label +1. */
};

struct svm_trainingInfo {
	unsigned int numIters;
	float elTime1; //el time without init, deinit and CPU-GPU data transfer
	float elTime2; //total of pure kernels time
};

#define SUPPORTED_FORMAT_DENSE 1
#define SUPPORTED_FORMAT_CSR 2
#define SUPPORTED_FORMAT_BINARY 4
struct svm_memory_dataformat {
	unsigned int supported_types; //bit mask: &1 - dense, &2 - CSR, &4 - binary
	bool transposed;
	bool allocate_write_combined;
	bool allocate_pinned;
	bool labelsInFloat;
	unsigned int dimAlignment;
	unsigned int vectAlignment;
};

enum SVM_DATA_TYPE {UNKNOWN, DENSE, SPARSE, BINARY};
enum SVM_FILE_TYPE {LIBSVM_TXT};
enum SVM_MODEL_FILE_TYPE {M_LIBSVM_TXT};

class SvmData {
private:
protected:
	SVM_DATA_TYPE type;
	unsigned int numVects;
	unsigned int numVects_aligned;
	unsigned int dimVects;
	unsigned int dimVects_aligned;
	unsigned int numClasses;
	float *data_raw;
	int *class_labels;
	int *vector_labels;
	bool allocatedByCudaHost;
	bool transposed;
	bool labelsInFloat;
	bool invertLabels;

	int load_libsvm_data_dense(FILE * &fid, SVM_DATA_TYPE data_type, svm_memory_dataformat req_data_format);

public:
	SvmData();
	virtual ~SvmData();
	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type, struct svm_memory_dataformat req_data_format);
	virtual int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) = 0; //requested data format settings need to be overloaded, than Load() is called 
	int Delete();
	unsigned int GetNumClasses() {return numClasses;}
	unsigned int GetNumVects() {return numVects;}
    unsigned int GetNumVectsAligned() {return numVects_aligned;}
	unsigned int GetDimVects() {return dimVects;}
    unsigned int GetDimVectsAligned() {return dimVects_aligned;}
	SVM_DATA_TYPE GetDataType() {return type;}
	float * GetDataRawPointer() {return data_raw;}
	float GetValue(unsigned int iVect, unsigned int iDim) {return transposed? data_raw[iDim * numVects_aligned + iVect] : data_raw[iVect * dimVects_aligned + iDim];}
	int * GetVectorLabelsPointer() {return vector_labels;}
	int * GetClassLabelsPointer() {return class_labels;}
	bool GetLabelsInFloat() {return labelsInFloat;}

	friend class SvmModel;
};

class SvmModel {
private:
protected:
	float *alphas;
	SvmData *data; //pointer to exiting external SvmData object - it is not own memory
	struct svm_params * params;
	bool allocatedByCudaHost;

	int StoreModel_LIBSVM_TXT(char *model_file_name);
	int StoreModelGeneric(char *model_file_name, SVM_MODEL_FILE_TYPE type);
	int CalculateSupperVectorCounts();

public:
	SvmModel();
	virtual ~SvmModel();
	int Delete();
	virtual int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo) = 0;
	virtual int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) = 0;
};

class Utils {
public:
	static int StoreResults(char *filename, int *results, unsigned int numResults);
};

#endif //_SVM_TEMPLATE
