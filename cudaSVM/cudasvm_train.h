#ifndef _CUDASVM_TRAIN_H_
#define _CUDASVM_TRAIN_H_

// main function providing the traning
int perform_training(float * X, int * labels, double *alphas, int N_, int D, double C,
				  double eps, float coef0, float degree, float gamma, double * rho, int kernel_type);

#ifdef __cplusplus
extern "C" {
#endif

struct cusvm_kernel {
	int svm_type;
	int kernel_type;
	int degree;	    /* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */
	double rho;     /* Threshold. */

	//unsigned int pcv;		/* parallelly computed values */
};

struct cusvm_model {
	/* Host content. */
	unsigned int m_nof_vectors;
	unsigned int m_vector_len;
	unsigned int pitch;         /* Pitch of the _dvectors array. */
	
	cusvm_kernel kernel_param;  /* Kernel parameters. */
	
	//Not needed ATM -- currently we are only interested in two class SVC
	//size_t *m_class_off;
	//int m_nof_classes;

	float *vector; //Vectors on host -- CPU/RAM
	float *alpha; //Alpha coefficients for individual SVs

	//GPU content
	float *_dvectors; /* SVs on device -- GPU*/
	float *_dalphas;  /* alphas on device */

	//float *data_dense;

	//void *_device_ctx; // device context, not used ATM
};

struct cusvm_prob {
	/* Host data. */
	unsigned int nof_vectors; /* Number of test vectors stored in device's memory. */
	unsigned int width;       /* Width of test vectors. */
	size_t pitch;			  /* Pitch of cuda array. */
	int * labels;             /* Correct decisions given with test vectors. */
	float * data;			  /* Vectors. */
	double * result;		  /* Results. */
	float * b;				  /* The partial results. */

	/* Device data. */
	float * _dlabels;          /* Correct decisions given with test vectors on the device. */
	float * _ddata;            /* Vectors on the device. */
	float * _dresult;          /* Temporary array of particular results at device. */
	float * _db;			   /* The partial results used in calculation of threshold rho and stopping criteria. */
};

void cusvm_destroy_model(cusvm_model * &cumodel);
void cusvm_destroy_data(cusvm_prob * &problem);

void malloc_host(void** ptr, size_t size);
void free_host(void* vector);

void cusvm_create_data(cusvm_prob *prob, float *test);
void cusvm_create_model(cusvm_model * model, float *alphas, float *vectors);

void transfer_to_gpu(void ** device_ptr, void * host_ptr, size_t size);
void transfer_to_gpu_2d(void ** device_ptr, void * host_ptr, size_t * pitch,
						size_t width, size_t height);

int cudasvm_train(struct cusvm_prob * prob, struct svm_params * params);

#ifdef __cplusplus
}
#endif

#endif /* _CUDASVM_TRAIN_H_ */
