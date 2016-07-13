//GPU LibSVM modeifications

#ifndef _ACCLIBSVM_H
#define _ACCLIBSVM_H
#define _DENSE_REP
//#define LIBSVM_VERSION 317

#ifdef __cplusplus
extern "C" {
#endif

struct acclibsvm_node
{
	int num;
	float *values;
	unsigned int *ind; //for sparse data
};

struct acclibsvm_problem
{
	int l;
	double *y;
	struct acclibsvm_node *x;
};

//enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
//enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

//struct svm_parameter
//{
//	int svm_type;
//	int kernel_type;
//	int degree;	/* for poly */
//	double gamma;	/* for poly/rbf/sigmoid */
//	double coef0;	/* for poly/sigmoid */
//
//	/* these are for training only */
//	double cache_size; /* in MB */
//	double eps;	/* stopping criteria */
//	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
//	int nr_weight;		/* for C_SVC */
//	int *weight_label;	/* for C_SVC */
//	double* weight;		/* for C_SVC */
//	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
//	double p;	/* for EPSILON_SVR */
//	int shrinking;	/* use the shrinking heuristics */
//	int probability; /* do probability estimates */
//};

//
// acclibsvm_model
// 
struct acclibsvm_model
{
	struct libsvm::svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct acclibsvm_node *SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */
	
	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};

struct acclibsvm_model *acclibsvm_train(const struct acclibsvm_problem *prob, const struct libsvm::svm_parameter *param);
void acclibsvm_cross_validation(const struct acclibsvm_problem *prob, const struct libsvm::svm_parameter *param, int nr_fold, double *target);

int acclibsvm_save_model(const char *model_file_name, const struct acclibsvm_model *model, int dimOffset=0);
//struct svm_model *svm_load_model(const char *model_file_name);

int acclibsvm_get_svm_type(const struct acclibsvm_model *model);
int acclibsvm_get_nr_class(const struct acclibsvm_model *model);
void acclibsvm_get_labels(const struct acclibsvm_model *model, int *label);
void acclibsvm_get_sv_indices(const struct acclibsvm_model *model, int *sv_indices);
int acclibsvm_get_nr_sv(const struct acclibsvm_model *model);
double acclibsvm_get_svr_probability(const struct acclibsvm_model *model);

double acclibsvm_predict_values(const struct acclibsvm_model *model, const struct acclibsvm_node *x, double* dec_values);
double acclibsvm_predict(const struct acclibsvm_model *model, const struct acclibsvm_node *x);
double acclibsvm_predict_probability(const struct acclibsvm_model *model, const struct acclibsvm_node *x, double* prob_estimates);
//
void acclibsvm_free_model_content(struct acclibsvm_model *model_ptr);
void acclibsvm_free_and_destroy_model(struct acclibsvm_model **model_ptr_ptr);
//void svm_destroy_param(struct svm_parameter *param);
//
//const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
//int svm_check_probability_model(const struct svm_model *model);
//
//void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _ACCLIBSVM_H */
