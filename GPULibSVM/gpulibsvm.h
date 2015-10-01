//GPU LibSVM modeifications

#ifndef _GPULIBSVM_H
#define _GPULIBSVM_H
#define _DENSE_REP
//#define LIBSVM_VERSION 317

#ifdef __cplusplus
extern "C" {
#endif

//extern int libsvm_version;

#ifdef _DENSE_REP
struct gpulibsvm_node
{
	int dim;
	double *values;
};

struct gpulibsvm_problem
{
	int l;
	double *y;
	struct gpulibsvm_node *x;
};

#else
struct gpulibsvm_node
{
	int index;
	double value;
};

struct gpulibsvm_problem
{
	int l;
	double *y;
	struct gpulibsvm_node **x;
};
#endif

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
// gpulibsvm_model
// 
struct gpulibsvm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
#ifdef _DENSE_REP
	struct gpulibsvm_node *SV;		/* SVs (SV[l]) */
#else
	struct gpulibsvm_node **SV;		/* SVs (SV[l]) */
#endif
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

struct gpulibsvm_model *gpulibsvm_train(const struct gpulibsvm_problem *prob, const struct svm_parameter *param);
void gpulibsvm_cross_validation(const struct gpulibsvm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int gpulibsvm_save_model(const char *model_file_name, const struct gpulibsvm_model *model);
//struct svm_model *svm_load_model(const char *model_file_name);

int gpulibsvm_get_svm_type(const struct gpulibsvm_model *model);
int gpulibsvm_get_nr_class(const struct gpulibsvm_model *model);
void gpulibsvm_get_labels(const struct gpulibsvm_model *model, int *label);
void gpulibsvm_get_sv_indices(const struct gpulibsvm_model *model, int *sv_indices);
int gpulibsvm_get_nr_sv(const struct gpulibsvm_model *model);
double gpulibsvm_get_svr_probability(const struct gpulibsvm_model *model);

double gpulibsvm_predict_values(const struct gpulibsvm_model *model, const struct gpulibsvm_node *x, double* dec_values);
double gpulibsvm_predict(const struct gpulibsvm_model *model, const struct gpulibsvm_node *x);
double gpulibsvm_predict_probability(const struct gpulibsvm_model *model, const struct gpulibsvm_node *x, double* prob_estimates);
//
void gpulibsvm_free_model_content(struct gpulibsvm_model *model_ptr);
void gpulibsvm_free_and_destroy_model(struct gpulibsvm_model **model_ptr_ptr);
//void svm_destroy_param(struct svm_parameter *param);
//
//const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
//int svm_check_probability_model(const struct svm_model *model);
//
//void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _GPULIBSVM_H */
