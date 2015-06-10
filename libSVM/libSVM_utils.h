#ifndef _LIBSVM_UTILS_H_
#define _LIBSVM_UTILS_H_

static char* readline(FILE *input);

void read_problem(const char *filename);

void load_problem(struct svm_problem * &problem);

#endif /* _LIBSVM_UTILS_H_ */
