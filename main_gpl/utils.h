/*
Copyright (C) 2015  University of West Bohemia

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>

#define SUCCESS 0
#define FAILURE 1
#define INPUT_ARGS_MIN 3

#define TRUE 1
#define FALSE 0

#define LINEAR_STRING "linear"
#define POLY_STRING "polynomial"
#define RBF_STRING "rbf"
#define SIGMOID_STRING "sigmoid"

#define LOAD_FLAG_ALL_RAM 0
#define LOAD_FLAG_ALPHAS_WC 1
#define LOAD_FLAG_VECTORS_WC 2 
#define LOAD_FLAG_ALL_WC 3
#define LOAD_FLAG_TRANSPOSE 4
#define LOAD_FLAG_FILE_BUFFER 8

#define ALLOC_ALPHAS_WC(X) (X & 1)
#define ALLOC_VECTORS_WC(X) (X & 2)
#define DO_TRANSPOSE_MATRIX(X) (X & 4)
#define USE_BUFFER(X) (X & 8)

#define ALIGN_UP(x, align) ((align) * (((x) + align - 1) / (align)))

#define SAFE_CALL(call) {int err = call; if (SUCCESS != err) { fprintf (stderr, "Error %i in file %s at line %i.", err, __FILE__, __LINE__ ); exit(EXIT_FAILURE);}}

#define REPORT_ERROR(error_string) { fprintf (stderr, "Error ""%s"" thrown in file %s at line %i.", error_string, __FILE__, __LINE__ ); exit(EXIT_FAILURE);}
#define REPORT_WARNING(error_string) { fprintf (stderr, "Warning ""%s"" raised in file %s at line %i.", error_string, __FILE__, __LINE__ );}

#define FILE_SAFE_OPEN(FID, FNAME, MODE) if ((FID = fopen(FNAME, MODE)) == 0) { \
	std::cerr << "Error: Can't open file \"" << FNAME << "\"!\n"; \
	exit(EXIT_FAILURE); \
}

#define MEM_SAFE_ALLOC(MEM_PTR, TYPE, LEN) if (MEM_PTR != NULL) { \
	free(MEM_PTR); \
} \
if ((MEM_PTR = (TYPE *) malloc((LEN) * sizeof(TYPE))) == NULL) { \
	std::cerr << "Error: Unable to allocate " << (LEN) * sizeof(TYPE) << " B of memory!\n"; \
	exit(EXIT_FAILURE); \
}

#define MEM_SAFE_FREE(MEM_PTR) if (MEM_PTR != NULL) { \
	free(MEM_PTR); \
	MEM_PTR = NULL; \
}

#define BUFFER_LENGTH 16
#define LINE_LENGTH 65535
#define EOL '\n'

/* kernel_type */
#define K_LINEAR 1
#define K_POLYNOMIAL 2
#define K_RBF 3
#define K_SIGMOID 4

#define INIT_DIST 10.0F

#define ABS(X) (((X) < 0) ? -(X) : (X))

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include "svm.h" //some common types used from libSVM
#include "svm_template.h"
#include "cuda_utils.h"

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

void print_help();

void exit_input_error(int line_num);

//void load_data_dense(FILE * &fid, float * &alphas, float * &vectors, unsigned int &height,
//					 unsigned int &width, svm_memory_dataformat format_type);

void cusvm_load_model2(char *filename, struct cuSVM_model * &model);

int get_closest_label(int * labels, int labels_len, float alpha);

void malloc_host_WC(void** ptr, size_t size);
void malloc_host_PINNED(void** ptr, size_t size);

void select_device(int device_id, unsigned int sm_limit, int is_max);

int equals(char *str1, char *str2);
char *next_string(char *str);
char *next_string_spec(char *str);
char *next_string_spec_space(char *str);
char *next_string_spec_space_plus(char *str);
char *next_string_spec_colon(char *str);
char * next_eol(char * str);
char *last_string_spec_colon(char *str);
unsigned int strtoi_reverse(char * str);
float strtof_fast(char * str, char ** end_ptr);

long long parse_line(FILE * & fid, char * & line, char * & line_end, size_t & line_len);

#endif /* _UTILS_H_ */
