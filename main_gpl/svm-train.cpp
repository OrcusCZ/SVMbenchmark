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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef COMPILE_WITH_GTSVM
#include "gtSVM_wrapper.h"
#endif

#include "utils.h"
#include "svm.h"
#include "my_stopwatch.h"

using namespace libsvm;

int g_cache_size = 0;
bool g_step_on_cpu = false;
int g_ws_size = 0;

int help(int argc, char **argv, SvmData * &data, SvmModel * &model, struct svm_params * params, SVM_FILE_TYPE *file_type, SVM_DATA_TYPE *data_type, SVM_MODEL_FILE_TYPE *model_file_type);

int main(int argc, char **argv) {
    struct svm_params params;
    struct svm_trainingInfo trainingInfo;
    SVM_FILE_TYPE file_type = LIBSVM_TXT;
    SVM_DATA_TYPE data_type = UNKNOWN;
    SVM_MODEL_FILE_TYPE model_file_type = M_LIBSVM_TXT;
    MyStopWatch clAll, clLoad, clProc, clStore;
    SvmData *data;
    SvmModel *model;

    /* Check input arguments. */
    if(help(argc, argv, data, model, &params, &file_type, &data_type, &model_file_type) != SUCCESS) {
        return EXIT_SUCCESS;
    }

    clAll.start();

    /* Load data. */ 
    clLoad.start();
    if(data->Load(argv[1], file_type, data_type) != SUCCESS) {
        return EXIT_FAILURE;
    }
    clLoad.stop();

    clProc.start();
    /* Train model. */
    if(model->Train(data, &params, &trainingInfo) != SUCCESS) {
        return EXIT_FAILURE;
    }
    clProc.stop();

    clStore.start();
    /* Predict values. */
    if(model->StoreModel(argv[2], model_file_type) != SUCCESS) {
        return EXIT_FAILURE;
    }

    /* Clean memory. */
    delete model;
    delete data;

    clStore.stop();

    clAll.stop();

    /* Print results. */
    printf("\nLoading    elapsed time : %0.4f s\n", clLoad.getTime());
    printf("Processing elapsed time : %0.4f s\n", clProc.getTime());
    printf("Storing    elapsed time : %0.4f s\n", clStore.getTime());
    printf("Total      elapsed time : %0.4f s\n", clAll.getTime());

    return EXIT_SUCCESS;
}

int help(int argc, char **argv, SvmData * &data, SvmModel * &model, struct svm_params *params, SVM_FILE_TYPE *file_type, SVM_DATA_TYPE *data_type, SVM_MODEL_FILE_TYPE *model_file_type) {
    int i, imp;

    params->kernel_type = RBF;/*k*/
    params->C = 1;                /*c, cost value*/
    params->eps = 1e-3;            /*e, stopping criteria */
    params->degree = 3;            /*d*/
    params->gamma = 0;            /*g*/
    params->coef0 = 0;            /*f*/
    params->nu = 0.5;            /*n*/
    params->p = 0.1;            /*p, regression parameter epsilon*/

    imp = 6;                    /*i*/

    if (argc >= INPUT_ARGS_MIN) {
        for (i = INPUT_ARGS_MIN; i < argc; i++) {
            if (argv[i][0] != '-' || (argv[i][1] != 'b' && (i + 1) == argc)) {
                print_help();
                return FAILURE;
            }
            
            switch (argv[i++][1]) {
            case 'k':
                switch (argv[i][0]) {
                case 'l':
                    params->kernel_type = LINEAR;
                    break;
                case 'p':
                    params->kernel_type = POLY;
                    break;
                case 'r':
                    params->kernel_type = RBF;
                    break;
                case 's':
                    params->kernel_type = SIGMOID;
                    break;
                default:
                    printf("Error: Invalid kernel selection \"%c\"!\n\n", argv[i][0]);
                    print_help();
                    return FAILURE;
                }
                break;
            case 'c':
                params->C = atof(argv[i]);
                break;
            case 'e':
                params->eps = atof(argv[i]);
                break;
            case 'd':
                params->degree = atoi(argv[i]);
                break;
            case 'g':
                params->gamma = atof(argv[i]);
                break;
            case 'f':
                params->coef0 = atof(argv[i]);
                break;
            case 'n':
                params->nu = atof(argv[i]);
                break;
            case 'p':
                params->p = atof(argv[i]);
                break;
            case 'i':
                imp = atoi(argv[i]);
                break;
            case 'r':
                g_cache_size = atoi(argv[i]);
                break;
            case 'w':
                g_ws_size = atoi(argv[i]);
                break;
            case 'b':
                *file_type = LASVM_BINARY;
                i--;
                break;
            default:
                printf("Error: Invalid attribute \"%c\"!\n\n", argv[i - 1][1]);
                print_help();
                return FAILURE;
            }
        }
    } else {
        printf("Error: Not enough arguments!\n\n");
        print_help();
        return FAILURE;
    }

    switch (imp) {
#ifdef COMPILE_WITH_GTSVM
    case 6:
        printf("Using gtSVM(large clusters) (Andrew Cotter)...\n\n");
        data = new GtSvmData;
        model = new GtSvmModel(false);
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_GTSVM
	case 7:
		printf("Using gtSVM(small clusters) (Andrew Cotter)...\n\n");
		data = new GtSvmData;
		model = new GtSvmModel(true);
		return SUCCESS;
#endif
	default:
        printf("Error: Invalid implementation \"%d\"!\n\n", imp);
        print_help();
        return FAILURE;
    }

    return SUCCESS;
}

/* Print help information. */
void print_help() {
    printf("SVM-train benchmark\n"
        "Use: SVM-train.exe <data> <model> [-<attr1> <value1> ...]\n\n"
        " data   File containing data to be used for training.\n"
        " model  Where to store model in LibSVM format.\n"
        " attrx  Attribute to set.\n"
        " valuex Value of attribute x.\n\n"
        " Attributes:\n"
        "  k  SVM kernel. Corresponding values:\n"
        "         l   Linear\n"
        "         p   Polynomial\n"
        "         r   RBF\n"
        "         s   Sigmoid\n"
        "  c  Training parameter C.\n"
        "  e  Stopping criteria.\n"
        "  n  Training parameter nu.\n"
        "  p  Training parameter p.\n"
        "  d  Kernel parameter degree.\n"
        "  g  Kernel parameter gamma.\n"
        "  f  Kernel parameter coef0.\n"
        "  i  Select implementation to use. Corresponding values:\n"
#ifdef COMPILE_WITH_GTSVM
        "         6   GTSVM - large clusters (Andrew Cotter)\n"
        "         7   GTSVM - small clusters (Andrew Cotter)\n"
#endif
        "  b  Read input data in binary format (lasvm dense or sparse format)\n"
        "  w  Working set size (currently only for implementation 12)\n"
        "  r  Cache size in MB\n"
        );
}
