#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libSVM_wrapper.h"
#ifndef __NO_CUDA
#include "GPULibSVM_wrapper.h"
#include "cuSVM_wrapper.h"
#include "gpuSVM_wrapper.h"
#include "multiSVM_wrapper.h"
#include "gtSVM_wrapper.h"
#include "orcusSVM/orcusSVM_wrapper.h"
#endif
#include "openclSVM/openclSVM_wrapper.h"

#include "utils.h"
#include "stopwatch.h"

int g_cache_size = 0;

int help(int argc, char **argv, SvmData * &data, SvmModel * &model, struct svm_params * params, SVM_FILE_TYPE *file_type, SVM_DATA_TYPE *data_type, SVM_MODEL_FILE_TYPE *model_file_type);

int main(int argc, char **argv) {
    struct svm_params params;
    struct svm_trainingInfo trainingInfo;
    SVM_FILE_TYPE file_type = LIBSVM_TXT;
    SVM_DATA_TYPE data_type = UNKNOWN;
    SVM_MODEL_FILE_TYPE model_file_type = M_LIBSVM_TXT;
    StopWatch clAll, clLoad, clProc, clStore;
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

    imp = 1;                    /*i*/

    if (argc >= INPUT_ARGS_MIN) {
        for (i = INPUT_ARGS_MIN; i < argc; i++) {
            if (argv[i][0] != '-' || (i + 1) == argc) {
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
    case 1:
        data = new LibSvmData;
        model = new LibSvmModel;
        return SUCCESS;
#ifndef __NO_CUDA
    case 2:
        printf("Using GPU-LibSVM (Athanasopoulos)...\n\n");
        data = new GPULibSvmData;
        model = new GPULibSvmModel;
        return SUCCESS;
    case 3:
        printf("Using cuSVM (Carpenter)...\n\n");
        data = new CuSvmData;
        model = new CuSvmModel;
        return SUCCESS;
    case 4:
        printf("Using gpuSVM (Catanzaro)...\n\n");
        data = new GpuSvmData;
        model = new GpuSvmModel;
        return SUCCESS;
    case 5:
        printf("Using multiSVM (Herrero-Lopez)...\n\n");
        data = new MultiSvmData;
        model = new MultiSvmModel;
        return SUCCESS;
    case 6:
        printf("Using gtSVM (Andrew Cotter)...\n\n");
        data = new GtSvmData;
        model = new GtSvmModel;
        return SUCCESS;
    case 7:
        printf("Using OrcusSVM...\n\n");
        data = new OrcusSvmData;
        model = new OrcusSvmModel;
        return SUCCESS;
#endif //NO_CUDA
    case 8:
        printf("Using OpenCLSVM...\n\n");
        data = new OpenCLSvmData;
        model = new OpenCLSvmModel;
        return SUCCESS;
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
        "         1   LibSVM (default)\n"
        "         2   GPU-LibSVM (Athanasopoulos)\n"
        "         3   CuSVM (Carpenter)\n"
        "         4   GpuSVM (Catanzaro)\n"
        "         5   MultiSVM (Herrero-Lopez)\n"
        "         6   GTSVM (Andrew Cotter)\n"
        "         7   OrcusSVM\n"
        "  b  Read input data in binary format (lasvm dense or sparse format)\n"
        );
}
