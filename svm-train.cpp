/*
MIT License

Copyright (c) 2015 University of West Bohemia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef COMPILE_WITH_LIBSVM
#include "libSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_GPULIBSVM
#include "GPUlibSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_CUSVM
#include "cuSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_GPUSVM
#include "gpuSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_MULTISVM
#include "multiSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_GTSVM
#ifndef SEPARATE_GPL_BIN
#include "gtSVM_wrapper.h"
#endif
#endif
#ifdef COMPILE_WITH_WUSVM
#include "wuSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_OHDSVM
#include "ohdSVM_wrapper.h"
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

    imp = 1;                    /*i*/

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
			case 't':
				switch (argv[i][0])
				{
				case 'd': *data_type = DENSE; break;
				case 's': *data_type = SPARSE; break;
				default:
					printf("Error: Invalid data type \"%c\"!\n\n", argv[i][0]);
					print_help();
					return FAILURE;
				}
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
#ifdef COMPILE_WITH_LIBSVM
    case 1:
        data = new LibSvmData;
        model = new LibSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_GPULIBSVM
    case 2:
        printf("Using GPU-LibSVM (Athanasopoulos)...\n\n");
        data = new GPULibSvmData;
        model = new GPULibSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_CUSVM
    case 3:
        printf("Using cuSVM (Carpenter)...\n\n");
        data = new CuSvmData;
        model = new CuSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_GPUSVM
    case 4:
        printf("Using gpuSVM (Catanzaro)...\n\n");
        data = new GpuSvmData;
        model = new GpuSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_MULTISVM
    case 5:
        printf("Using multiSVM (Herrero-Lopez)...\n\n");
        data = new MultiSvmData;
        model = new MultiSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_GTSVM
#ifndef SEPARATE_GPL_BIN
   case 6:
       printf("Using gtSVM(large clusters) (Andrew Cotter)...\n\n");
       data = new GtSvmData;
       model = new GtSvmModel(false);
       return SUCCESS;
	case 7:
		printf("Using gtSVM(small clusters) (Andrew Cotter)...\n\n");
		data = new GtSvmData;
		model = new GtSvmModel(true);
		return SUCCESS;
#endif
#endif
#ifdef COMPILE_WITH_WUSVM
	case 8:
		printf("Using WuLibSVM<double, lasp, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, false, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 9:
		printf("Using WuLibSVM<double, lasp, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, false, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 10:
		printf("Using WuLibSVM<double, pegasos, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, true, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 11:
		printf("Using WuLibSVM<double, pegasos, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, true, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 12:
		printf("Using WuLibSVM<float, lasp, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, false, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 13:
		printf("Using WuLibSVM<float, lasp, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, false, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 14:
		printf("Using WuLibSVM<float, pegasos, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, true, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 15:
		printf("Using WuLibSVM<float, pegasos, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, true, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
#endif
#ifdef COMPILE_WITH_OHDSVM
	case 16:
		printf("Using OHD-SVM...\n\n");
		data = new ohdSVMData;
		model = new ohdSVMModel;
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
        "Use: SVMbenchmark.exe <data> <model> [-<attr1> <value1> ...]\n\n"
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
		"  t  Force data type. Values are:\n"
		"         d   Dense data\n"
		"         s   Sparse data\n"
        "  i  Select implementation to use. Corresponding values:\n"
#ifdef COMPILE_WITH_LIBSVM
        "         1   LibSVM (default)\n"
#endif
#ifdef COMPILE_WITH_GPULIBSVM
        "         2   GPU-LibSVM (Athanasopoulos)\n"
#endif
#ifdef COMPILE_WITH_CUSVM
        "         3   CuSVM (Carpenter)\n"
#endif
#ifdef COMPILE_WITH_GPUSVM
        "         4   GpuSVM (Catanzaro)\n"
#endif
#ifdef COMPILE_WITH_MULTISVM
        "         5   MultiSVM (Herrero-Lopez)\n"
#endif
#ifdef COMPILE_WITH_GTSVM
        "         6   GTSVM - large clusters (Andrew Cotter)\n"
        "         7   GTSVM - small clusters (Andrew Cotter)\n"
#ifdef SEPARATE_GPL_BIN
		"             Disabled due to license issues, use GPL build\n"
#endif
#endif
#ifdef COMPILE_WITH_WUSVM
		"         8   WuSVM<double, lasp, openMP> (Tyree et al.)\n"
		"         9   WuSVM<double, lasp, GPU> (Tyree et al.)\n"
		"        10   WuSVM<double, pegasos, openMP> (Tyree et al.)\n"
		"        11   WuSVM<double, pegasos, GPU> (Tyree et al.)\n"
		"        12   WuSVM<float, lasp, openMP> (Tyree et al.)\n"
		"        13   WuSVM<float, lasp, GPU> (Tyree et al.)\n"
		"        14   WuSVM<float, pegasos, openMP> (Tyree et al.)\n"
		"        15   WuSVM<float, pegasos, GPU> (Tyree et al.)\n"
#endif
#ifdef COMPILE_WITH_OHDSVM
		"        16   OHD-SVM (Michalek,Vanek)\n"
#endif
        "  b  Read input data in binary format (lasvm dense or sparse format)\n"
        "  w  Working set size (currently only for implementation 16)\n"
        "  r  Cache size in MB\n"
        );
}
