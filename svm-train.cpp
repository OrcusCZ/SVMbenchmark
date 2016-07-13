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
#include "gtSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_ORCUSSVM
#include "orcusSVM/orcusSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_ORCUSSVMCPU
#include "orcusSVMCPU/orcusSVMCPU_wrapper.h"
#endif
#ifdef COMPILE_WITH_ORCUSSVM1B
#include "orcusSVM1Block/orcusSVM1Block_wrapper.h"
#endif
#ifdef COMPILE_WITH_ORCUSSVMDP
#include "orcusSVMDP/orcusSVMDP_wrapper.h"
#endif
#ifdef COMPILE_WITH_OPENCLSVM
#include "openclSVM/openclSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_ACCLIBSVM
#include "acclibSVM_wrapper.h"
#endif
#ifdef COMPILE_WITH_WUSVM
#include "wuSVM_wrapper.h"
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
    case 6:
        printf("Using gtSVM(large clusters) (Andrew Cotter)...\n\n");
        data = new GtSvmData;
        model = new GtSvmModel(false);
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_ORCUSSVM
    case 7:
        printf("Using OrcusSVM...\n\n");
        data = new OrcusSvmData;
        model = new OrcusSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_OPENCLSVM
    case 8:
        printf("Using OpenCLSVM...\n\n");
        data = new OpenCLSvmData;
        model = new OpenCLSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_ACCLIBSVM
    case 9:
        printf("Using AccLibSVM...\n\n");
        data = new AccLibSvmData;
        model = new AccLibSvmModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_ORCUSSVMCPU
    case 10:
        printf("Using OrcusSVM CPU cache...\n\n");
        data = new OrcusSvmCPUData;
        model = new OrcusSvmCPUModel;
        return SUCCESS;
    case 11:
        printf("Using OrcusSVM CPU cache and step...\n\n");
        g_step_on_cpu = true;
        data = new OrcusSvmCPUData;
        model = new OrcusSvmCPUModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_ORCUSSVM1B
    case 12:
        printf("Using OrcusSVM1Block...\n\n");
        data = new OrcusSvm1BlockData;
        model = new OrcusSvm1BlockModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_ORCUSSVMDP
    case 13:
        printf("Using OrcusSVMDP...\n\n");
        data = new OrcusSvmDPData;
        model = new OrcusSvmDPModel;
        return SUCCESS;
#endif
#ifdef COMPILE_WITH_GTSVM
	case 16:
		printf("Using gtSVM(small clusters) (Andrew Cotter)...\n\n");
		data = new GtSvmData;
		model = new GtSvmModel(true);
		return SUCCESS;
#endif
#ifdef COMPILE_WITH_WUSVM
	case 21:
		printf("Using WuLibSVM<double, lasp, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, false, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 22:
		printf("Using WuLibSVM<double, lasp, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, false, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 23:
		printf("Using WuLibSVM<double, pegasos, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, true, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 24:
		printf("Using WuLibSVM<double, pegasos, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(false, true, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 25:
		printf("Using WuLibSVM<float, lasp, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, false, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 26:
		printf("Using WuLibSVM<float, lasp, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, false, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 27:
		printf("Using WuLibSVM<float, pegasos, openMP>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, true, false); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
		return SUCCESS;
	case 28:
		printf("Using WuLibSVM<float, pegasos, GPU>...\n\n");
		data = new WuSvmData;
		model = new WuSvmModel(true, true, true); //first bool is: single(true) or double(false), second use_pegasos implementation, third is GPU (true) or CPU openMP (false)
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
#endif
#ifdef COMPILE_WITH_ORCUSSVM
		"         7   OrcusSVM (Michalek,Vanek)\n"
#endif
#ifdef COMPILE_WITH_OPENCLSVM
		"         8   OpenCLSVM (Michalek,Vanek)\n"
#endif
#ifdef COMPILE_WITH_ACCLIBSVM
		"         9   AccLibSVM (Michalek,Vanek)\n"
#endif
#ifdef COMPILE_WITH_ORCUSSVMCPU
        "        10   OrcusSVM CPU cache (Michalek,Vanek)\n"
        "        11   OrcusSVM CPU cache and step (Michalek,Vanek)\n"
#endif
#ifdef COMPILE_WITH_ORCUSSVM1B
        "        12   Chunking OrcusSVM (Michalek,Vanek)\n"
#endif
#ifdef COMPILE_WITH_ORCUSSVMDP
        "        13   OrcusSVM with dynamic parallelism (Michalek,Vanek)\n"
#endif
#ifdef COMPILE_WITH_GTSVM
		"        16   GTSVM - small clusters (Andrew Cotter)\n"
#endif
#ifdef COMPILE_WITH_WUSVM
		"        21   WuSVM<double, lasp, openMP> (Tyree et al.)\n"
		"        22   WuSVM<double, lasp, GPU> (Tyree et al.)\n"
		"        23   WuSVM<double, pegasos, openMP> (Tyree et al.)\n"
		"        24   WuSVM<double, pegasos, GPU> (Tyree et al.)\n"
		"        25   WuSVM<float, lasp, openMP> (Tyree et al.)\n"
		"        26   WuSVM<float, lasp, GPU> (Tyree et al.)\n"
		"        27   WuSVM<float, pegasos, openMP> (Tyree et al.)\n"
		"        28   WuSVM<float, pegasos, GPU> (Tyree et al.)\n"
#endif
        "  b  Read input data in binary format (lasvm dense or sparse format)\n"
        "  w  Working set size (currently only for implementation 12)\n"
        "  r  Cache size in MB\n"
        );
}
