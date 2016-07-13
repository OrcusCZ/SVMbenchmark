#include "parallelthreads.h"

#ifdef USE_PTHREADS
#else
std::mutex ParallelThreads::mutex;
std::mutex ParallelThreads::mutex_done;
std::condition_variable ParallelThreads::cv;
std::condition_variable ParallelThreads::cv_done;
std::vector<std::thread> ParallelThreads::worker_threads;
std::queue<INTYPE> ParallelThreads::queue;
int ParallelThreads::done;
bool ParallelThreads::terminate = false;
#endif