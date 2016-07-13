#ifndef PARALEL_THREADS
#define PARALEL_THREADS

#ifdef USE_PTHREADS
#include <pthread.h>

#define OUTTYPE void*
#define INTYPE void*

class ParallelThreads {

public:
	static void run_parallel(OUTTYPE fcn(INTYPE), void *data, size_t dataItemSize, int numThreads) {
	
		pthread_t *o_thread = new pthread_t [numThreads];
		pthread_attr_t attr;
		void *th_return;
	
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		//run all threads
		for(int i=0;i<numThreads;i++) pthread_create(&o_thread[i], &attr, fcn, (void *) (((char*)data)+i*dataItemSize));

		pthread_attr_destroy(&attr);

		for(int i=0;i<numThreads;i++) pthread_join(o_thread[i], &th_return);
		
		delete[] o_thread;
 	
	} //run_parallel

    static void terminate_threads()
    {
    }
};
#else
#define __STDC_LIMIT_MACROS
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>

#define OUTTYPE void*
#define INTYPE void*


class ParallelThreads {
    static std::mutex mutex, mutex_done;
    static std::condition_variable cv, cv_done;
    static std::vector<std::thread> worker_threads;
    static std::queue<INTYPE> queue;
    static int done;
    static bool terminate;
    static void worker_fcn(OUTTYPE fcn(INTYPE))
    {
        while (true)
        {
            INTYPE data_package;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]{ return !queue.empty() || terminate; });
                if (terminate)
                    return;
                data_package = queue.front();
                queue.pop();
            }

            fcn(data_package);

            {
                std::unique_lock<std::mutex> lock(mutex_done);
                done--;
            }
            cv_done.notify_one();
        }
    }
public:
    static void init(OUTTYPE fcn(INTYPE), int numThreads)
    {
        terminate = false;
        for (int i = 0; i < numThreads; i++)
            worker_threads.push_back(std::thread(worker_fcn, fcn));
    }

	static void run_parallel(OUTTYPE fcn(INTYPE), void *data, size_t dataItemSize, int numThreads) {
        if (worker_threads.empty())
            init(fcn, numThreads);

        {
            std::unique_lock<std::mutex> lock(mutex);
            done = numThreads;
            for (int i = 0; i < numThreads; i++)
                queue.push((char *)data + i * dataItemSize);
        }
        cv.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex_done);
            if (done != 0)
                cv_done.wait(lock, [&]{ return done == 0; });
        }
        
	} //run_parallel

    static void terminate_threads()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            terminate = true;
        }
        cv.notify_all();
        for (auto & t : worker_threads)
            t.join();
        worker_threads.clear();
    }
};
#endif

#endif
