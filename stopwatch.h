/*
*      StopWatch -- a portable stop watches for measuring time intervals (real-time, i.e. not process or thread -consumed time)
*
* Copyright (c) Jan Trmal (jtrmal@kky.zcu.cz)
* All rights reserved.
*
*/

/* $Id: htk_fast.cpp 348 2008-10-29 14:32:42Z jtrmal $ */
/* $HeadURL: svn+ssh://jtrmal@147.228.47.73/people/jtrmal/projects/ymann/branches/mymANN/stopwatch.h $ */


#ifndef __STOP_WATCH_H_INCLUDED__
#define __STOP_WATCH_H_INCLUDED__


#include <inttypes.h>

#ifdef _MSC_VER 
#include <Windows.h>
class WindowsStopWatch;

typedef WindowsStopWatch StopWatch;

class WindowsStopWatch {
	unsigned nofCycles;
	uint64_t timeAccum;
	
	LARGE_INTEGER curStart;
	LARGE_INTEGER curStop;

	static inline uint64_t clockResulution() {
			LARGE_INTEGER foo;
			QueryPerformanceFrequency(&foo);
			return foo.QuadPart;
	};
	
	static inline uint64_t diffFileTime(LARGE_INTEGER &a, LARGE_INTEGER &b) 
	{
		return b.QuadPart - a.QuadPart;
	};

public:
	WindowsStopWatch() : nofCycles(0), timeAccum(0UL) {};
	
	inline void start() 
	{ 
		QueryPerformanceCounter(&curStart);
	};

	inline void stop() 
	{ 
		QueryPerformanceCounter(&curStop);
		timeAccum += diffFileTime(curStart, curStop);
		nofCycles +=1;
	};

	inline void stopstart() 
	{ 
		QueryPerformanceCounter(&curStop);
		timeAccum += diffFileTime(curStart, curStop);
		nofCycles +=1;
		curStart = curStop;
	};

	inline void reset() 
	{
		nofCycles = 0;
		timeAccum = 0;
	};

	inline const float getTime() const
	{
		return (float) timeAccum / (float)clockResulution();
	}
    
    inline const uint64_t getTicks() const
    {
        return (uint64_t)timeAccum;
    }

	inline const float avgTime() const
	{
		return getTime() / (float)nofCycles;
	}

	inline float getCurrentTime()
	{
    QueryPerformanceCounter(&curStop);
		return (float)((double)diffFileTime(curStart, curStop) / clockResulution());
	}
};
#else
#include <sys/time.h>
class PosixStopWatch;

typedef PosixStopWatch StopWatch;


class PosixStopWatch {
	unsigned nofCycles;
	uint64_t timeAccum;
	
	timeval curStop;
	timeval curStart;

	static inline uint64_t clockResulution() {
		static const unsigned usec_per_sec = 1000000;
		return usec_per_sec;
	};

	static inline void getClocks(timeval &time) {
		gettimeofday(&time, NULL);
	};
	
	static inline uint64_t timeDiff(timeval start, timeval stop) {
		uint64_t ticks1 = start.tv_sec * clockResulution() + start.tv_usec;
		uint64_t ticks2 = stop.tv_sec * clockResulution() + stop.tv_usec;
		
		return ticks2-ticks1;
	};

public:
	PosixStopWatch() : nofCycles(0), timeAccum(0UL) {};
	
	inline void start() 
	{ 
		getClocks(curStart);
	};

	inline void stop() 
	{ 
		getClocks(curStop);

		timeAccum += timeDiff(curStart, curStop);
		nofCycles +=1;
	};

	inline void reset() 
	{
		nofCycles = 0;
		timeAccum = 0;
	};

	inline const float getTime() const
	{	
		return (float) timeAccum / (float)clockResulution();
	}
    
	inline const uint64_t getTicks() const
	{
        	return (uint64_t)timeAccum;
	}

	inline const float avgTime() const
	{
		return getTime() / (float)nofCycles;
	}

	inline float getCurrentTime()
	{
		getClocks(curStop);
		return (float)((double)timeDiff(curStart, curStop) / clockResulution());
	}
};

#endif
#endif
