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

#ifndef __MY_STOP_WATCH_H_INCLUDED__
#define __MY_STOP_WATCH_H_INCLUDED__


#include <inttypes.h>

//usefull definitions for simple use:
#define MYSTOPWATCH_START MyStopWatch cl; cl.start();
#define MYSTOPWATCH_STOP_AND_PRINT cl.stop(); printf("Elapsed time: %f s\n", cl.getTime());
#define MYSTOPWATCH_STOP cl.stop();
#define MYSTOPWATCH_PRINT printf("Elapsed time: %f s\n", cl.getTime());

#ifdef _MSC_VER 
#include <Windows.h>
class WindowsStopWatch;

typedef WindowsStopWatch MyStopWatch;

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

typedef PosixStopWatch MyStopWatch;


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
