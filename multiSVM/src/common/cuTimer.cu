#include <stdio.h>
int timerCycles = 0;
float timerTime = 0;

#ifdef _WIN32

#include <windows.h>

static LARGE_INTEGER t;
static float         f;
static int           freq_init = 0;

void cuResetTimer(void) {
  if (!freq_init) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    f = (float) freq.QuadPart;
    freq_init = 1;
  }
  QueryPerformanceCounter(&t);
}

float cuGetTimer(void) {
  LARGE_INTEGER s;
  float d;
  QueryPerformanceCounter(&s);

  d = ((float)(s.QuadPart - t.QuadPart)) / f;

  return (d*1000.0f);
}

#else

#include <sys/time.h>

static struct timeval t;

/**
 * Resets timer
 */
void cuResetTimer() {
  gettimeofday(&t, NULL);
}


/**
 * Gets time since reset
 */
float cuGetTimer() { // result in miliSec
  static struct timeval s;
  gettimeofday(&s, NULL);

  return (s.tv_sec - t.tv_sec) * 1000.0f + (s.tv_usec - t.tv_usec) / 1000.0f;
}

#endif

