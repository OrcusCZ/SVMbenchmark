#include "Controller.h"
#include <math.h>
#include <stdio.h>

Controller::Controller(float initialGap, SelectionHeuristic currentMethodIn, int samplingIntervalIn, int problemSize) {
  progress.push_back(initialGap);
  currentMethod = currentMethodIn;
  if (currentMethod == ADAPTIVE) {
    adaptive = true;
    currentMethod = SECONDORDER;
  } else {
    adaptive = false;
  }
  samplingInterval = samplingIntervalIn;
  inspectionPeriod = problemSize/(10*samplingInterval);

  
  timeSinceInspection = inspectionPeriod - 2;
  beginningOfEpoch = 0;
  rates.push_back(0);
  rates.push_back(0);
  currentInspectionPhase = 0;
  printf("Controller: currentMethod: %i (%s), inspectionPeriod: %i\n", currentMethod, adaptive?"dynamic":"static", inspectionPeriod);
}

void Controller::addIteration(float gap) {
  progress.push_back(gap);
  method.push_back(currentMethod);
  
}

float Controller::findRate(struct time_struct* start, struct time_struct* finish, int beginning, int end) {
  //printf("findRate: (%i -> %i) = ", beginning, end);
  float time = ((float)(finish->tv_sec - start->tv_sec))*1000000 + ((float)(finish->tv_usec - start->tv_usec));
  int length = end - beginning;
  int filterLength = length/2;
  float phase1Gap = filter(beginning, beginning + filterLength);
  float phase2Gap = filter(beginning + filterLength, end);
  float percentageChange = (phase2Gap - phase1Gap)/phase1Gap;
  float percentRate = percentageChange / time;
  //printf("%f\n", percentRate);
  return percentRate;
}

SelectionHeuristic Controller::getMethod() {
  if (!adaptive) {
    if (currentMethod == RANDOM) {
      if ((rand() & 0x1) > 0) {
        return SECONDORDER;
      } else {
        return FIRSTORDER;
      }
    }
    return currentMethod;
  }

  
  if (timeSinceInspection >= inspectionPeriod) {
    int currentIteration = progress.size();
    get_timer_time(&start);
    currentInspectionPhase = 1;
    timeSinceInspection = 0;
    beginningOfEpoch = currentIteration;
  } else if (currentInspectionPhase == 1) {
    int currentIteration = progress.size();

    middleOfEpoch = currentIteration;
    get_timer_time(&mid);
    rates[currentMethod] = findRate(&start, &mid, beginningOfEpoch, middleOfEpoch);
    currentInspectionPhase++;

    if (currentMethod == FIRSTORDER) {
      currentMethod = SECONDORDER;
    } else {
      currentMethod = FIRSTORDER;
    }
    
  } else if (currentInspectionPhase == 2) {
    int currentIteration = progress.size();
        
    get_timer_time(&finish);
    rates[currentMethod] = findRate(&mid, &finish, middleOfEpoch, currentIteration);
    timeSinceInspection = 0;
    currentInspectionPhase = 0;
    
    if (fabs(rates[1]) > fabs(rates[0])) {
      currentMethod = SECONDORDER;
    } else {
      currentMethod = FIRSTORDER;
    }
    //printf("Rate 0: %f, Rate 1: %f, choose method: %i\n", rates[0], rates[1], currentMethod);
  } else {
    timeSinceInspection++;
  }
  return currentMethod;
}

float Controller::filter(int begin, int end) {
  float accumulant = 0;
  for (int i = begin; i < end; i++) {
    accumulant += progress[i];
  }
  accumulant = accumulant / ((float)(end - begin));
  return accumulant;
}

void Controller::print() {
  FILE* outputFilePointer = fopen("gap.dat", "w");
  if (outputFilePointer == NULL) {
    printf("Can't write %s\n", "gap.dat");
    exit(1);
  }
  for(vector<float>::iterator i = progress.begin(); i != progress.end(); i++) {
    fprintf(outputFilePointer, "%f ", *i);
  }
  fprintf(outputFilePointer, "\n");
  fclose(outputFilePointer);

  outputFilePointer = fopen("method.dat", "w");
  if (outputFilePointer == NULL) {
    printf("Can't write %s\n", "method.dat");
    exit(1);
  }
  for(vector<int>::iterator i = method.begin(); i != method.end(); i++) {
    fprintf(outputFilePointer, "%d ", *i);
  }
  fprintf(outputFilePointer, "\n");
  fclose(outputFilePointer);
}

LARGE_INTEGER f;
LARGE_INTEGER ct;
int time_set = 0;

void get_timer_time(struct time_struct * time) {
	double current_time;

	if (time_set == 0) {
		time_set = 1;
		QueryPerformanceFrequency(&f);
		QueryPerformanceCounter(&ct);
	}

	current_time = (double) ct.QuadPart / (double) f.QuadPart;

	time->tv_sec = (long) current_time;
	time->tv_usec = (long) ((current_time - ((double) time->tv_sec)) * 1000000);
}
