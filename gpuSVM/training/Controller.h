#ifndef _CONTROLLER_H_
#define _CONTROLLER_H_

#include <vector>
#include "svmCommon.h"
#include <cstdlib>

#include <inttypes.h>
#include <windows.h>

struct time_struct {
	long tv_sec;         /* seconds */
	long tv_usec;        /* and microseconds */
};

void get_timer_time(struct time_struct * time);

using std::vector;

class Controller {
 public:
  Controller(float initialGap, SelectionHeuristic currentMethodIn, int samplingIntervalIn, int problemSize);
  void addIteration(float gap);
  void print();
  SelectionHeuristic getMethod();
 private:
  bool adaptive;
  int samplingInterval;
  vector<float> progress;
  vector<int> method;
  SelectionHeuristic currentMethod;
  vector<float> rates;
  int timeSinceInspection;
  int inspectionPeriod;
  int beginningOfEpoch;
  int middleOfEpoch;
  int currentInspectionPhase;
  float filter(int begin, int end);
  float findRate(struct time_struct* start, struct time_struct* finish, int beginning, int end);
  struct time_struct start;
  struct time_struct mid;
  struct time_struct finish;
};

#endif /* _CONTROLLER_H_ */
