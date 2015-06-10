#ifndef SVMCOMMONH
#define SVMCOMMONH
#include <string>


struct Kernel_params{
	float gamma;
	float coef0;
	int degree;
	float b;
	std::string kernel_type;
};

enum SelectionHeuristic {FIRSTORDER, SECONDORDER, RANDOM, ADAPTIVE};

#endif
