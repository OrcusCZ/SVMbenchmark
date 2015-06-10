#ifndef REDUCTION_OPERATORS_H
#define REDUCTION_OPERATORS_H

__device__ void argMax(int mapAIndex, float mapAValue, int mapBIndex, float mapBValue, int* reduceIndex, float* reduceValue) {
	if (mapBValue > mapAValue) {
		*reduceIndex = mapBIndex;
		*reduceValue = mapBValue;
	} else {
		*reduceIndex = mapAIndex;
		*reduceValue = mapAValue;
	}
}


__device__ void argMin(int mapAIndex, float mapAValue, int mapBIndex, float mapBValue, int* reduceIndex, float* reduceValue) {
	if (mapBValue < mapAValue) {
		*reduceIndex = mapBIndex;
		*reduceValue = mapBValue;
	} else {
		*reduceIndex = mapAIndex;
		*reduceValue = mapAValue;
	}
}

template<typename T>
__device__ void maxOperator(T a, T b, T* result) {
  *result = (a > b) ? a : b;
}

#endif
