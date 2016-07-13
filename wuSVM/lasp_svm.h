/*
Copyright (c) 2014, Washington University in St. Louis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Washington University in St. Louis nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL WASHINGTON UNIVERSITY BE LIABLE FOR ANY 
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef LASP_SVM_H
#define LASP_SVM_H

#include "options.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <ctime>
#include <map>

#include "my_stopwatch.h"


#define CHUNKSIZE 1000
#define NUM_ASYNCS 10

using namespace std;

namespace lasp{
  enum errors{ CORRECT, UNOPENED_FILE_ERROR, WRONG_NUMBER_ARGS, CUBLAS_ERROR, MISMATCH_ERROR, INVALID_INPUT };
  enum arg { FILE_IN, C_IN, GAMMA_IN, DONE };
  #define LINE_SIZE 100000

  struct svm_node{
    int index;
    double value;
    
    friend bool operator==(const svm_node& a,const svm_node& b) {return (a.index == b.index
									 &&
									 a.value == b.value);}
    friend bool operator!=(const svm_node& a,const svm_node& b) {return (a.index != b.index
									 ||
									 a.value != b.value);}

    friend bool operator<(const svm_node& a, const svm_node& b) {
      if(a.index != b.index)
	return a.index < b.index;
      if(a.value != b.value)
	return a.value < b.value;
      return false;
    }
  };

  //struct used to compare vectors of svm_nodes.
  struct CompareSparseVectors {
    bool operator()(const vector<svm_node> a, const vector<svm_node> b) const
    {
      if(a.size() != b.size()) return a.size() < b.size();
      else {
	for(int i = 0; i < a.size(); ++i) {
	  if (a[i] == b[i]) continue;
	  else return a[i] < b[i];
	}
      }
      return false;
    }
  };

  //This struct represents a two-class svm_problem that we solve with
  //the lasp_svm function.
  struct svm_problem{
    //These must be set BEFORE solving the svm_problem
    int n, features;
    opt options;
    double *y;
    double *xS;
    //vector of length 2 that holds the names of the classes in this problem.
    vector<int> classifications;

    //These are set while you solve the svm_problem
    vector<int> S;
    vector<double> time;
    vector<double> tre;
    vector<double> obj;
    vector<double> bs;
    vector<vector<double> > betas;
  };


  //Struct used for storing model parameters when loading a model
  //from a file.
  struct svm_model {

    //The type of kernel used in this model
    int kernelType;
    int numFeatures;
    //These are all parameters relating to how the kernel is computed
    //they are set based on the kernel type provided, so they don't
    //always all have values.
    double degree;
    double coef;
    double gamma;
	
	opt options;

    vector<int> orderSeen;

    //vector containing the offset values.
    //maps from {class1 -> {class2 -> offset}} where class1 was seen before class2
    map<int, map<int, double> > offsets;
    
    //This data structure maps {class label -> {sparse support vector -> {class label -> beta}}}
    map<int, map<vector<svm_node>, map<int, double>, CompareSparseVectors> > modelData;
    
    //Does this model incorporate platt scaling?
    int plattScale;
    //This struct keeps track of the A,B pairs for each trained sigmoid for platt scaling
    map<int, map<int, pair<double, double> > > plattScaleCoefs;

  };

  //a binary svm model, meaning a 1 vs1 classification model.
  struct svm_binary_model {
    //The type of kernel used in this model
    int kernelType;

    int numFeatures;
    //These are all parameters relating to how the kernel is computed
    //they are set based on the kernel type provided, so they don't
    //always all have values.
    float degree;
    float coef;
    float gamma;

    int numSupportVectors;
    float b; //Offset parameter

    //ordered weighting parameters for each support vector
    //after loading a model, the length of this vector should
    //equal numSupportVectors.
    float* betas;

    //The values of the support vectors, stored in the "normal" way.
    //The first numFeatures elements correspond to the first support
    //vector. Ie if numSupportVectors = 3 and numFeatures = 2,
    //then if xS -> [5,6,7,5,3,6], the first support vector
    //would be [5,6], the second would be [7,5] and the third
    //would be [3,6].
    float* xS;

  };


  //Struct used to store the full versions of the data files.
  //Members of this struct will be used to actually call the
  //svm computation functions.
  struct svm_full_data {
    //The non-sparse version of the data we'd like to classify,
    //stored in the "normal" 1-D array way.
    float* x;
    //The classification provided by the user. According to libsvm
    //standards, the user must provide these, even if the data
    //is technically "unlabeled" (putting a 0 in front of each
    //data point suffices).
    float* y;

    //the number of features in the data set
    int numFeatures;

    //the number of data points you have
    int numPoints;
  };

  //Struct that holds the sparse version of all the data.
  struct svm_sparse_data {
    //allData is a map that maps from
    //classificationID -> vector<vector<svm_node> >
    //where classificationID is the class value the user has input for a given piece of data
    //and the vector of vector<svm_node> is a list of all the points of that given classification
    //the vector of svm_nodes represent a singluar sparse representation of a data point.
    map<int, vector<vector<svm_node> > > allData;

    //the order in which the classifications were seen in the data input.
    vector<int> orderSeen;

    //Tracks the number of features in this sparse dataset.
    int numFeatures;

    //keeps track of whether or not this dataset has more than two classes.
    bool multiClass;

    int numPoints;
  };

  //struct used for keeping track of timing
  struct svm_time_recorder {
    //maps from {name of timing section -> [time1, time2, ...]
    map<string, vector<double> > times;
  };
  
  //struct used to time segments of code.
  struct svm_timer {
    char* name;
    //struct timespec start, finish;
	MyStopWatch cl;
    double time;
  };

  void normalize_host(double*, int, int, double*, double=-1);
  void gather_host(int*, double*, double*, int, int);


  double* cublas_matmult_pinned(double*, int, int ,int ,int);

  template<class T>
  int lasp_svm_host(svm_problem&);
  //The timed version of the above function

  void tempOutputCheck(svm_node**,double*);
  void printMatrix(int, int, double*,int=0);
  vector<int> mysetdiff(vector<int>& a, vector<int>& b);

  //void uTest1();
  float* double_to_float(double*, int);
  double* float_to_double(float*, int);

  //This method takes in a reference to an svm_problem,
  //a sparse dataset myData, and an options object
  //and sets up the svm problem appropriately.
  //Note that myData can only contain TWO classes
  //of data. An svm_problem can only represent single-class
  //classification problems. The data in problem will be
  //unsorted. Also, puts 30% of the training data into
  //holdout data, which is NOT used for training the SVM,
  //but rather for training the sigmoid later.
  void setup_svm_problem_shuffled(svm_problem& problem,
				  svm_sparse_data myData,
				  svm_sparse_data& holdoutData,
				  opt options,
				  int negClass,
				  int posClass);

  //Non platt-scale version of the above method.
  void setup_svm_problem_shuffled(svm_problem& problem,
				  svm_sparse_data myData,
				  opt options,
				  int negClass,
				  int posClass);
}

#endif
