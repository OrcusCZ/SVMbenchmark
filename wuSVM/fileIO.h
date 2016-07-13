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

#ifndef LASP_FILEIO_H
#define LASP_FILEIO_H

#include "lasp_svm.h"

namespace lasp
{
  //Given the svm_model, writes a libsvm style output file
  int write_model(svm_model, char* filename);

  //Given a vector of solved svm_problems, and an integer
  //vector representing the order in which the classes were seen
  //in the training data, returns an associated svm_model object.
  svm_model get_model_from_solved_problems(vector<svm_problem> solvedProblems,
					   vector<svm_sparse_data> holdouts,
					   vector<int> orderSeen);

  void write_header(svm_model model,
		    ofstream& fout);

  
  void write_support_vectors(svm_model myModel,
			     ofstream& fout);


  //given a path to a libsvm style data file and
  //a reference to myData, a svm_sparse_data object, loads
  //all appropriate data into myData.
  int load_sparse_data(char* filename,
		       svm_sparse_data& myData);


  //given a path to a libsvm style model file,
  //a reference to an svm_model object, parses the
  //file at filename and outputs the support vectors
  //in sparse form in sparseOutput, and the number
  //of features in numFeaturesFound. Sets all
  //model parameters except myModel.xS and myModel.numFeatures
  int load_model(char* filename,
		 svm_model& myModel);
  
  //Given a svm_time_recorder and an ouput file name,
  //outputs the timing data to that file.
  void write_time_data(svm_time_recorder timer,
		       char* filename);


  //Given an output file and an interger vector of predictions
  //outputs the predictions int
  void output_classifications(char* filename,
			      vector<int> const& predictions);

  //Given a vector of svm_nodes and the number of features
  //in the data set, puts the full, non-sparse output into
  //the output array.
  void sparse_vector_to_full(vector<double>& output,
			    vector<svm_node> input,
			    int numFeatures);

  //Given a float array to output into, copies all the values
  //from the input to the output after creating a new output
  //array.
  void double_vector_to_array(double*& output,
			      vector<double> input);


  void double_vector_to_float_array(float*& output,
				    vector<double> input);


  void double_vector_to_float_array_two_dim(float*& output,
					    vector<vector<double> > input);


  void sparse_data_to_full(svm_full_data& fullData,
			   svm_sparse_data sparseData);


  //Given a sparse data set, the two classes that you want to
  //use for your problem and an options struct, return an svm
  //problem that is set up with only examples from those two
  //classes. Also, fills holdoutData with 30% of the potential
  //training data for platt scaling.
  svm_problem get_onevsone_subproblem(svm_sparse_data myData,
				      svm_sparse_data& holdoutData,
				      int class1,
				      int class2,
				      opt options);

  
  //the non-platt scaled version of the above.
  svm_problem get_onevsone_subproblem(svm_sparse_data myData,
				      int class1,
				      int class2,
				      opt options);
}
#endif
