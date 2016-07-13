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

#include "fileIO.h"
#include "lasp_svm.h"

int lasp_atoi(const char* input){
	char *endptr;
	int result = static_cast<int>(strtol(input, &endptr, 10));
	
	if(result == 0 && endptr == input){
		cout << "Unrecoverable error reading data" << endl;
		std::exit(1);
	}
	
	return result;
}

double lasp_atof(const char* input){
	char *endptr;
	double result = static_cast<double>(strtod(input, &endptr));
	
	if(result == 0.0 && endptr == input){
		cout << "Unrecoverable error reading data" << endl;
		std::exit(1);
	}
	
	return result;
}

int lasp::load_sparse_data(char* filename,
						   lasp::svm_sparse_data& myData)
{
	int pointCount = 0;
	ifstream fin;
	fin.open(filename);
	if(fin) {
		bool zeroIndexed = false;
		int maxIndex = -1;
		char curLine[LINE_SIZE];
		
		while(fin.getline(curLine, LINE_SIZE)) {
			vector<svm_node> currentNodes;
			char* tokens = strtok(curLine, " ");
			int classification = lasp_atoi(tokens);
			bool seen = false;
			for(int i = 0; i < myData.orderSeen.size(); ++i) {
				if(classification == myData.orderSeen[i]) {
					seen = true;
					break;
				}
			}
			
			if (!seen) myData.orderSeen.push_back(classification);
			tokens = strtok(NULL, " "); //move forward one token
			while(tokens) {
				//parsing stuff, kind of messy, but gets job done.
				string oneToken;
				oneToken.assign(tokens);
				size_t found = oneToken.find(":");
				string index = oneToken.substr(0,found);
				if (lasp_atoi(index.c_str()) == 0){
					zeroIndexed = true;
				}
				string value = oneToken.substr(found+1, oneToken.length());
				
				svm_node newNode;
				newNode.index = lasp_atoi(index.c_str());
				newNode.value = lasp_atof(value.c_str());
				if(newNode.index > maxIndex) maxIndex = newNode.index;
				currentNodes.push_back(newNode);
				tokens = strtok(NULL, " ");
			}
			myData.allData[classification].push_back(currentNodes);
			pointCount ++;
		}
		myData.numPoints = pointCount;
		
		//I hate this piece of code, but I don't want to try to figure out a better way to deal with zero-indexed files (Gabe)
		if(zeroIndexed){
			maxIndex++;
			for(map<int, vector<vector<svm_node> > >::iterator iter = myData.allData.begin(); iter != myData.allData.end(); ++iter){
				for(vector<vector<svm_node> >::iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2){
					for(vector<svm_node>::iterator iter3 = iter2->begin(); iter3 != iter2->end(); ++iter3){
						iter3->index++;
					}
				}
			}
		}
		
		myData.numFeatures = maxIndex;
		myData.multiClass = myData.allData.size() > 2;
		fin.close();
		return 0;
	}
	else {
		cout << " The file \' "<< filename << "\' was not found." << endl;
		cout << "You must specify an existing file to read your" << endl;
		cout << "training data from." << endl;
		return UNOPENED_FILE_ERROR;
	}
}


int lasp::load_model(char* filename,
					 svm_model& myModel)
{
	//We need to first parse the header file, which is kind
	//of a pain, but I'll maintain a boolean doneWithHeader
	//that is true if we are done parsing the header.
	ifstream fin;
	fin.open(filename);
	if(fin.is_open()) {
		int maxIndex = -1;
		//do some checks to make sure the model file is valid.
		//could be improved, but probably catches 99% of invalid
		//model files.
		bool validHeader = false;
		bool kernelType = false;
		bool numSV = false;
		bool label = false;
		
		bool doneWithHeader = false;
		char curLine[LINE_SIZE];
		//the current support vector we're parsing, indexed from 1.
		int curNumSupportVector = 1;
		//the number of classes specified in the header file.
		int numClasses = -1;
		//map from {class -> num support vectors in that class
		map<int, int> supportVectorCounts;
		
		//the vector of offsets specified by "rho"
		//we need a temporary vector because we don't know the class
		//labels at the point we read these.
		vector<double> tempOffsetVector;
		
		//the vector of platt scale coefficients A,B for each 1v1 problem.
		//we need a temporary vector because we don't know the class
		//labels at the point we read these.
		vector<double> tempPlattA;
		vector<double> tempPlattB;
		
		myModel.plattScale = 0;
		//While looping through, I will also track the maximum
		//svm_node index to see how many features we have.
		while(fin.getline(curLine, LINE_SIZE)) {
			//HERE is where we parse the header
			if(!doneWithHeader) {
				char* tokens = strtok(curLine, " ");
				if(strcmp(tokens, "svm_type") == 0) {
					tokens = strtok(NULL, " ");
					if(strcmp(tokens, "c_svc") != 0) {
						return INVALID_INPUT;
					}
				}
				else if(strcmp(tokens, "kernel_type") == 0){
					tokens = strtok(NULL, " ");
					kernelType = true;
					if(strcmp(tokens, "rbf") == 0) {
						myModel.kernelType = RBF;
					}
					else if(strcmp(tokens, "linear") == 0) {
						myModel.kernelType = LINEAR;
					}
					else if(strcmp(tokens, "polynomial") == 0) {
						myModel.kernelType = POLYNOMIAL;
					}
					else if(strcmp(tokens, "sigmoid") == 0) {
						myModel.kernelType = SIGMOID;
					}
					else if(strcmp(tokens, "precomputed") == 0) {
						myModel.kernelType = PRECOMPUTED;
					}
					else {
						cout << tokens << " is not a valid kernel type" << endl;
						return INVALID_INPUT;
					}
				}
				else if(strcmp(tokens, "gamma") == 0) {
					tokens = strtok(NULL, " ");
					myModel.gamma = lasp_atof(tokens);
				}
				else if(strcmp(tokens, "coef0") == 0) {
					tokens = strtok(NULL, " ");
					myModel.coef = lasp_atof(tokens);
				}
				else if(strcmp(tokens, "degree") == 0) {
					tokens = strtok(NULL, " ");
					myModel.degree = lasp_atof(tokens);
				}
				else if(strcmp(tokens, "rho") == 0) {
					for(int i = 0; i < (numClasses*(numClasses-1))/2; ++i) {
						tokens = strtok(NULL, " ");
						tempOffsetVector.push_back(-lasp_atof(tokens));
					}
				}
				else if(strcmp(tokens, "probA") == 0) {
					myModel.plattScale = 1; //set the platt scaling flag in model
					for(int i = 0; i < numClasses; ++i) {
						tokens = strtok(NULL, " ");
						tempPlattA.push_back(lasp_atof(tokens));
					}
				}
				else if(strcmp(tokens, "probB") == 0) {
					for(int i = 0; i < numClasses; ++i) {
						tokens = strtok(NULL, " ");
						tempPlattB.push_back(lasp_atof(tokens));
					}
				}
				else if(strcmp(tokens, "nr_class") == 0) {
					tokens = strtok(NULL, " ");
					numClasses = lasp_atoi(tokens);
				}
				else if(strcmp(tokens, "label") == 0) {
					label = true;
					for(int i = 0 ; i < numClasses; ++i) {
						tokens = strtok(NULL, " ");
						int curClass = lasp_atoi(tokens);
						myModel.orderSeen.push_back(curClass);
					}
				}
				else if(strcmp(tokens, "nr_sv") == 0) {
					numSV = true;
					for(int i = 0; i < numClasses; ++i) {
						tokens = strtok(NULL, " ");
						supportVectorCounts[myModel.orderSeen[i]] = lasp_atoi(tokens);
					}
				}
				else if(strcmp(tokens, "SV") == 0) {
					doneWithHeader = true;
				}
				else if(strcmp(tokens, "total_sv") == 0) {
					tokens = strtok(NULL, " ");
				}
				else {
					cout << tokens << " is not a valid header in a model file." << endl;
					return INVALID_INPUT;
				}
			}
			//HERE is where we parse the support vectors
			else {
				//check to make sure we specified some required parameters in the header
				validHeader = (label && numSV && kernelType);
				if(!validHeader) {
					cout << "model header incomplete, make sure you include all required headers" << endl;
					return INVALID_INPUT;
				}
				
				vector<svm_node> currentNodes;
				char* tokens = strtok(curLine, " ");
				int curClass;
				//variable used to determine what class the current support vector is in.
				int runningSVSum = 0;
				for(int i = 0; i < numClasses; ++i) {
					runningSVSum += supportVectorCounts[myModel.orderSeen[i]];
					if(runningSVSum >= curNumSupportVector) {
						curClass = myModel.orderSeen[i];
						break;
					}
				}
				//now that we know what class we're in, we'll load the betas in
				//using a temporary storage vector (we'll input them to the map once
				//we've actually loaded the SV).
				vector<double> tempBetaVec;
				for(int i = 0; i < numClasses-1; ++i) {
					tempBetaVec.push_back(lasp_atof(tokens));
					tokens = strtok(NULL, " ");
				}
				
				while(tokens) {
					//parsing stuff, kind of messy, but gets job done.
					string oneToken;
					oneToken.assign(tokens);
					size_t found = oneToken.find(":");
					string index = oneToken.substr(0,found);
					string value = oneToken.substr(found+1, oneToken.length());
					
					svm_node newNode;
					newNode.index = lasp_atoi(index.c_str());
					newNode.value = lasp_atof(value.c_str());
					if(newNode.index > maxIndex) { maxIndex = newNode.index; }
					
					currentNodes.push_back(newNode);
					tokens = strtok(NULL, " ");
				}
				int curBetaVecIndex = 0;
				for(int i = 0; i < numClasses; ++i) {
					int otherClass = myModel.orderSeen[i];
					if(curClass == otherClass) continue;
					myModel.modelData[curClass][currentNodes][otherClass] = tempBetaVec[curBetaVecIndex];
					curBetaVecIndex++;
				}
				curNumSupportVector++;
			}
		}
		myModel.numFeatures = maxIndex;
		int curIndex = 0;
		for(int i = 0; i < numClasses; ++i) {
			for(int j = i+1; j < numClasses; ++j) {
				int c1, c2;
				c1 = myModel.orderSeen[i]; c2 = myModel.orderSeen[j];
				myModel.offsets[c1][c2] = tempOffsetVector[curIndex];
				if(tempPlattA.size() != 0) {
					pair<double, double> curPair;
					curPair.first = tempPlattA[curIndex]; curPair.second = tempPlattB[curIndex];
					myModel.plattScaleCoefs[c1][c2] = curPair;
				}
				curIndex++;
			}
		}
		fin.close();
		return 0;
	}
	else {
		cout << filename << " not found." << endl;
		return UNOPENED_FILE_ERROR;
	}
}


int lasp::write_model(svm_model myModel, char* filename)
{
	ofstream fout;
	fout.open(filename);
	if(fout.is_open()) {
		write_header(myModel, fout);
		write_support_vectors(myModel, fout);
		fout.close();
		return 0;
	}
	else {
		return UNOPENED_FILE_ERROR;
	}
}

void lasp::write_header(svm_model model,
						ofstream &fout)
{
	//Here, we write the header file for libsvm models.
	//do we need to support more model types?
	fout << "svm_type c_svc" << endl;
	switch(model.kernelType)
	{
			case RBF:
			fout << "kernel_type rbf" << endl;
			fout << "gamma " << model.gamma << endl;
			break;
			case LINEAR:
			fout << "kernel_type linear" << endl;
			break;
			case POLYNOMIAL:
			fout << "kernel_type polynomial" << endl;
			fout << "degree " << model.degree << endl;
			fout << "gamma " << model.gamma << endl;
			fout << "coef0 " << model.coef << endl;
			break;
			case SIGMOID:
			fout << "kernel_type sigmoid" << endl;
			fout << "gamma " << model.gamma << endl;
			fout << "coef0 " << model.coef << endl;
			break;
			case PRECOMPUTED:
			fout << "kernel_type precomputed" << endl;
			//TODO: This needs to be written, I don't know what this would look like
			//because I can't get libsvm to work with a precomputed kernel at the moment.
			break;
			default:
			fout << endl;
	}
	fout << "nr_class " << model.modelData.size() << endl;
	
	int totalSV = 0;
	//this map maps from {class -> numSupportVectors}
	map<int, int> svClassMapping;
	
	for(int i = 0; i < model.orderSeen.size(); ++i) {
		totalSV += model.modelData[model.orderSeen[i]].size();
		svClassMapping[model.orderSeen[i]] = model.modelData[model.orderSeen[i]].size();
	}
	
	
	fout << "total_sv "<< totalSV << endl;
	fout << "rho ";
	for(int i = 0; i < model.orderSeen.size(); ++i) {
		for(int j = i + 1; j < model.orderSeen.size(); ++j) {
			int c1 = model.orderSeen[i]; int c2 = model.orderSeen[j];
			if (model.offsets[c1][c2] != 0) {
				fout << -model.offsets[c1][c2] << " ";
			}
			else {
				fout << "0 ";
			}
			
		}
	}
	fout << endl;
	fout << "label ";
	for(int i = 0; i < model.orderSeen.size(); ++i) {
		fout << model.orderSeen[i];
		if(i != model.orderSeen.size() - 1) fout << " ";
	}
	fout << endl;
	fout << "nr_sv ";
	for(int i = 0; i < model.orderSeen.size(); ++i) {
		int curClass = model.orderSeen[i];
		fout << svClassMapping[curClass];
		if(i != svClassMapping.size()-1) fout << " ";
	}
	fout << endl;
	
	if(model.plattScale) {
		fout << "probA";
		for(int i = 0; i < model.orderSeen.size(); ++i) {
			for(int j = i + 1; j < model.orderSeen.size(); ++j) {
				int c1 = model.orderSeen[i]; int c2 = model.orderSeen[j];
				fout << " " << model.plattScaleCoefs[c1][c2].first;
			}
		}
		fout << endl;
		
		fout << "probB";
		for(int i = 0; i < model.orderSeen.size(); ++i) {
			for(int j = i + 1; j < model.orderSeen.size(); ++j) {
				int c1 = model.orderSeen[i]; int c2 = model.orderSeen[j];
				fout << " " << model.plattScaleCoefs[c1][c2].second;
			}
		}
		fout << endl;
	}
	
	
	fout << "SV" << endl;
}

void lasp::write_support_vectors(svm_model myModel,
								 ofstream& fout)
{
	for(int i = 0; i < myModel.orderSeen.size(); ++i) {
		int curClassification = myModel.orderSeen[i];
		//first write the betas, in the order seen
		typedef map<vector<svm_node>, map<int, double>, CompareSparseVectors>::iterator SVIter;
		for(SVIter iter = myModel.modelData[curClassification].begin();
			iter != myModel.modelData[curClassification].end();
			++iter) {
			//first write the betas
			for(int j = 0; j < myModel.orderSeen.size(); ++j) {
				int otherClassification = myModel.orderSeen[j];
				if(curClassification == otherClassification) continue;
				fout << iter->second[otherClassification] << " ";
			}
			//now write the actual SV
			for(int j = 0; j < iter->first.size(); ++j) {
				fout << iter->first[j].index << ":" << iter->first[j].value;
				if(j != iter->first.size()-1) fout << " ";
			}
			fout << endl;
		}
	}
}

void lasp::write_time_data(svm_time_recorder timer,
						   char* filename)
{
	ofstream fout;
	fout.open(filename);
	if(fout.is_open())
	{
		typedef map<string, vector<double> >::iterator timeIterator;
		fout << "All times are in ms" << endl;
		for(timeIterator myIter = timer.times.begin();
			myIter != timer.times.end();
			++myIter)
		{
			double total = 0.0;
			fout << myIter->first << ": ";
			for(int i = 0; i < myIter->second.size(); ++i)
			{
				total += myIter->second[i];
			}
			fout << total << " in " << myIter->second.size() << " calls" << endl;
			int to = min(10, (int)myIter->second.size());
			if (to == 10) fout << "truncated output..." << endl;
			for(int i = 0; i < to; ++i)
			{
				fout << myIter->second[i] << endl;
			}
			fout << endl;
		}
		fout.close();
	}
}


void lasp::sparse_vector_to_full(vector<double>& output,
								 vector<svm_node> input,
								 int numFeatures)
{
	output.assign(numFeatures,0.0);
	for(int i = 0; i < input.size(); ++i) {
		//The -1 here is because the file indexing is one greater
		//than the array indexing.
		output[input[i].index-1] = input[i].value;
	}
}

void lasp::double_vector_to_array(double*& output,
								  vector<double> input)
{
	output = new double[input.size()];
	for(int i = 0; i < input.size(); ++i) {
		output[i] = input[i];
	}
}

void lasp::double_vector_to_float_array(float*& output,
										vector<double> input)
{
	output = new float[input.size()];
	for(int i = 0; i < input.size(); ++i) {
		output[i] = input[i];
	}
}

void lasp::double_vector_to_float_array_two_dim(float*& output,
												vector<vector<double> > in)
{
	//this requires that all double vectors in input have the same
	//length. A bit sketchy, but it will work here. Also, this
	//assumes that you have at least one double vector.
	output = new float[in.size()*in[0].size()];
	for(int i = 0; i < in.size(); ++i) {
		for(int j = 0; j < in[i].size(); ++j) {
			output[i*in[0].size()+j] = in[i][j];
		}
	}
}

void lasp::output_classifications(char* filename,
								  vector<int> const& predictions)
{
	ofstream myOut;
	myOut.open(filename);
	for(int i = 0; i < predictions.size(); ++i) {
		myOut << predictions[i] << endl;
	}
	myOut.close();
}


void lasp::sparse_data_to_full(svm_full_data& fullData,
							   svm_sparse_data sparseData)
{
	vector<double> fullClassifications;
	vector<vector<double> > fullVectorData;
	typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
	for(SparseIterator iter = sparseData.allData.begin();
		iter != sparseData.allData.end();
		++iter) {
		for(int i = 0; i < iter->second.size(); ++i) {
			int curSparseIndex = 0;
			vector<svm_node> curSparse = iter->second[i];
			vector<double> curFull;
			for(int j = 0; j < sparseData.numFeatures; ++j) {
				if(curSparse.size() > 0 && curSparse[curSparseIndex].index == j+1) {
					curFull.push_back(curSparse[curSparseIndex].value);
					curSparseIndex++;
				}
				else {
					curFull.push_back(0);
				}
			}
			fullClassifications.push_back(iter->first);
			fullVectorData.push_back(curFull);
		}
	}
	double_vector_to_float_array_two_dim(fullData.x, fullVectorData);
	double_vector_to_float_array(fullData.y, fullClassifications);
	fullData.numFeatures = sparseData.numFeatures;
	fullData.numPoints = fullClassifications.size();
}

lasp::svm_model lasp::get_model_from_solved_problems(vector<lasp::svm_problem> solvedProblems,
													 vector<lasp::svm_sparse_data> holdoutData,
													 vector<int> orderSeen)
{
	
	svm_model returnModel;
	returnModel.orderSeen = orderSeen;
	//First, we need to fill up returnMode.idMap by assigning an id to each support vector
	
	//First, we will create a mapping from {class1 -> {class2 -> 1vs2}}
	//where we saw class1 before class2 and 1vs2 is an svm_problem that represents
	//the solution to pitting class1 against class2.
	map<int, map<int, svm_problem> > comparisonMapping;
	
	for(int i = 0; i < solvedProblems.size(); ++i) {
		int c1 = solvedProblems[i].classifications[0];
		int c2 = solvedProblems[i].classifications[1];
		//Enforce that the model contains all classes
		returnModel.modelData[c1];
		returnModel.modelData[c2];
		int c1ind = -1;
		int c2ind = -1;
		for(int j = 0; j < orderSeen.size(); ++j) {
			if(c1 == orderSeen[j]) c1ind = j;
			else if(c2 == orderSeen[j]) c2ind = j;
		}
		//swap to make sure we're indexing our map correctly
		if(c1ind > c2ind) {
			int temp = c1;
			c1 = c2;
			c2 = temp;
		}
		
		comparisonMapping[c1][c2] = solvedProblems[i];
	}
	
	//Now that we have the mapping and the sorted list of classes we've seen, we can
	//begin to fill in the model.
	
	//This assumes that we have at least one solved problem
	//this is pretty ugly, there's definitely a better way to do this.
	returnModel.kernelType = solvedProblems[0].options.kernel;
	returnModel.numFeatures = solvedProblems[0].features;
	returnModel.degree = solvedProblems[0].options.degree;
	returnModel.coef = solvedProblems[0].options.coef;
	returnModel.gamma = solvedProblems[0].options.gamma;
	
	//Here's the big data-copy loop.
	for(int i = 0; i < orderSeen.size(); ++i) {
		for(int j = i+1; j < orderSeen.size(); ++j) {
			int c1, c2;
			c1 = orderSeen[i]; c2 = orderSeen[j];
			
			svm_problem currentProblem = comparisonMapping[c1][c2];
			returnModel.offsets[c1][c2] = currentProblem.bs.back();
			//Now, let's extract the support vectors from each class
			vector<vector<svm_node> > classOneSV;
			vector<vector<svm_node> > classTwoSV;
			vector<double> classOneBetas;
			vector<double> classTwoBetas;
			
			for(int k = 0; k < currentProblem.S.size(); ++k) {
				int curClassification = currentProblem.y[currentProblem.S[k]] == 1 ?
				currentProblem.classifications[0]
				: currentProblem.classifications[1];
				
				vector<svm_node> currentSupportVector;
				//Note that svm_nodes are indexed from 1.
				for(int x = 0; x < currentProblem.features; ++x) {
					double value = currentProblem.xS[k*currentProblem.features+x];
					if(fabs(value) > 1E-20) {
						svm_node newNode;
						newNode.index = x + 1;
						newNode.value = value;
						//cout << newNode.index << "," << newNode.value << endl;
						currentSupportVector.push_back(newNode);
					}
				}
				
				//we have reconstructed the sparse representation of a given support vector,
				//now we just need to put it into the correct list of support vectors
				if(curClassification == c1) {
					classOneSV.push_back(currentSupportVector);
					classOneBetas.push_back(currentProblem.betas.back()[k]);
				}
				else { //assume it's in class two.
					classTwoSV.push_back(currentSupportVector);
					classTwoBetas.push_back(currentProblem.betas.back()[k]);
				}
				
			}
			
			//we now have two vectors containing the sparse representations for each class
			//and two more vectors representing the corresponding beta values.
			//Now, we need to put the data into the svm_model's modelData.
			
			typedef map<vector<svm_node>, map<int, double>, CompareSparseVectors>::iterator MyIter;
			
			for(int k = 0; k < classOneSV.size(); ++k) {
				returnModel.modelData[c1][classOneSV[k]][c2] += classOneBetas[k];
			}
			for(int k = 0; k < classTwoSV.size(); ++k) {
				returnModel.modelData[c2][classTwoSV[k]][c1] += classTwoBetas[k];
			}
			
			//TODO: This should probably be referenced counted or something
			//delete [] currentProblem.xS;
			//currentProblem.xS = 0;
		}
		
	}
	
	returnModel.plattScale = (holdoutData.size() != 0);
	if(returnModel.plattScale) {
		//now, we train the sigmoid parameters.
		//First, lets create a mapping from {c1 -> {c2 -> holdout data}}
		//where we saw c1 before c2.
		map<int, map<int, svm_sparse_data> > holdoutMapping;
		for(int i = 0; i < holdoutData.size(); ++i) {
			vector<int> holdoutClasses;
			svm_sparse_data curHoldout = holdoutData[i];
			for(map<int, vector<vector<svm_node> > >::iterator iter = curHoldout.allData.begin();
				iter != curHoldout.allData.end();
				++iter) {
				holdoutClasses.push_back(iter->first);
			}
			if(holdoutClasses.size() != 2) cout << "holdout data had more/less than 2 classes." << endl;
			int c1 = holdoutClasses[0];
			int c2 = holdoutClasses[1];
			int c1ind = -1;
			int c2ind = -1;
			for(int j = 0; j < orderSeen.size(); ++j) {
				if(c1 == orderSeen[j]) c1ind = j;
				else if(c2 == orderSeen[j]) c2ind = j;
			}
			//swap to make sure we're indexing our map correctly
			if(c1ind > c2ind) {
				int temp = c1;
				c1 = c2;
				c2 = temp;
			}
			holdoutMapping[c1][c2] = curHoldout;
		}
		
		for(int i = 0; i < orderSeen.size(); ++i) {
			for(int j = i + 1; j < orderSeen.size(); ++j) {
				svm_sparse_data curHoldout = holdoutMapping[orderSeen[i]][orderSeen[j]];
				pair<double, double> curPair(1.0, 1.0); //= get_optimal_sigmoid_parameters(curHoldout, returnModel);
				returnModel.plattScaleCoefs[orderSeen[i]][orderSeen[j]] = curPair;
			}
		}
	}
	
	
	return returnModel;
}



lasp::svm_problem lasp::get_onevsone_subproblem(svm_sparse_data myData,
												svm_sparse_data& holdoutData,
												int class1,
												int class2,
												opt options)
{
	svm_problem returnProblem;
	
	svm_sparse_data tempSparseData;
	tempSparseData.numFeatures = myData.numFeatures;
	tempSparseData.multiClass = myData.multiClass;
	
	tempSparseData.allData[class1] = myData.allData[class1];
	tempSparseData.allData[class2] = myData.allData[class2];
	//class 1 is the positive class, class 2 is the negative class.
	setup_svm_problem_shuffled(returnProblem, tempSparseData, holdoutData, options, class1, class2);
	
	return returnProblem;
}


lasp::svm_problem lasp::get_onevsone_subproblem(svm_sparse_data myData,
												int class1,
												int class2,
												opt options)
{
	svm_problem returnProblem;
	
	svm_sparse_data tempSparseData;
	tempSparseData.numFeatures = myData.numFeatures;
	tempSparseData.multiClass = myData.multiClass;
	
	tempSparseData.allData[class1] = myData.allData[class1];
	tempSparseData.allData[class2] = myData.allData[class2];
	//class 1 is the positive class, class 2 is the negative class.
	setup_svm_problem_shuffled(returnProblem, tempSparseData, options, class1, class2);
	
	return returnProblem;
}
