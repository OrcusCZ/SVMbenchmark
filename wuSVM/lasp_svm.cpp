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

#include "lasp_svm.h"
#include "parsing.h"
#include "retraining.h"
#include "kernels.h"
#include "train_subset.h"
#include "next_point.h"
#include "hessian.h"
#include "fileIO.h"
#include <algorithm>
#include "lasp_matrix.h"
#include <cstdlib>
#include <ctime>
#include <cfloat>

namespace lasp{
	template<class T>
	int lasp_svm_host(svm_problem& p) {
		int stopIters = 0;
		
		srand (time(0));
		bool gpu = p.options.usegpu;
				
		//Might want to move this check elsewhere (i.e. to parsing.cpp)
		if (gpu && DeviceContext::instance()->getNumDevices() < 1) {
			cerr << "No CUDA device found, reverting to CPU-only version" << endl;
			p.options.usegpu = false;
			gpu = false;
		} else if(!gpu){
			DeviceContext::instance()->setNumDevices(0);
		} else if (p.options.maxGPUs > -1){
			DeviceContext::instance()->setNumDevices(p.options.maxGPUs);
		}
				
		//Timing Variables
		time_t baseTime = time(0);
		
		//kernel options struct for computing the kernel
		kernel_opt kernelOptions;
		kernelOptions.kernel = p.options.kernel;
		kernelOptions.gamma = p.options.gamma;
		kernelOptions.degree = p.options.degree;
		kernelOptions.coef = p.options.coef;
		
		//Select iterations for full retraining
		vector<int> R;
		R = retrainIters(p);
		
		//If Dataset is small, prepare to solve it all at once
		bool smallDataSet(false);
		if (p.n < 3000){
			R.clear();
			R.push_back(0);
			R.push_back(p.n);
			smallDataSet=true;
		}
		
		//Training examples
		LaspMatrix<T> x = LaspMatrix<double>(p.n, p.features, p.xS).convert<T>();
		
		//Training labels
		LaspMatrix<T> y= LaspMatrix<double>(p.n,1,p.y).convert<T>();
	
		LaspMatrix<T> K;
		
		//Kernel is generally the largest allocation, if its allocation fails, we may need to reduce our set size
        while (p.options.set_size > 0 && !p.options.smallKernel) {
			try {
				if (p.options.start_size > p.options.set_size) {
					p.options.start_size = p.options.set_size;
				}
				
				K.resize(p.n, p.options.set_size + 1, false, false);
			} catch (bad_alloc) {
				p.options.set_size /= 2;
				cerr << "Kernel failed to allocate, reducing max training set to: " << p.options.set_size << endl;
				continue;
			}
			break;
		}
		
		
		if(!p.options.smallKernel){
			K.setRow(0,y);
		}
		
		LaspMatrix<T> K_temp;
		
		LaspMatrix<T> Ksum(1,p.options.set_size + 1,0.0);
		
		{
			for(int i = 0; i < p.n; i++){
				Ksum(0,0) += y(i);
			}
		}
		
		//contains support vectors
		vector<int> S;
		
		//Training examples still incorrectly classified
		vector<int> * erv = new vector<int>();
		
		int out_minus1Length = p.n;
		int xInd = 0;
		
		LaspMatrix<T> HESS(1,1, 0.0,(R.back()+1),(R.back()+1));
		HESS(0,0) = p.options.C * p.n * (1 + 1e-20);
		int ldHESS = (R.back()+1);
		int sizeHESS = 1;
		
		LaspMatrix<T> old_out(1,p.n, 0.0);
		LaspMatrix<T> out(1,p.n, 0.0);
		
		LaspMatrix<T> w(0, 0);
		LaspMatrix<T> x_old(1, R.back()+1, 0.0);
		int x_old_size(0);
		
		//INITIALIZE TRACE VARIABLES: tre, time, obj, betas and bs
		LaspMatrix<T> xNorm (p.n,1, 0.0);
		LaspMatrix<T> xS(0,0,0.0,p.options.set_size,p.features);
		LaspMatrix<T> xnormS(0,0,0.0,p.options.set_size, 1);
		LaspMatrix<T> xerv;
		LaspMatrix<T> yerv;
		LaspMatrix<T> xnormerv;
		LaspMatrix<T> out_minus1 = LaspMatrix<T>(1,p.n);
		
		if (gpu){
			//Check that all gpu allocations succeed
			int error = 0;
			error += x.transferToDevice();
			error += y.transferToDevice();
			error += xNorm.transferToDevice();
			error += xS.transferToDevice();
			error += w.transferToDevice();
			error += out_minus1.transferToDevice();
			error += xnormS.transferToDevice();
			
			if(error != MATRIX_SUCCESS){
				x.transferToHost();
				y.transferToHost();
				xNorm.transferToHost();
				xS.transferToHost();
				w.transferToHost();
				out_minus1.transferToHost();
				xnormS.transferToHost();
				
				cerr << "Device memory insufficient for data, switching to host computation" << endl;
				gpu = false;

			} else {
				error = K.transferToDevice();
				
				if (error != MATRIX_SUCCESS) {
					cerr << "Device memory insufficient for full kernel, leaving on host" << endl;
					K.transferToHost();
				}
			}
		}
				
		//Allocating memory and transfering matrices to device
		x.colSqSum(xNorm);
		//boolean array to track what's been selected already
		//we will fix candidates to work better later. this is a terrible hack.
		bool *alreadySelected = new bool[p.n];
		for(int i = 0; i < p.n; ++i){
			alreadySelected[i] = false;
		}
		
		//initialize with 100 random support vectors
		for (vector<int>::iterator r = R.begin(); r!=R.end(); ++r){
			
			//initialize/reset candidate buffer
			vector<int> candidates;
			int cc = 1;
			//add basis vectors until |S|==r
			int d0 = S.size();
			LaspMatrix<T> cand_K2;
			LaspMatrix<T> cand_K2_norm;
			if (gpu){
				cand_K2.transferToDevice();
				cand_K2_norm.transferToDevice();
			}
			for(int d = d0; d < *r; ++d){
				//for small data sets, use everything as basis vector and train
				if (smallDataSet){
					
					for (int i = 0; i < p.n; ++i){
						S.push_back(i);
						alreadySelected[i] = true;
					}
					xS = x.copy();
					xnormS = xNorm.copy();
					break;
					
				}
				
				//Begin by selecting a hundred random basis vectors
				if (*r == p.options.start_size && p.options.start_size != 0){
					int set = p.options.start_size;
				 	xS.resize(set, p.features);
					xnormS.resize(set, 1);
					for(int i = 1; i < set + 1; ++i){
						int randIndex = (i*10) % p.n;
						S.push_back(randIndex);
						alreadySelected[randIndex] = true;
					}
					
					x.gather(xS, S);
					xNorm.gather(xnormS, S);
					break;
				}

				//If we have just started or exhausted the previous batch of candidates to consider, get a new batch
				if (cc*10 > candidates.size()){
					//selecting candidate points to select new basis vectors from
					int candstonextretrain = 0;
					for(vector<int>::iterator r = R.begin(); r != R.end(); ++r){
						if (*r > d){
							candstonextretrain = *r - d;
							break;
						}
					}
					int candbatchsize= p.options.nb_cand *( min(candstonextretrain, p.options.maxcandbatch));
					d0=S.size();
					candidates.clear();
					for (int i = 1; i <= candbatchsize; ++i){
						candidates.push_back(((i + p.options.nb_cand*d0 - 1) % p.n));
					}
					
					if (!p.options.randomize) {
						//rows of x determined by candidates
						LaspMatrix<T> Xc = LaspMatrix<T>(p.features,candidates.size(),0.0);
						if (gpu){
							Xc.transferToDevice();
						}
						x.gather(Xc,candidates);
						
						//norm of dXc
						LaspMatrix<T> xnormc = LaspMatrix<T>(1,candidates.size());
						if (gpu){
							xnormc.transferToDevice();
						}
						
						xNorm.gather(xnormc,candidates);
						
						cand_K2 = compute_kernel<T>(kernelOptions, xerv, xnormerv, 0, 0, Xc, xnormc, 0, 0, gpu);
						cand_K2.colWiseMult(yerv);
						cand_K2.colSqSum(cand_K2_norm);
						
					}
					
					cc = 1;
				}

				// Which subgroup of the current candidate batch are we working on?
				vector<int> ccI;
				for (int i = (cc-1)*10+1; i <= cc*10; ++i){
					ccI.push_back(i-1);
				}
				
				vector<int> candidatesCCI;
				for (int i = 0; i < ccI.size(); ++i){
					candidatesCCI.push_back(candidates[ccI[i]]);
				}
				
				int select = rand() % ccI.size();
				
				if (!p.options.randomize) {
					
					LaspMatrix<T> cand_K2G;
					LaspMatrix<T> cand_K2normG;
					if (gpu){
						cand_K2G.transferToDevice();
						cand_K2normG.transferToDevice();
					}
					cand_K2.gather(cand_K2G, ccI);
					cand_K2_norm.gather(cand_K2normG, ccI);
										
					//heuristically choose next support vector
					select = choose_next_point<T>(kernelOptions , candidatesCCI, S, xS, xnormS, x, xNorm, p.features, w, xInd, cand_K2G, ccI.size(), erv->size(), out_minus1, out_minus1Length, p.options.C, p.options.gamma, cand_K2normG, gpu);				
				}
				
				int nonSelectIndex = select;
				if(alreadySelected[select]) {
					for(int i = 0; i < p.n;  ++i) {
						if(!alreadySelected[i]) {
							nonSelectIndex = i;
							break;
						}
					}
				}
				
				alreadySelected[nonSelectIndex] = true;
				S.push_back(nonSelectIndex);
				
				xS.resize(d+1, p.features);
				xnormS.resize(d+1, 1);

				xS.setCol(S.size()-1,x,S.back());
				xnormS.setCol(S.size()-1,xNorm,S.back());

				// Move to next subgroup
				++cc;
			}

			if (xInd != 0 && gpu){
				w.transferToHost();
			}
			
			int d = S.size();
			d0= sizeHESS - 1;
						
			LaspMatrix<T> dK;
			if (d != d0){
				//gathers the support vectors that are new into their own vector
				vector<int> SG;
				for(int i= S.size()-(d-d0);i < S.size();++i){
					SG.push_back(i);
				}

				if (!p.options.smallKernel) {
					K_temp = compute_kernel<T>(kernelOptions, x,xNorm,0,0,xS,xnormS,SG.data(), SG.size(), gpu);
									
					K.resize(K.cols(), d+1);
					
					if (gpu){
						K_temp.transferToHost();
						y.transferToHost();
						K.transferToHost();
					}

					{
						K_temp.colWiseMult(y);
						LaspMatrix<T> K_new = K(0, d0+1, p.n, d+1);
						K_temp.transpose(K_new);
					}
				}
								
				//calculating Ksum
				if (gpu){
					Ksum.transferToHost();
				}
				
				LaspMatrix<T> Ksum_new = Ksum(0, d0+1, 1, d+1);
				
				if (p.options.smallKernel) {
					LaspMatrix<T> Kerv = compute_kernel<T>(kernelOptions, x,xNorm, erv->data(), erv->size(), xS,xnormS,SG.data(), SG.size(), gpu);
					
					LaspMatrix<T> yerv_temp;
					y.gather(yerv_temp, *erv);
					Kerv.colWiseMult(yerv_temp);
					
					LaspMatrix<T> kSum_temp;
					Kerv.colSum(kSum_temp);
					kSum_temp.transpose(Ksum_new);
					
				} else {
					LaspMatrix<T> Kerv = K(0, d0+1, p.n, d+1);
					Kerv.gatherSum(Ksum_new, *erv);
				}
				
				if (gpu){
					HESS.transferToHost();
					K.transferToHost();
					y.transferToHost();
				}
				
				LaspMatrix<double> HESS_in = HESS.template convert<double>(true);
				LaspMatrix<double> K_in = K.template convert<double>();
				LaspMatrix<double> y_in = y.template convert<double>();
				LaspMatrix<double> x_in = x.template convert<double>();
				LaspMatrix<double> xNorm_in = xNorm.template convert<double>();
				
				update_hess(p, HESS_in, K_in, erv, y_in, S, x_in, xNorm_in);
				HESS = HESS_in.convert<T>(true);
				sizeHESS = HESS.rows();
				ldHESS = HESS.mRows();
			}
			
			double obj2 = train_subset_host<T>( p, S, w, HESS, erv, K, Ksum, out, old_out, x_old, x_old_size, x, y, xNorm, xerv, yerv, xnormerv);
			xInd = sizeHESS;
						
			if (obj2<0 || obj2 != obj2){ //Newton steps didn't converge. Probably because the Hessian is not well conditioned. This could be a precision problem.
				cerr << "Converge  problem in Newton retraining." << endl;
			}
			
			double newTre = 0;
			
			for(int i = 0; i < p.n; ++i){
				if(out(i) < 0)
					newTre++;
			}
			p.tre.push_back(newTre/p.n);
			
			
			double stopvalue = FLT_MAX;
				//numeric_limits<double>::max();
			
			if (p.tre.size() > 1){
				stopvalue = - (p.tre[p.tre.size()-1] - p.tre[p.tre.size()-2])/(S.size() - d0);
			}
			
			//Update trace variables
			p.obj.push_back(obj2);
			double newTime = (double)difftime(time(0), baseTime);
			p.time.push_back(newTime);
			vector<double> newBetas;
			newBetas.resize(w.size());
			for(int i = 1; i < w.size(); ++i){
				newBetas[i - 1] = w(i);
			}
			p.betas.push_back(newBetas);
			p.bs.push_back(w(0));
			
			//Output status, if verbose is enabled
			if (p.options.verb > 1 ){
				if (p.options.verb < 3) {
					cout << endl;
				}
				
				cout << "Training Error = " <<  p.tre.back() << ", Time = " << p.time.back() << ", Stopping value = " << stopvalue << "\n\n" << endl;
			} else if (p.options.verb > 0) {
				cout << ". " << flush;
			}
			
			
			for( int i = 0; i< erv->size(); ++i){
				out(i) = out((*erv)[i]);
			}
			
			fill_n(out.data()+erv->size(), p.n-erv->size(),0);
			
			out_minus1.resize(1, erv->size());
			
			
			if (gpu){
				out_minus1.transferToHost();
			}

			LaspMatrix<T> out_erv = out(0,0,1,erv->size());
			out_erv.eWiseOp(out_minus1,-1, 1, 1);
			
			
			if (gpu){
				out_minus1.transferToDevice();
				w.transferToDevice();
			}

			//check stopping criterion
			if (stopvalue >= 0 && stopvalue < p.options.stoppingcriterion && S.size() > 10){
				++stopIters;
				
				if (stopIters >= p.options.stopIters) {
					break;
				}
			} else {
				stopIters = 0;
			}
		}
		
		delete alreadySelected;

		p.y = y.template getRawArrayCopy<double>();
		p.xS = xS.template getRawArrayCopy<double>();
		p.S=S;
		
		if(p.options.verb > 0){
			cout << endl << "Training Complete" << endl;
		}
		return CORRECT;
	}
	
	
	void tempOutputCheck(svm_node** x, double* y){
		ofstream out;
		out.open("outputTest.txt");
		for(int i = 0; i < 3; i++){
			out << "Node " << i / 3 << ": " << x[i / 3][i % 3].index << " " << x[i / 3][i % 3].value << ", Y: " << y[i/3] << endl;
		}
		out.close();
	}
	
	vector<int> mysetdiff(vector<int>& a, vector<int>& b){
		vector<int> d;
		int an = a.size();
		int bn = b.size();
		d.resize(an > bn ? an : bn); // TODO probably not efficient maybe it works
		int dn = 0;
		
		for (int i=0, j=0; i<an; i++) {
			while(bn !=0 && j<bn && a[i] > b[j]){ j++;}
			if (bn != 0 && (a[i] < b[j] || (j==bn && a[i] > b[j]))) d[dn++] = a[i];
		}
		
		d.resize(dn);
		bool allZweros(true);
		for (int i=0; i < d.size(); ++i){
			if (d[i]!=0){
				allZweros=false;
				break;
			}
		}
		
		if (allZweros){
			d.resize(0);
		}
		
		return d;
	}
	
	template int lasp_svm_host<float>(svm_problem& p);
	template int lasp_svm_host<double>(svm_problem& p);
}

void lasp::setup_svm_problem_shuffled(svm_problem& problem,
									  svm_sparse_data myData,
									  svm_sparse_data& holdoutData,
									  opt options,
									  int posClass,
									  int negClass)
{
	if(myData.allData.size() > 2) {
		cerr << "myData must only have 2 classes!" << endl;
		exit_with_help();
	}
	
	srand (time(0));
	
	problem.options = options;
	problem.features = myData.numFeatures;
	//a vector of support vectors and their associated classification
	//in sparse form.
	vector<pair<vector<svm_node>, double> > allData;
	
	int numDataPoints = 0;
	typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
	for(SparseIterator myIter = myData.allData.begin();
		myIter != myData.allData.end();
		++myIter) {
		numDataPoints += myIter->second.size();
		for(int dataPoint = 0; dataPoint < myIter->second.size(); ++dataPoint) {
			pair<vector<svm_node>, double> curVector;
			curVector.first = myIter->second[dataPoint];
			curVector.second = myIter->first;
			allData.push_back(curVector);
		}
	}
	problem.classifications.push_back(posClass);
	problem.classifications.push_back(negClass);
	
	
	//now, we need to shuffle allData
	if (problem.options.shuffle){
		random_shuffle(allData.begin(), allData.end());
	}
	//here, we pop off 30% of the data as "holdout data" to be used to
	//accomplish platt scaling later.
	vector<pair<vector<svm_node>, double> > holdout;
	for(int i = 0; i < .3 * allData.size(); ++i) {
		holdout.push_back(allData.back());
		allData.pop_back();
	}
	
	holdoutData.orderSeen.push_back(posClass); holdoutData.orderSeen.push_back(negClass);
	holdoutData.numFeatures = myData.numFeatures;
	holdoutData.numPoints = holdout.size();
	holdoutData.multiClass = false;
	//now lets fill up the holdoutData.
	for(int i = 0; i < holdout.size(); ++i) {
		pair<vector<svm_node>, double> curPair = holdout[i];
		holdoutData.allData[int(curPair.second)].push_back(curPair.first);
	}
	
	problem.n = numDataPoints - holdout.size();
	
	//now that we've shuffled everything and pulled out the holdout data,
	//lets put all the data into vectors.
	vector<double> fullDataVector;
	//classificationVector must contain -1 and 1 only, as surrogates for the
	//classes. The lesser of the two classes is represented by -1 and the greater
	//of the two classes is 1.
	vector<double> classificationVector;
	
	for(int x = 0; x < problem.n; ++x) {
		vector<double> fullDataPoint;
		sparse_vector_to_full(fullDataPoint, allData[x].first, myData.numFeatures);
		for(int i = 0; i < myData.numFeatures; ++i) fullDataVector.push_back(fullDataPoint[i]);
		if(allData[x].second == negClass)
			classificationVector.push_back(-1);
		else //it must be the positive class
			classificationVector.push_back(1);
	}
	double_vector_to_array(problem.xS, fullDataVector);
	double_vector_to_array(problem.y, classificationVector);
}


void lasp::setup_svm_problem_shuffled(svm_problem& problem,
									  svm_sparse_data myData,
									  opt options,
									  int posClass,
									  int negClass)
{
	if(myData.allData.size() > 2) {
		cout << "myData must only have 2 classes!" << endl;
		exit_with_help();
	}
	
	srand (time(0));
	
	problem.options = options;
	problem.features = myData.numFeatures;
	//a vector of support vectors and their associated classification
	//in sparse form.
	vector<pair<vector<svm_node>, double> > allData;
	
	int numDataPoints = 0;
	typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
	for(SparseIterator myIter = myData.allData.begin();
		myIter != myData.allData.end();
		++myIter) {
		numDataPoints += myIter->second.size();
		for(int dataPoint = 0; dataPoint < myIter->second.size(); ++dataPoint) {
			//			pair<vector<svm_node>, double> curVector(;
			//			curVector.first = ;
			//			curVector.second = ;
			allData.push_back(make_pair(myIter->second[dataPoint], myIter->first));
		}
	}
	problem.classifications.push_back(posClass);
	problem.classifications.push_back(negClass);
	
	problem.n = numDataPoints;
	
	//now, we need to shuffle allData
	if(problem.options.shuffle){
		random_shuffle(allData.begin(), allData.end());
	}
	
	//now that we've shuffled everything, lets put all the data into vectors.
	vector<double> fullDataVector;
	//classificationVector must contain -1 and 1 only, as surrogates for the
	//classes. The lesser of the two classes is represented by -1 and the greater
	//of the two classes is 1.
	vector<double> classificationVector;
	
	for(int x = 0; x < problem.n; ++x) {
		vector<double> fullDataPoint;
		sparse_vector_to_full(fullDataPoint, allData[x].first, myData.numFeatures);
		for(int i = 0; i < myData.numFeatures; ++i) fullDataVector.push_back(fullDataPoint[i]);
		if(allData[x].second == negClass)
			classificationVector.push_back(-1);
		else //it must be the positive class
			classificationVector.push_back(1);
	}
	double_vector_to_array(problem.xS, fullDataVector);
	double_vector_to_array(problem.y, classificationVector);
	
}




