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

#ifndef LASP_MATRIX_H
#define LASP_MATRIX_H

#include "abstract_matrix.h"
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <ostream>
#include <cmath>
#include <iomanip>
#include <string.h>

#include "blas_wrappers.h"

#ifdef WUCUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

	namespace lasp
	{	

	template<class T>
    class LaspMatrix {
		
		//Use pointers so data can be transferred effficiently
		// rc: reference count, key: memory identification key
		// Offset/end used for submatricies 
		int *cols_, *rows_, *mCols_, *mRows_, *rc_, *colOffset_, *rowOffset_, *colEnd_, *rowEnd_, *subrc_;
		void **key_;
		bool *device_, *registered_;
		DeviceContext *context_;
		
		//The data itself, the format of which vary based on what particular
		//subclass we're in.
		T **data_, **dData_;

		//Private accessors given by reference
		inline int& _rc() const{ return *rc_; };
		inline int& _subrc() const{ return *subrc_;};
		inline void*& _key() const{ return *key_; }; 
		inline int& _cols() const{ return *cols_; };
		inline int& _mCols() const{ return *mCols_; };
		inline int& _rows() const{ return *rows_; };
		inline int& _mRows() const{ return *mRows_; };
		inline int& _colOffset() const{ return *colOffset_; }
		inline int& _rowOffset() const{ return *rowOffset_; }
		inline int& _colEnd() const{ return *colEnd_; }
		inline int& _rowEnd() const{ return *rowEnd_; }
		inline bool& _device() const{ return *device_; }
		inline bool& _registered() const{ return *registered_; }
		inline T*& _data() const{ return *data_; };
		inline T*& _dData() const{ return *dData_; }
		
		//Internal methods for reference counting management
		void cleanup();
		void freeData();

		//Internal device methods
		int deviceCopy(LaspMatrix<T> &other);
		LaspMatrix<T> deviceCopy();

		int deviceResize(int newCols, int newRows, bool copy = true, bool fill = false, T val = 0.0);

		int deviceSetRow(int row, LaspMatrix<T>& other);
		int deviceSetCol(int col, LaspMatrix<T>& other);

		int deviceSetRow(int row, LaspMatrix<T>& other, int otherRow);
		int deviceSetCol(int col, LaspMatrix<T>& other, int otherCol);
		
	public:
		//Public accessors for member variables
		inline int rc() const{ return *rc_; };
		inline int subrc() const{ return *subrc_;};
		inline void* key() const{ return *key_; }; 
		inline int cols() const{ return (*colEnd_ != 0) ? max(min(*cols_, *colEnd_) - *colOffset_, 0) : max(*cols_ - *colOffset_, 0); };
		inline int mCols() const{ return *mCols_ - *colOffset_; };
		inline int rows() const{ return (*rowEnd_ != 0) ? max(min(*rows_, *rowEnd_) - *rowOffset_, 0) : max(*rows_ - *rowOffset_, 0); };
		inline int mRows() const{ return *mRows_; };
		inline int colOffset() const{ return *colOffset_; }
		inline int rowOffset() const{ return *rowOffset_; }
		inline int colEnd() const{ return *colEnd_; }
		inline int rowEnd() const{ return *rowEnd_; }
		inline int size() const{ return rows() * cols(); };
		inline int mSize() const{ return (*mRows_) * (*mCols_) - (*colOffset_ * *mRows_ + *rowOffset_); };
		inline int elements() const{ return size(); };
		inline int mElements() const{ return size(); };
		inline T* data() const{ return *data_ + *mRows_ * *colOffset_ + *rowOffset_; };
		inline T* dData() const{ return *dData_ + *mRows_ * *colOffset_ + *rowOffset_; }
		inline bool device() const{ return *device_; }
		inline bool registered() const{ return *registered_; }
		inline bool isSubMatrix() const { return rowOffset() != 0 || colOffset() != 0 || rowEnd() != 0 || colEnd() != 0; };
		inline DeviceContext& context() const { return *context_; }

		
		//CONSTRUCTORS: Here, constructors only allocate
		//memory, rather than setting it.
		
		//Default Constructor
		LaspMatrix();
		
		//Standard Constructor, d is pointer to pre-allocated data
		LaspMatrix(int col, int row, T* d = 0, int mCol = 0, int mRow = 0);
		
		//Fill Constructor, val is the initial value for all elements
		LaspMatrix(int col, int row, T val, int mCol = 0, int mRow = 0, bool fill=true, bool fill_mem=false);
		
		//Copy Constructor: Same as assignment operator
		LaspMatrix(const LaspMatrix<T> &other);
		
		//Copy data into new memory
		int copy(LaspMatrix<T> &other, bool copyMem=false);
		LaspMatrix<T> copy(bool copyMem=false);
		
		
		//Resize matrix
		int resize(int newCols, int newRows, bool copy = true, bool fill = false, T val = 0.0);
		
		//Release data (NOT SAFE, USE WITH CAUTION!!!)
		int release();
		
		//DESTRUCTOR
		~LaspMatrix();

		//Transfer data to/from device
		int transfer();
		//Safer ways to call transfer
		int transferToDevice();
		int transferToHost();
		int registerHost();
		
		//Assignment Operator: Assigns to same data in memory, managed with reference counting
		LaspMatrix<T>& operator=(const LaspMatrix<T>& other);
		LaspMatrix<T>& operator=(const T val);
		
		//Element access operator
		T& operator()(int index, bool add=false);
		T& operator()(int col, int row, bool add=false);
		LaspMatrix<T> operator()(int startCol, int startRow, int endCol, int endRow);
		
		//Raw memory access
		T& operator[](int matrixPosition);
		
		//Copy 
		int setRow(int row, LaspMatrix<T>& other);
		int setCol(int col, LaspMatrix<T>& other);

		int setRow(int row, LaspMatrix<T>& other, int otherRow);
		int setCol(int col, LaspMatrix<T>& other, int otherCol);

		//Multiplication with BLAS-like syntax
		int multiply(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix, bool transposeMe = false, bool transposeOther = false, T a = 1.0, T b = 0.0, int numRowsToSkip = 0);
		int multiply(LaspMatrix<T>& otherMatrix, bool transposeMe = false, bool transposeOther = false, T a = 1.0, T b = 0.0, int numRowsToSkip = 0);
		int multiply(T scalar, LaspMatrix<T>& outputMatrix);
		int multiply(T scalar);
		
		int transpose(LaspMatrix<T>& outputMatrix);
		int transpose();

		LaspMatrix<T> diag(bool column = true);
		int diagAdd(T scalar);
		
		int add(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix);
		int add(LaspMatrix<T>& otherMatrix);
		int add(T scalar, LaspMatrix<T>& outputMatrix);
		int add(T scalar);
		
		int subtract(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix);
		int subtract(LaspMatrix<T>& otherMatrix);
		int subtract(T scalar, LaspMatrix<T>& outputMatrix);
		int subtract(T scalar);
		
		int negate(LaspMatrix<T>& output);
		int negate();
		
		int colWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output);
		int colWiseMult(LaspMatrix<T>& vec);
		
		int rowWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output);
		int rowWiseMult(LaspMatrix<T>& vec);
		
		int pow(T exp, LaspMatrix<T>& output);
		int pow(T exp);
		
		int exp(LaspMatrix<T>& output, T gamma = -1);
		int exp(T gamma = -1);

		int tanh(LaspMatrix<T>& output);
		int tanh();
		
		int solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, LaspMatrix<T>& LU, LaspMatrix<int>& ipiv);
		int solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output);
		int solve(LaspMatrix<T>& otherMatrix);
		
		int colSqSum(LaspMatrix<T>& output, T scalar = 1);
		LaspMatrix<T> colSqSum(T scalar = 1);
		
		int colSum(LaspMatrix<T>& output, T scalar = 1);
		LaspMatrix<T> colSum(T scalar = 1);

		int eWiseOp(LaspMatrix<T>& output, T add, T mult, T pow1);
		LaspMatrix<T> eWiseOp(T add, T mult, T pow1);

		int eWiseDivM( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1 = 1, T pow2 = 1);
		LaspMatrix<T> eWiseDivM( LaspMatrix<T>& otherMatrix, T pow1 = 1, T pow2 = 1);

		int eWiseMultM( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1 = 1, T pow2 = 1);
		LaspMatrix<T> eWiseMultM( LaspMatrix<T>& otherMatrix, T pow1 = 1, T pow2 = 1);

		int eWiseScale( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, int d0, int d);
		LaspMatrix<T> eWiseScale(LaspMatrix<T>& otherMatrix, int d0, int d);

		//Just swaps rows and columns
		int swap();

		template<class ITER>
		int gather(LaspMatrix<T>& output, ITER begin, ITER end);
		int gather(LaspMatrix<T>& output, LaspMatrix<int> map);
		int gather(LaspMatrix<T>& output, vector<int>& map);
		
		template<class ITER>
		LaspMatrix<T> gather(ITER begin, ITER end);
		LaspMatrix<T> gather(LaspMatrix<int> map);
		LaspMatrix<T> gather(vector<int>& map);
		
		template<class ITER>
		int gatherSum(LaspMatrix<T>& output, ITER begin, ITER end);
		int gatherSum(LaspMatrix<T>& output, LaspMatrix<int> map);
		int gatherSum(LaspMatrix<T>& output, vector<int>& map);

		LaspMatrix<T> getSubMatrix(int startCol, int startRow, int endCol, int endRow);
		LaspMatrix<T> getSubMatrix(int endCol, int endRow);
		
		int addMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out);
		int subMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out);
		
		int ger(LaspMatrix<T>& output, LaspMatrix<T> X, LaspMatrix<T> Y, T alpha = 1);
		int ger(LaspMatrix<T> X, LaspMatrix<T> Y, T alpha = 1);
		
		int getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& Xnorm1, LaspMatrix<T>& X2, LaspMatrix<T>& Xnorm2, bool mult = false, bool transMult = false, bool useGPU = true);

		template<class N>
		LaspMatrix<N> convert(bool mem=false);
		
		void printMatrix(string name = "", int c = 0, int r = 0);
		void printInfo(string name = "");
		void checksum(string name = "");

		template<class N>
		N* getRawArrayCopy();

		bool operator==(LaspMatrix<T>& other);

	};
	
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(): rc_(new int), rows_(new int), cols_(new int), mCols_(new int), mRows_(new int), colOffset_(new int), rowOffset_(new int), colEnd_(new int), rowEnd_(new int), subrc_(new int), data_(new T*), dData_(new T*), device_(new bool), registered_(new bool), context_(DeviceContext::instance()), key_(new void*){
		_rc() = 1;
		_subrc() = 1;
		_rows() = 0;
		_cols() = 0;
		_mRows() = 0;
		_mCols() = 0;
		_data() = 0;
		_dData() = 0;
		_rowOffset() = 0;
		_colOffset() = 0;
		_rowEnd() = 0;
		_colEnd() = 0;
		_device() = false;
		_registered() = false;
		_key() = context().getNextKey();
	}
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(int col, int row, T* d, int mCol, int mRow): rc_(new int), rows_(new int), cols_(new int), mCols_(new int), mRows_(new int), colOffset_(new int), rowOffset_(new int), colEnd_(new int), rowEnd_(new int), subrc_(new int), data_(new T*), dData_(new T*), device_(new bool), registered_(new bool), context_(DeviceContext::instance()), key_(new void*){
		_rc() = 1;
		_subrc() = 1;
		_rows() = row;
		_cols() = col;
		_mRows() = mRow == 0 ? row : mRow;
		_mCols() = mCol == 0 ? col : mCol;
		_data() = d == 0 ? new T[_mRows() * _mCols()] : d;
		_rowOffset() = 0;
		_colOffset() = 0;
		_rowEnd() = 0;
		_colEnd() = 0;
		_dData() = 0;
		_device() = false;
		_registered() = false;
		_key() = context().getNextKey();
	}
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(int col, int row, T val, int mCol, int mRow, bool fill, bool fill_mem): rc_(new int), rows_(new int), cols_(new int), mCols_(new int), mRows_(new int), colOffset_(new int), rowOffset_(new int), colEnd_(new int), rowEnd_(new int), subrc_(new int), data_(new T*), dData_(new T*), device_(new bool), registered_(new bool), context_(DeviceContext::instance()), key_(new void*){
		_rc() = 1;
		_subrc() = 1;
		_rows() = row;
		_cols() = col;
		_mRows() = mRow == 0 ? row : mRow;
		_mCols() = mCol == 0 ? col : mCol;
		_rowOffset() = 0;
		_colOffset() = 0;
		_rowEnd() = 0;
		_colEnd() = 0;
		_data() = new T[_mRows() * _mCols()];
		_dData() = 0;
		_key() = context().getNextKey();
		_device() = false;
		_registered() = false;

		if(fill && !fill_mem){
			int rowsTemp = rows();
			int colsTemp = cols();
			int mRowsTemp = mRows();
			T* dataTemp = data();

			for(int i = 0; i < colsTemp; ++i){
				for(int j = 0; j < rowsTemp; ++j){
					dataTemp[i * mRowsTemp + j] = val;
				}
			}

		} else if(fill_mem){ 
			fill_n(data(), mSize(), val);
		}
	}
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(const LaspMatrix<T>& other): rc_(other.rc_), rows_(other.rows_), cols_(other.cols_), mCols_(other.mCols_), mRows_(other.mRows_), colOffset_(other.colOffset_), rowOffset_(other.rowOffset_), colEnd_(other.colEnd_), rowEnd_(other.rowEnd_), subrc_(other.subrc_),  data_(other.data_), dData_(other.dData_), context_(other.context_), device_(other.device_), registered_(other.registered_), key_(other.key_){
		_rc()++;
		_subrc()++;
	}
	
	template<class T>
	int LaspMatrix<T>::copy(LaspMatrix<T>& other, bool copyMem){
		int resizeResult = resize(other.cols(), other.rows());

		if(resizeResult != MATRIX_SUCCESS){
			return resizeResult;
		}

		if(device() || other.device()){
			return deviceCopy(other);
		}

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		T* other_dataTemp = other.data();
		int other_mrowsTemp = other.mRows();

		#ifdef _OPENMP
		int ompCount = colsTemp * rowsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				dataTemp[mrowsTemp * j + i] = other_dataTemp[other_mrowsTemp * j + i];
			}
		}

		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::copy(bool copyMem){
		LaspMatrix<T> result;
		result.copy(*this, copyMem);
		return result;
	}
	
	template<class T>
	int LaspMatrix<T>::release(){
		#ifndef NDEBUG
		if(rc() > 1){
			cerr << "Warning: Releasing data with multiple references" << endl;
		}
		#endif
		_rc() = -1;
		_subrc() = -1;
		cleanup();
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	void LaspMatrix<T>::cleanup(){
		if (--_rc() == 0 && key_ != 0) {
			freeData();
			delete rc_;
			delete data_;
			delete rows_;
			delete cols_;
			delete mRows_;
			delete mCols_;
			delete key_;
			delete dData_;
			delete device_;
			delete registered_;
			key_ = 0;
		}

		if(--_subrc() == 0 && key_ != 0){
			delete subrc_;
			delete colOffset_;
			delete rowOffset_;
			delete colEnd_;
			delete rowEnd_;
		}
	}
	
	template<class T>
	LaspMatrix<T>::~LaspMatrix<T>(){
		if(key_ != 0){
			cleanup();
		}
	}
	
	template<class T>
	LaspMatrix<T>& LaspMatrix<T>::operator=(const LaspMatrix<T>& other){
		cleanup();
		
		rc_ = other.rc_;
		subrc_ = other.subrc_;
		_rc()++;
		_subrc()++;
		
		cols_ = other.cols_;
		rows_ = other.rows_;
		mCols_ = other.mCols_;
		mRows_ = other.mRows_;
		data_ = other.data_;
		key_ = other.key_;
		dData_ = other.dData_;
		device_ = other.device_;
		registered_ = other.registered_;
		context_ = other.context_;
		rowOffset_ = other.rowOffset_;
		colOffset_ = other.colOffset_;
		rowEnd_ = other.rowEnd_;
		colEnd_ = other.colEnd_;

		
		return *this;
	}
	
	template<class T>
	LaspMatrix<T>& LaspMatrix<T>::operator=(const T val){
		return eWiseOp(*this, val, 0, 0);
	}
	
	template<class T>
	T& LaspMatrix<T>::operator()(int index, bool add) {
		if(device()){
			transferToHost();
		}

		#ifndef NDEBUG
		if(index >= rows() * cols()){
			cerr << "Error: Index (" << index << ") out of bounds!" << endl;
			throw INVALID_DIMENSIONS;
		}
		#endif

		int row = index % rows();
		int col = index / rows();
		
		return data()[col * mRows() + row];
	}
	
	template<class T>
	T& LaspMatrix<T>::operator()(int col, int row, bool add) {
		if(device()){
			transferToHost();
		}

		#ifndef NDEBUG
		if(col >= cols() || row >= rows()){
			cerr << "Error: Index (" << col << ", " << row << ") out of bounds!" << endl;
			throw INVALID_DIMENSIONS;
		}
		#endif
		return data()[col * mRows() + row];
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::operator()(int startCol, int startRow, int endCol, int endRow){
		return getSubMatrix(startCol, startRow, endCol, endRow);
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::getSubMatrix(int startCol, int startRow, int endCol, int endRow){
		#ifndef NDEBUG
		if(startCol >= cols() || startRow >= rows() || startRow > endRow || startCol > endCol){
			cerr << "Error: Indicies (" << startCol << ", " << startRow << ") :: (" << endCol << ", " << endRow << ") out of bounds!" << endl;
		}
		#endif

		LaspMatrix<T> output(*this);

		output.colOffset_ = new int;
		output.rowOffset_ = new int;
		output.colEnd_ = new int;
		output.rowEnd_ = new int;
		output.subrc_ = new int;

		--_subrc();
		output._subrc() = 1;
		output._colOffset() = startCol + colOffset();
		output._rowOffset() = startRow + rowOffset();
		output._colEnd() = endCol == 0 ? endCol : colOffset() + endCol;
		output._rowEnd() = endRow == 0 ? endRow : rowOffset() + endRow;

		return output;
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::getSubMatrix(int endCol, int endRow){
		return getSubMatrix(0, 0, endCol, endRow);
	}
	
	template<class T>
	T& LaspMatrix<T>::operator[](int matrixPosition){
		#ifndef NDEBUG
		//cerr << "Warning: LaspMatrix::operator[] will result in unchecked memory access, consider using LaspMatrix::operator() instead" << endl;
		#endif
		return data()[matrixPosition];
	}
	
	template<class T>
	int LaspMatrix<T>::setRow(int row, LaspMatrix<T>& other){
		if(other.size() != cols()){
			cerr << "Error: Dimension mismatch in setRow" << endl;
			return INVALID_DIMENSIONS;
		}

		if(device() || other.device()){
			return deviceSetRow(row, other);
		}
		
		for(int i = 0; i < other.size(); ++i){
			operator()(i, row) = other(i);
		}
		
		return MATRIX_SUCCESS;
	}

	template<class T>
	int LaspMatrix<T>::setRow(int row, LaspMatrix<T>& other, int otherRow){
		if(other.cols() != cols()){
			cerr << "Error: Dimension mismatch in setRow" << endl;
			return INVALID_DIMENSIONS;
		}

		if(device() || other.device()){
			return deviceSetRow(row, other, otherRow);
		}
		
		for(int i = 0; i < other.cols(); ++i){
			operator()(i, row) = other(i, otherRow);
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::setCol(int col, LaspMatrix<T>& other){
		if(other.size() != rows()){
			cerr << "Error: Dimension mismatch in setCol" << endl;
			return INVALID_DIMENSIONS;
		}

		if(device() || other.device()){
			return deviceSetCol(col, other);
		}
		
		for(int i = 0; i < other.size(); ++i){
			operator()(col, i) = other(i);
		}
		
		return MATRIX_SUCCESS;
	}

	template<class T>
	int LaspMatrix<T>::setCol(int col, LaspMatrix<T>& other, int otherCol){
		if(other.rows() != rows()){
			cerr << "Error: Dimension mismatch in setCol" << endl;
			return INVALID_DIMENSIONS;
		}

		if(device() || other.device()){
			return deviceSetCol(col, other, otherCol);
		}
		
		for(int i = 0; i < other.rows(); ++i){
			operator()(col, i) = other(otherCol, i);
		}
		
		return MATRIX_SUCCESS;
	}
	
	//TODO: Check excess fill values
	template<class T>
	int LaspMatrix<T>::resize(int newCols, int newRows, bool copy, bool fill, T val){
		if(cols() == newCols && rows() == newRows){
			return MATRIX_SUCCESS;
		}

		if(isSubMatrix()){
			#ifndef NDEBUG
			cerr << "Cannot resize a sub-matrix" << endl;
			#endif
			return CANNOT_COMPLETE_OPERATION;
		}


		if(device()){
			return deviceResize(newCols, newRows, copy, fill, val);
		}

		if ((newCols == 1 && _rows() == 1 && _cols() != 1) || (newRows == 1 && _cols() == 1 && _rows() != 1)){
			std::swap(_rows(), _cols());
			std::swap(_mRows(), _mCols());
		}

		if(_mRows() < newRows || _mCols() < newCols){
			if ((max(newCols, _cols()) * max(newRows, _rows()) * 8) / 1000000000.0 > 1.0){
#ifndef NDEBUG
				cerr << "Allocating size: " << (max(newCols, _cols()) * max(newRows, _rows()) * 8) / 1000000000.0 << " GB" << endl;
#endif
			}
			T* newptr = new T[max(newCols, cols()) * max(newRows, rows())];
			
			if(copy && rows() > 0 && cols() > 0){
				#ifdef _OPENMP
				int ompCount = _rows() * _cols();
				int ompLimit = context().getOmpLimit();
				#endif
				
				#pragma omp parallel for if(ompCount > ompLimit)
				for(int i = 0; i < _rows(); ++i){
					for(int j = 0; j < _cols(); ++j){
						//Fix this to use memcpy
						newptr[j * newRows + i] = _data()[j * _rows() + i];
					}
				}
			}
			
			if(fill){
				//Done stupidly on purpose, fix at some point
				#ifdef _OPENMP
				int ompCount = newRows * newCols;
				int ompLimit = context().getOmpLimit();
				#endif
				
				#pragma omp parallel for if(ompCount > ompLimit)
				for(int i = 0; i < newRows; ++i){
					for(int j = 0; j < newCols; ++j){
						if(i >= _rows() || j >= _cols()){
							newptr[j * newRows + i] = val;
						}
					}
				}
				
			}
			
			delete [] data();
			_data() = newptr;
			_mCols() = newCols;
			_mRows() = newRows;
		}
		
		_cols() = newCols;
		_rows() = newRows;

		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::multiply( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix, bool transposeMe, bool transposeOther, T a, T b, int numRowsToSkip){
		return METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::multiply(LaspMatrix<T>& otherMatrix, bool transposeMe, bool transposeOther, T a, T b, int numRowsToSkip){
		return this->multiply(otherMatrix, *this, transposeMe, transposeOther, a, b);
	}
	
	template<class T>
	int LaspMatrix<T>::multiply(T scalar, LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, 0, scalar, 1);
		}

		//Resize output
		bool copy = outputMatrix.key() == key();
		outputMatrix.resize(cols(), rows(), copy);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){	
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] * scalar;
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::multiply(T scalar) {
		return multiply(scalar, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::transpose(LaspMatrix<T> &outputMatrix){
		if (device()) {
			int error = MATRIX_SUCCESS;
			error += outputMatrix.transferToDevice();
			error += outputMatrix.resize(rows(), cols());

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &outputMatrix);
				return device_transpose(params, dData(), outputMatrix.dData(), cols(), rows(), mRows(), outputMatrix.mRows());
			}
		}

		transferToHost();
		outputMatrix.transferToHost();
		outputMatrix.resize(rows(), cols());

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * i + j] = dataTemp[mrowsTemp * j + i];
			}
		}

		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::transpose(){
		if (isSubMatrix()) {
		#ifndef NDEBUG
			cerr << "Cannot transpose subMatrix!" << endl;
		#endif
			return CANNOT_COMPLETE_OPERATION;
		}

		if(mRows() == 1 || mCols() == 1){
			return this->swap();
		}
		
		LaspMatrix<T> output;
		transpose(output);
		
		std::swap(_data(), output._data());
		std::swap(_dData(), output._dData());
		std::swap(_cols(), output._cols());
		std::swap(_rows(), output._rows());
		std::swap(_mCols(), output._mCols());
		std::swap(_mRows(), output._mRows());
		
		return MATRIX_SUCCESS;
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::diag(bool column){
		_mRows()++;
		LaspMatrix<T> subMat = this->operator()(0, 0, min(cols(), rows()), 1);
		LaspMatrix<T> output;
		subMat.transpose(output);
		_mRows()--;

		if (!column){
			output.transpose();
		}

		return output;
	}

	template<class T>
	int LaspMatrix<T>::diagAdd(T scalar){
		_mRows()++;
		LaspMatrix<T> subMat = this->operator()(0, 0, min(cols(), rows()), 1);
		int error = subMat.add(scalar);
		_mRows()--;

		return error;
	}
	
	template<class T>
	int LaspMatrix<T>::add(LaspMatrix<T> &otherMatrix, LaspMatrix<T> &outputMatrix) {
		if (cols() != otherMatrix.cols() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in add" << endl;
			return INVALID_DIMENSIONS;
		}

	  //Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		outputMatrix.resize(cols(), rows(), copy);


		if (device()){
			int error = MATRIX_SUCCESS;

			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix, &outputMatrix);
				return device_addMatrix(params,dData(),otherMatrix.dData(),outputMatrix.dData(), rows(), cols(), mRows(), otherMatrix.mRows(), outputMatrix.mRows());
			}
		}

		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();

		T* other_dataTemp = otherMatrix.data();
		int other_mrowsTemp = otherMatrix.mRows();

		#ifdef _OPENMP
		int ompCount = colsTemp * rowsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] + other_dataTemp[other_mrowsTemp * j + i];
			}
		}

		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::add(LaspMatrix<T> &otherMatrix) {
		return add(otherMatrix, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::add(T scalar, LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, scalar, 1, 1);
		}

		outputMatrix.transferToHost();

		//Resize output
		bool copy = outputMatrix.key() == key();
		outputMatrix.resize(cols(), rows(), copy);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();
		
		#ifdef _OPENMP
		int ompCount = colsTemp * rowsTemp;
		int ompLimit = context().getOmpLimit();
		#endif

		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] + scalar;
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::add(T scalar) {
		return add(scalar, *this);
	}

	template<class T>
	int LaspMatrix<T>::eWiseMultM(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1, T pow2){
		if(otherMatrix.rows() != rows() || otherMatrix.cols() != cols()){
			#ifndef NDEBUG
			cerr << "Error: Matrix dimensions must be equal for eWiseDivM" << endl;
			#endif
			return INVALID_DIMENSIONS;
		}
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows(), false);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix, &output);
				device_eWiseMult(params, dData(), otherMatrix.dData(), output.dData(), size(), pow1, pow2, rows(), mRows(), otherMatrix.mRows(), output.mRows());

				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		otherMatrix.transferToHost();
		output.transferToHost();

		output.resize(cols(), rows(), false);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();

		T* other_dataTemp = otherMatrix.data();
		int other_mrowsTemp = otherMatrix.mRows();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for( int j = 0; j < colsTemp; ++j){
			for( int i = 0; i < rowsTemp; ++i){
				output_dataTemp[j * output_mrowsTemp + i] = std::pow(dataTemp[j*mrowsTemp + i], pow1) * std::pow(other_dataTemp[j * other_mrowsTemp + i], pow2);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseMultM(LaspMatrix<T>& otherMatrix, T pow1, T pow2){
		LaspMatrix<T> output;
		int err = eWiseMultM(otherMatrix, output, pow1, pow2);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(LaspMatrix<T> &otherMatrix, LaspMatrix<T> &outputMatrix) {
		if(device()){
			otherMatrix.negate(outputMatrix);
			return add(outputMatrix, outputMatrix);
		}

		otherMatrix.transferToHost();
		outputMatrix.transferToHost();

		if (cols() != otherMatrix.cols() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in subtract" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		outputMatrix.resize(cols(), rows(), copy);
		
		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();

		T* other_dataTemp = otherMatrix.data();
		int other_mrowsTemp = otherMatrix.mRows();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] - other_dataTemp[other_mrowsTemp * j + i];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(LaspMatrix<T> &otherMatrix) {
		return subtract(otherMatrix, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(T scalar, LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, -scalar, 1, 1);
		}

		return add(-scalar, outputMatrix);
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(T scalar) {
		return subtract(scalar, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::negate(LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, 0, -1, 1);
		}

		return multiply(-1.0, outputMatrix);
	}
	
	template<class T>
	int LaspMatrix<T>::negate() {
		return negate(*this);
	}
	
	template<class T>
	int LaspMatrix<T>::colWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output){
		// mat is x by y, vec has length x
		if (!((vec.rows() == rows() && vec.cols() == 1) || (vec.cols() == rows() && vec.rows() == 1))){
			cerr << "Error: you must pass a vector with the same number of rows as the input matrix" << endl;
			return INVALID_DIMENSIONS;
		}

		if(device()){
			int error = MATRIX_SUCCESS;
			error += vec.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows());

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &vec, &output);
				device_colWiseMult(params, dData(), output.dData(), vec.dData(), rows(), cols(), mRows(), output.mRows(), (vec.rows() == 1 ? vec.mRows() : 1));

				return MATRIX_SUCCESS;
			}	
		}

		transferToHost();
		vec.transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);

		int colsTemp = cols();
		int rowsTemp = rows();
		int mRowsTemp = mRows();

		int output_mrowsTemp = output.mRows();
		int vec_stride = (vec.rows() == rows() && vec.cols() == 1) ? 1 : vec.mRows();

		T* output_dataTemp = output.data();
		T* dataTemp = data();
		T* vecData = vec.data();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for (int i = 0; i < colsTemp; ++i){
			for (int j =0; j < rowsTemp; ++j){
				output_dataTemp[output_mrowsTemp * i + j] = dataTemp[mRowsTemp * i + j] * vecData[vec_stride * j];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::colWiseMult(LaspMatrix<T>& vec){
		return colWiseMult(vec, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::rowWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output){
		// mat is x by y, vec has length x
		if (!((vec.rows() == cols() && vec.cols() == 1) || (vec.cols() == cols() && vec.rows() == 1))){
			cerr << "Error: you must pass a vector with the same number of rows as the input matrix" << endl;
			return INVALID_DIMENSIONS;
		}

		if(device()){
			int error = MATRIX_SUCCESS;
			error += vec.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows());

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &vec, &output);
				device_rowWiseMult(params, dData(), output.dData(), vec.dData(), rows(), cols(), mRows(), output.mRows(), (vec.rows() == 1 ? vec.mRows() : 1));

				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		vec.transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);

		int colsTemp = cols();
		int rowsTemp = rows();
		int mRowsTemp = mRows();

		int output_mrowsTemp = output.mRows();
		int vec_stride = (vec.rows() == rows() && vec.cols() == 1) ? 1 : vec.mRows();

		T* output_dataTemp = output.data();
		T* dataTemp = data();
		T* vecData = vec.data();
		
		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for (int j =0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				output[output_mrowsTemp * j + i]= dataTemp[mRowsTemp * j + i] * vecData[vec_stride * j];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::rowWiseMult(LaspMatrix<T>& vec){
		return rowWiseMult(vec, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::pow(T exp, LaspMatrix<T>& output){
		if(device()){
			return eWiseOp(output, 0, 1, exp);
		}

		output.transferToHost();

		//Resize output
		output.resize(cols(), rows(), false);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = std::pow(dataTemp[mrowsTemp * j + i], exp);
			}
		}
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::pow(T exp){
		return pow(exp, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::exp(LaspMatrix<T>& output, T gamma){
		//Resize output
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_exp(params,dData(), output.dData(), cols(), rows(), mRows(), output.mRows(), gamma);

				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		output.transferToHost();

		output.resize(cols(), rows(), false);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();

		int sizeTemp = rowsTemp * colsTemp;

		#ifdef _OPENMP
		int ompCount = sizeTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int k = 0; k < sizeTemp; ++k){
			int i = k % rowsTemp;
			int j = k / rowsTemp;
			output_dataTemp[output_mrowsTemp * j + i] = std::exp(dataTemp[mrowsTemp * j + i] * -gamma);
		}

		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::exp(T gamma){
		return exp(*this, gamma);
	}


	template<class T>
	int LaspMatrix<T>::tanh(LaspMatrix<T>& output){
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_tanh(params,dData(), output.dData(), cols(), rows(), mRows(), output.mRows());
				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		output.transferToHost();

		output.resize(cols(), rows(), false);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < colsTemp; ++j){
			for (int i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = std::tanh(dataTemp[mrowsTemp * j + i]);
			}
		}

		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::tanh(){
		return tanh(*this);
	}
	
	template<class T>
	int LaspMatrix<T>::colSqSum(LaspMatrix<T>& output, T scalar){

		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(), 1);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_colSqSum(params, dData(), cols(), rows(), output.dData(), scalar, mRows(), output.mRows());

				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		output.transferToHost();

		T* newptr = new T[cols()];

		memset(newptr, 0, cols()*sizeof(T));

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		for(int i = 0; i < colsTemp; ++i){
			for (int j = 0; j < rowsTemp; ++j){
				T val = dataTemp[ i * mrowsTemp + j];
				newptr[i] += val * val;
			}
			newptr[i] *= scalar;
		}
		output = LaspMatrix<T>(cols(), 1, newptr);

		return MATRIX_SUCCESS;
		
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::colSqSum(T scalar){
		LaspMatrix<T> output;
		int err = colsSqSum(output, scalar);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::colSum(LaspMatrix<T>& output, T scalar){
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(), 1);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_colSum(params, dData(), cols(), rows(), output.dData(), scalar, mRows(), output.mRows());

				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		output.transferToHost();

		T* newptr = new T[cols()];
		
		memset(newptr, 0, cols()*sizeof(T));
		
		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();
		
		for(int i = 0; i < colsTemp; ++i){
			for (int j = 0; j < rowsTemp; ++j){
				T val = dataTemp[ i * mrowsTemp + j];
				newptr[i] += val;
			}
			newptr[i] *= scalar;
		}
		output = LaspMatrix<T>(cols(), 1, newptr);
		
		return MATRIX_SUCCESS;
		
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::colSum(T scalar){
		LaspMatrix<T> output;
		int err = colSum(output, scalar);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::eWiseOp(LaspMatrix<T> &output, T add, T mult, T pow1){
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_ewiseOp(params, dData(), output.dData(), size(), add, mult, pow1, rows(), mRows(), output.mRows());

				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		output.transferToHost();

		output.resize(cols(), rows(), false);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		if(pow1 == 1 || mult == 0){
			#pragma omp parallel for if(ompCount > ompLimit)
			for (int j = 0; j < colsTemp; ++j){
				for ( int i = 0; i < rowsTemp; ++i){
					output_dataTemp[j * output_mrowsTemp + i] =  dataTemp[j * mrowsTemp + i] * mult + add;
				}
			}	
		} else if(pow1 == 2.0){
			#pragma omp parallel for if(ompCount > ompLimit)
			for (int j = 0; j < colsTemp; ++j){
				for ( int i = 0; i < rowsTemp; ++i){
					output_dataTemp[j * output_mrowsTemp + i] =  dataTemp[j * mrowsTemp + i] * dataTemp[j * mrowsTemp + i] * mult + add;
				}
			}
		} 
		else {
			#pragma omp parallel for if(ompCount > ompLimit)
			for (int j = 0; j < colsTemp; ++j){
				for ( int i = 0; i < rowsTemp; ++i){
					output_dataTemp[j * output_mrowsTemp + i] =  std::pow(dataTemp[j * mrowsTemp + i], pow1) * mult + add;
				}
			}
		}

		return MATRIX_SUCCESS;
	
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseOp(T add, T mult, T pow1){
		LaspMatrix<T> output;
		int err = eWiseOp(output, add, mult, pow1);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::eWiseDivM(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1, T pow2){
		if(otherMatrix.rows() != rows() || otherMatrix.cols() != cols()){
			#ifndef NDEBUG
			cerr << "Error: Matrix dimensions must be equal for eWiseDivM" << endl;
			#endif
			return INVALID_DIMENSIONS;
		}
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows(), false);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix, &output);
				device_eWiseDiv(params, dData(), otherMatrix.dData(), output.dData(), size(), pow1, pow2, rows(), mRows(), otherMatrix.mRows(), output.mRows());

				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		otherMatrix.transferToHost();
		output.transferToHost();

		output.resize(cols(), rows(), false);

		int rowsTemp = rows();
		int colsTemp = cols();
		int mrowsTemp = mRows();
		T* dataTemp = data();

		int output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();

		T* other_dataTemp = otherMatrix.data();
		int other_mrowsTemp = otherMatrix.mRows();

		#ifdef _OPENMP
		int ompCount = rowsTemp * colsTemp;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for( int j = 0; j < colsTemp; ++j){
			for( int i = 0; i < rowsTemp; ++i){
				output_dataTemp[j * output_mrowsTemp + i] = std::pow(dataTemp[j*mrowsTemp + i], pow1) / std::pow(other_dataTemp[j * other_mrowsTemp + i], pow2);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseDivM(LaspMatrix<T>& otherMatrix, T pow1, T pow2){
		LaspMatrix<T> output;
		int err = eWiseDivM(otherMatrix, output, pow1, pow2);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}

	template<class T>
	int LaspMatrix<T>::eWiseScale(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix, int d0, int d){
		if (device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &outputMatrix);
				return device_eWiseScale(params, dData(), outputMatrix.dData(), otherMatrix.dData(),rows(),cols(), d0);
			}
		}

		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();

		for (int i=d0+1; i<=d; ++i){
			for (int j=0; j< rows(); j++){
				outputMatrix(j, i) = this->operator()((i-(d0+1)), j) * otherMatrix(j);
			}
		}
	
		return 0;
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseScale(LaspMatrix<T>& otherMatrix, int d0, int d){
		LaspMatrix<T> output;
		int err = eWiseScale(otherMatrix, output, d0, d);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::swap(){
		if(isSubMatrix()){
			#ifndef NDEBUG
			cerr << "Cannot swap a sub-matrix" << endl;
			#endif
			return CANNOT_COMPLETE_OPERATION;
		}

		std::swap(_rows(), _cols());
		std::swap(_mRows(), _mCols());
		return MATRIX_SUCCESS;
	}


	template<class T>
	template<class ITER>
	int LaspMatrix<T>::gather(LaspMatrix<T>& output, ITER begin, ITER end){
		
		bool copy = output.key() == key();
		long size = end - begin;

		if(copy){
			cerr << "ERROR: Gather into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}

		if (device()){
			int* map = new int[size];
			std::copy(begin, end, map);
			LaspMatrix<int> mapMat(size, 1, map);
			return gather(output, mapMat);
		}
		else{
			output.transferToHost();
			output.resize(size, rows(), false);

			int ind = 0;

			int rowsTemp = rows();
			int colsTemp = cols();
			int mRowsTemp = mRows();
			int output_mrowsTemp = output.mRows();

			T* dataTemp = data();
			T* output_dataTemp = output.data();


			for (ITER i = begin; i != end; ++i, ++ind){
				for(int j = 0; j < rowsTemp; ++j){
					if(*i >= colsTemp){
						cerr << "ERROR: Map index for gather is out of bounds" << endl;
						return OUT_OF_BOUNDS;
					}

					output_dataTemp[ind * output_mrowsTemp + j] = dataTemp[(*i) * mRowsTemp + j];
				}
			}

			return MATRIX_SUCCESS;
		}
	}
	
	template<class T>
	int LaspMatrix<T>::gather(LaspMatrix<T>& output, LaspMatrix<int> map){
		int size = map.size();

		bool copy = output.key() == key();
		if(copy){
			cerr << "ERROR: Gather into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}
		if (device()){
			int error = MATRIX_SUCCESS;
			error += map.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(size, rows(), false);


			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_gather(params, map.dData(), dData(), output.dData(), rows(), mRows(), output.mRows(), size);
				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		map.transferToHost();
		output.transferToHost();

		output.resize(size, rows(), false);

		T* output_dataTemp = output.data();
		T* dataTemp = data();

		int rowsTemp = rows();
		int output_mrowsTemp = output.mRows();
		int mRowsTemp = mRows();

		#ifdef _OPENMP
		int ompCount = rowsTemp * size;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for (int ind = 0; ind < size; ++ind){
			int map_ind = map(ind);

			for(int j = 0; j < rowsTemp; ++j){

				output_dataTemp[ind * output_mrowsTemp + j] = dataTemp[map_ind * mRowsTemp + j];
			}
		}

		return MATRIX_SUCCESS;
	
	}
	
	template<class T>
	int LaspMatrix<T>::gather(LaspMatrix<T>& output, vector<int>& map){
		if(device()){
			return gather(output, map.begin(), map.end());
		}

		int size = map.size();

		bool copy = output.key() == key();
		if(copy){
			cerr << "ERROR: Gather into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}

		transferToHost();
		output.transferToHost();

		output.resize(size, rows(), false);

		T* output_dataTemp = output.data();
		T* dataTemp = data();

		int rowsTemp = rows();
		int output_mrowsTemp = output.mRows();
		int mRowsTemp = mRows();

		#ifdef _OPENMP
		int ompCount = rowsTemp * size;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for (int ind = 0; ind < size; ++ind){
			int map_ind = map[ind];

			for(int j = 0; j < rowsTemp; ++j){

				output_dataTemp[ind * output_mrowsTemp + j] = dataTemp[map_ind * mRowsTemp + j];
			}
		}

		return MATRIX_SUCCESS;
	}
	
	template<class T>
	template<class ITER>
	LaspMatrix<T> LaspMatrix<T>::gather(ITER begin, ITER end){
		LaspMatrix<T> output;
		
		int err = gather(output, begin, end);
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		
		return output;		
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::gather(LaspMatrix<int> map){
		LaspMatrix<T> output;
		
		int err = gather(output, map);
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		
		return output;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::gather(vector<int>& map){
		LaspMatrix<T> output;
		
		int err = gather(output, map);
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		
		return output;
	}
	
	
	
	template<class T>
	template<class ITER>
	int LaspMatrix<T>::gatherSum(LaspMatrix<T>& output, ITER begin, ITER end){
		if (output.rows() > 1 && output.cols() > 1) {
			cerr << "ERROR: Gather sum ouput must be a vector" << endl;
			return CANNOT_COMPLETE_OPERATION;
		}
		
		bool copy = output.key() == key();
		long size = end - begin;
		
		if(copy){
			cerr << "ERROR: Gather sum into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}
		
		if (device()){
			int* map = new int[size];
			std::copy(begin, end, map);
			LaspMatrix<int> mapMat(size, 1, map);
			return gatherSum(output, mapMat);
		}
		else{
			if(output.size() < rows()){
				output.resize(1, rows(), false, true);
			}

			int ind = 0;
			
			int rowsTemp = rows();
			int colsTemp = cols();
			int mRowsTemp = mRows();
			
			T* dataTemp = data();
			T* output_dataTemp = output.data();
			int stride = output.rows() == 1 ? output.mRows() : 1;
			
			for(int j = 0; j < rowsTemp; ++j){
				for (ITER i = begin; i != end; ++i, ++ind){
					if(*i >= colsTemp){
						cerr << "ERROR: Map index for gather is out of bounds" << endl;
						return OUT_OF_BOUNDS;
					}
					
					output_dataTemp[j * stride] += dataTemp[(*i) * mRowsTemp + j];
				}
			}
			
			return MATRIX_SUCCESS;
		}
	}
	
	template<class T>
	int LaspMatrix<T>::gatherSum(LaspMatrix<T>& output, LaspMatrix<int> map){
		if (output.rows() > 1 && output.cols() > 1) {
			cerr << "ERROR: Gather sum ouput must be a vector" << endl;
			return CANNOT_COMPLETE_OPERATION;
		}
		
		int size = map.size();
		
		bool copy = output.key() == key();
		if(copy){
#ifndef NDEBUG
			cerr << "ERROR: Gather sum into same matrix not supported" << endl;
#endif
			return INVALID_LOCATION;
		}
		if (device()){
			int error = MATRIX_SUCCESS;
			error += map.transferToDevice();
			error += output.transferToDevice();
			if(output.size() < rows()){
				error += output.resize(size, 1, false, true);
			}
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_gatherSum(params, map.dData(), dData(), output.dData(), rows(), mRows(), output.mRows(), size, output.rows());
				return MATRIX_SUCCESS;
			}
		}

		transferToHost();
		map.transferToHost();
		output.transferToHost();

		if(output.size() < rows()){
			output.resize(1, rows(), false, true);
		}
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		
		int rowsTemp = rows();
		int output_mrowsTemp = output.mRows();
		int mRowsTemp = mRows();
		int stride = output.rows() == 1 ? output.mRows() : 1;
		
		#ifdef _OPENMP
		int ompCount = rowsTemp * size;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < rowsTemp; ++j){
			for (int ind = 0; ind < size; ++ind){
				int map_ind = map(ind);
				output_dataTemp[j * stride] += dataTemp[map_ind * mRowsTemp + j];
			}
		}
		
		return MATRIX_SUCCESS;
	
	}
	
	template<class T>
	int LaspMatrix<T>::gatherSum(LaspMatrix<T>& output, vector<int>& map){
		if(device()){
			return gatherSum(output, map.begin(), map.end());
		}

		if (output.rows() > 1 && output.cols() > 1) {
			cerr << "ERROR: Gather sum ouput must be a vector" << endl;
			return CANNOT_COMPLETE_OPERATION;
		}
		
		int size = map.size();
		
		bool copy = output.key() == key();
		if(copy){
#ifndef NDEBUG
			cerr << "ERROR: Gather sum into same matrix not supported" << endl;
#endif
			return INVALID_LOCATION;
		}

		transferToHost();
		output.transferToHost();

		if(output.size() < rows()){
			output.resize(1, rows(), false, true);
		}
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		
		int rowsTemp = rows();
		int output_mrowsTemp = output.mRows();
		int mRowsTemp = mRows();
		int stride = output.rows() == 1 ? output.mRows() : 1;
		
		#ifdef _OPENMP
		int ompCount = rowsTemp * size;
		int ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(int j = 0; j < rowsTemp; ++j){
			for (int ind = 0; ind < size; ++ind){
				int map_ind = map[ind];
				output_dataTemp[j * stride] += dataTemp[map_ind * mRowsTemp + j];
			}
		}
		
		return MATRIX_SUCCESS;

	}
	
	
	
	template<class T>
	int LaspMatrix<T>::solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, LaspMatrix<T>& LU, LaspMatrix<int>& ipiv){
		cerr << "Error: Solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output){
		cerr << "Error: Solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::solve(LaspMatrix<T>& otherMatrix){
		cerr << "Error: Solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}

	template<class T>
	void LaspMatrix<T>::printMatrix(string name, int c, int r){
		bool transfer = device();
		transferToHost();

		cout.precision(16);
		if(c == 0)
			c = cols();
		if(r == 0)
			r = rows();
		
		if(!name.empty()){
			cout << name << ":" << endl;
		}
		for(int i=0; i < r; ++i){
			for (int j=0; j< c; ++j){
				cout << setw(20) << data()[j*mRows()+i]  << " ";
			}
			cout << endl;
		}

		cout.precision(8);

		if(transfer){
	#ifndef NDEBUG
			cerr << "Warning: printing device matrix requires expensive memory transfer";
	#endif
			transferToDevice();
		}
	}

	template<class T>
	void LaspMatrix<T>::printInfo(string name){


		cout.precision(5);
		
		if(!name.empty()){
			cout << name << ":" << endl;
		}
		
		cout << "Size: (" << cols() << ", " << rows() << "), mSize: (" << cols() << ", " << rows() << ")" << endl;
		cout << "On device: " << device() << endl;

	}

	template<class T>
	int LaspMatrix<T>::addMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out){
		if (device()){
			b.transferToDevice();
			out.transferToDevice();
			DeviceParams params = context().setupOperation(this, &out);
			device_addMatrix(params, dData(), b.dData(), out.dData(), rows(), cols(), mRows(), b.mRows(), out.mRows());
		}

		else{
			b.transferToHost();
			out.transferToHost();

			int size = rows()*cols();
			for (int i = 0; i < size; ++i){
				out.data()[i] = data()[i] + b.data()[i];
			}
		}
		return 0;

	}


	template<class T>
	int LaspMatrix<T>::subMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out){
		if (device()){
			b.transferToDevice();
			out.transferToDevice();
			DeviceParams params = context().setupOperation(this, &out);
			device_subMatrix(params, dData(), b.dData(), out.dData(), rows(), cols(), mRows(), b.mRows(), out.mRows());
		}

		else{
			b.transferToHost();
			out.transferToHost();

			int size = rows()*cols();

			for (int i = 0; i < size; ++i){
				out.data()[i] = data()[i] - b.data()[i];
			}
		}
		return 0;
	}


	
	template<class T>
	int LaspMatrix<T>::ger(LaspMatrix<T>& output, LaspMatrix<T> X, LaspMatrix<T> Y, T alpha){
		cerr << "Error: Ger not implemented for type!" << endl;
		return METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::ger(LaspMatrix<T> X, LaspMatrix<T> Y, T alpha){
		return ger(*this, X, Y, alpha);
	}
	
	
	//Hacky way of checking for the same types
	template<class T, class N>
	inline bool sameType(T x, N y){
		return false;
	}
	
	template<>
	inline bool sameType(float x, float y){
		return true;
	}
	
	template<>
	inline bool sameType(double x, double y){
		return true;
	}
	
	template<>
	inline bool sameType(int x, int y){
		return true;
	}
	
	
  	template<class T>
  	template<class N>
	LaspMatrix<N> LaspMatrix<T>::convert(bool mem){
		//Check that T and N are not already the same type
		T x = 1;
		N y = 1;
		if (sameType(x, y)){
			return *(reinterpret_cast<LaspMatrix<N>*>(this));
		}
		
		LaspMatrix<N> result;

		if(device()){
			int error = result.transferToDevice();
			error += result.resize(cols(), rows(), false);

			if(error == MATRIX_SUCCESS){
				//temporary hack, templating didn't work immediately moving on TODO
				DeviceParams params = context().setupOperation(this, this);
				device_convert(params, dData(), result.dData(), rows(), cols(), mRows());
				return result;
			}
		}

		transferToHost();

		if(mem){
			result.resize(mCols(), mRows());
		}

		result.resize(cols(), rows());
		
		N* output_dataTemp = result.data();
		T* dataTemp = data();

		int rowsTemp = rows();
		int colsTemp = cols();
		int output_mrowsTemp = result.mRows();
		int mRowsTemp = mRows();

		for(int i = 0; i < colsTemp; ++i){
			for(int j = 0; j < rowsTemp; ++j){
				output_dataTemp[i * output_mrowsTemp + j] = static_cast<N>(dataTemp[mRowsTemp * i + j]);
			}
		}
		
		return result;
	}
	
	template<class T>
	bool LaspMatrix<T>::operator==(LaspMatrix<T>& other){
		if(key() == other.key()){
			return true;
		}
		
		if(rows() != other.rows() || cols() != other.cols()){
			return false;
		}
		
		bool result = true;
		for(int i = 0; i < cols(); ++i){
			for(int j = 0; j < rows(); ++j){
				T epsilon = operator()(i, j) / 1000.0;
				if(abs(operator()(i, j) - other(i, j)) > epsilon){
					result = false;
				}
			}
		}
		
		return result;
	}
	
	template<class T>
	void LaspMatrix<T>::checksum(string name){
		T sum = 0;
		T xor_sum = 0;
		for(int i = 0; i < rows(); ++i){
			for(int j = 0; j < cols(); ++j){
				T elem = operator()(j, i);
				sum += elem;
			}
		}
		
		cout << name << ", sum: " << sum << ", xor: " << xor_sum << endl;
	}

	template<class T>
	template<class N>
	N* LaspMatrix<T>::getRawArrayCopy(){
		if(device()){
			transferToHost();
		}

		N* output = new N[rows() * cols()];
		
		for (int jj = 0; jj < cols(); jj++) {
			for (int ii = 0; ii < rows(); ii++) {
				output[rows() * jj + ii] = static_cast<N>(operator()(jj, ii));
			}
		}

		return output;
	}
	
	template<>
	inline int LaspMatrix<float>::multiply(LaspMatrix<float>& otherMatrix, LaspMatrix<float>& outputMatrix, bool transposeMe, bool transposeOther, float a, float b, int numRowsToSkip){
		int myRows = transposeMe ? cols() : rows();
		int myCols = transposeMe ? rows() : cols();
		int otherRows = transposeOther ? otherMatrix.cols() : otherMatrix.rows();
		int otherCols = transposeOther ? otherMatrix.rows() : otherMatrix.cols();
		
		if (myCols != (transposeOther ? otherRows : otherRows - numRowsToSkip)) {
			cerr << "Error: Dimension mismatch in multiply" << endl;
			return INVALID_DIMENSIONS;
		}
		
		int numToSkip = 0;
		if (!transposeOther) {
			numToSkip = numRowsToSkip;
			numRowsToSkip = 0;
		}
		
		int m = myRows;
		int n = otherCols - numRowsToSkip;
		int k = myCols;
		
		//Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		
		float alpha(a);
		float beta(b);
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();
			error += outputMatrix.resize(n, m, copy);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix);
				return device_sgemm(params, transposeMe, transposeOther, m, n, k, alpha, dData(), mRows(), otherMatrix.dData() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, outputMatrix.dData(), outputMatrix.mRows());
			}
		}

		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		outputMatrix.resize(n, m, copy);
		host_sgemm(transposeMe, transposeOther, m, n, k, alpha, data(), mRows(), otherMatrix.data() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, outputMatrix.data(), outputMatrix.mRows());
	
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<float>::solve(LaspMatrix<float>& otherMatrix, LaspMatrix<float>& output, LaspMatrix<float>& LU, LaspMatrix<int>& ipiv){
		if (cols() != rows() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in solve" << endl;
			return INVALID_DIMENSIONS;
		}

		//Apparently the cuda getrfbatched kernel is terrible for single solves
		transferToHost();
		
		int n = rows();
		int nrhs = otherMatrix.cols();

		int error = MATRIX_SUCCESS;
		
		LaspMatrix<float> A;

		if(device()){
			error += A.transferToDevice();
		}

		if (LU.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		
		int lda = A.mRows();
		LaspMatrix<int> ipivOut(1, n);
		LaspMatrix<float> B;

		if(device()){
			error += otherMatrix.transferToDevice();
			error += B.transferToDevice();
			error += ipivOut.transferToDevice();
		}

		if (output.key() == otherMatrix.key()){
			B = otherMatrix;
		} else {
			B.copy(otherMatrix);
		}
		
		int ldb = B.mRows();
		int info = 0;

		if(device() && error == MATRIX_SUCCESS){			
			DeviceParams params = context().setupOperation(this, &otherMatrix);
			device_sgesv(params, n, nrhs, A.dData(), lda, ipivOut.dData(), B.dData(), ldb);
		} else {

			A.transferToHost();
			ipivOut.transferToHost();
			B.transferToHost();
			otherMatrix.transferToHost();

			host_sgesv(n, nrhs, A.data(), lda, ipivOut.data(), B.data(), ldb);
			
			if(info < 0){
	#ifndef NDEBUG
				cerr << "Argument " << -info << " to gesv invalid!" << endl;
	#endif
				return ARGUMENT_INVALID;
			} else if (info > 0){
	#ifndef NDEBUG
				cerr << "Factor " << info << " in gesv is singular!" << endl;
	#endif
				return CANNOT_COMPLETE_OPERATION;
			}
		}
		
		ipiv = ipivOut;
		output = B;
		LU = A;
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<float>::solve(LaspMatrix<float>& otherMatrix, LaspMatrix<float>& output){
		LaspMatrix<float> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, output, LU, ipiv);
	}
	
	template<>
	inline int LaspMatrix<float>::solve(LaspMatrix<float>& otherMatrix){
		LaspMatrix<float> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, otherMatrix, LU, ipiv);
	}
	
	template<>
	inline int LaspMatrix<float>::ger(LaspMatrix<float>& output, LaspMatrix<float> X, LaspMatrix<float> Y, float alpha){	  
		int m = rows();
		int n = cols();
		
		int incx;
		if(!((X.rows() >= m && X.cols() == 1) || (X.cols() >= m && X.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (X.cols() == m){
			incx = X.mRows();
		} else {
			incx = 1;
		}
		
		int incy;
		if(!((Y.rows() >= n && Y.cols() == 1) || (Y.cols() >= n && Y.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (Y.cols() == n){
			incy = Y.mRows();
		} else {
			incy = 1;
		}
		
		LaspMatrix<float> A;
		if (output.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		if(A.device()){
			int error = MATRIX_SUCCESS;
			error += X.transferToDevice();
			error += Y.transferToDevice();

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(&A, &X, &Y);
				return device_sger(params, m, n, alpha, X.dData(), incx, Y.dData(), incy, A.dData(), A.mRows());
			}
		}

		transferToHost();
		X.transferToHost();
		Y.transferToHost();
		host_sger(m, n, alpha, X.data(), incx, Y.data(), incy, A.data(), A.mRows());
	
		output = A;
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline void LaspMatrix<double>::checksum(string name){
		double sum = 0;
		char xor_sum [sizeof(double) + 1];
		
		for (int k = 0; k < sizeof(double); ++k) {
			xor_sum[k] = 0;
		}
		
		for(int i = 0; i < rows(); ++i){
			for(int j = 0; j < cols(); ++j){
				double e = operator()(j, i);
				char xor_e [sizeof(double)];
				memcpy((void*)xor_e, (void*)&e,  sizeof(double));
				sum += operator()(j, i);
				for (int k = 0; k < sizeof(double); ++k) {
					xor_sum[k] = xor_sum[k] ^ xor_e[k];
				}
			}
		}
		
		cout << name.c_str() << ", sum: " << setprecision(40) << sum << ", xor: ";

		for (int k = 0; k < sizeof(double); ++k) {
			cout << setbase(16) << (unsigned short) xor_sum[k] << " ";
		}
		
		cout << setbase(10) << setprecision(6);
		cout << endl;
	}
	
	template<>
	inline int LaspMatrix<double>::multiply(LaspMatrix<double>& otherMatrix, LaspMatrix<double>& outputMatrix, bool transposeMe, bool transposeOther, double a, double b, int numRowsToSkip){
		int myRows = transposeMe ? cols() : rows();
		int myCols = transposeMe ? rows() : cols();
		int otherRows = transposeOther ? otherMatrix.cols() : otherMatrix.rows();
		int otherCols = transposeOther ? otherMatrix.rows() : otherMatrix.cols();
		
		if (myCols != (transposeOther ? otherRows : otherRows - numRowsToSkip)) {
			cerr << "Error: Dimension mismatch in multiply" << endl;
			return INVALID_DIMENSIONS;
		}
		
		int numToSkip = 0;
		if (!transposeOther) {
			numToSkip = numRowsToSkip;
			numRowsToSkip = 0;
		}
		
		int m = myRows;
		int n = otherCols - numRowsToSkip;
		int k = myCols;
		
		//Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		
		//WHY?!?!?!?
		double alpha(a);
		double beta(b);
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();
			error += outputMatrix.resize(n, m, copy);

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix);
				return device_dgemm(params, transposeMe, transposeOther, m, n, k, alpha, dData(), mRows(), otherMatrix.dData() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, outputMatrix.dData(), outputMatrix.mRows());
			}
		}

		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		outputMatrix.resize(n, m, copy);
		host_dgemm(transposeMe, transposeOther, m, n, k, alpha, data(), mRows(), otherMatrix.data() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, outputMatrix.data(), outputMatrix.mRows());
	

		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<double>::solve(LaspMatrix<double>& otherMatrix, LaspMatrix<double>& output, LaspMatrix<double>& LU, LaspMatrix<int>& ipiv){
		if (cols() != rows() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in solve" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Apparently the cuda getrfbatched kernel is terrible for single solves
		transferToHost();

		int n = rows();
		int nrhs = otherMatrix.cols();
		
		int error = MATRIX_SUCCESS;

		LaspMatrix<double> A;

		if(device()){
			error += A.transferToDevice();
		}

		if (LU.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		
		int lda = A.mRows();
		LaspMatrix<int> ipivOut(1, n);
		LaspMatrix<double> B;

		if(device()){
			error += otherMatrix.transferToDevice();
			error += B.transferToDevice();
			error += ipivOut.transferToDevice();
		}

		if (output.key() == otherMatrix.key()){
			B = otherMatrix;
		} else {
			B.copy(otherMatrix);
		}

		int ldb = B.mRows();
		int info = 0;
		
		if(device() && error == MATRIX_SUCCESS){

			DeviceParams params = context().setupOperation(this, &otherMatrix);
			device_dgesv(params, n, nrhs, A.dData(), lda, ipivOut.dData(), B.dData(), ldb);

		} else {
			
			A.transferToHost();
			ipivOut.transferToHost();
			B.transferToHost();
			otherMatrix.transferToHost();

			host_dgesv(n, nrhs, A.data(), lda, ipivOut.data(), B.data(), ldb);
			
			if(info < 0){
	#ifndef NDEBUG
				cerr << "Argument " << -info << " to gesv invalid!" << endl;
	#endif
				return ARGUMENT_INVALID;
			} else if (info > 0){
	#ifndef NDEBUG
				cerr << "Factor " << info << " in gesv is singular!" << endl;
	#endif
				return CANNOT_COMPLETE_OPERATION;
			}
		}
		
		ipiv = ipivOut;
		output = B;
		LU = A;
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<double>::solve(LaspMatrix<double>& otherMatrix, LaspMatrix<double>& output){
		LaspMatrix<double> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, output, LU, ipiv);
	}
	
	template<>
	inline int LaspMatrix<double>::solve(LaspMatrix<double>& otherMatrix){
		LaspMatrix<double> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, otherMatrix, LU, ipiv);
	}
	

	template<>
	inline int LaspMatrix<double>::ger(LaspMatrix<double>& output, LaspMatrix<double> X, LaspMatrix<double> Y, double alpha){
		int m = rows();
		int n = cols();
		
		int incx;
		if(!((X.rows() >= m && X.cols() == 1) || (X.cols() >= m && X.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (X.cols() == m){
			incx = X.mRows();
		} else {
			incx = 1;
		}
		
		int incy;
		if(!((Y.rows() >= n && Y.cols() == 1) || (Y.cols() >= n && Y.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (Y.cols() == n){
			incy = Y.mRows();
		} else {
			incy = 1;
		}
		
		LaspMatrix<double> A;
		if (output.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		
		if(A.device()){
			int error = MATRIX_SUCCESS;
			error += X.transferToDevice();
			error += Y.transferToDevice();

			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(&A, &X, &Y);
				return device_dger(params, m, n, alpha, X.dData(), incx, Y.dData(), incy, A.dData(), A.mRows());
			}
		}
		transferToHost();
		X.transferToHost();
		Y.transferToHost();
		host_dger(m, n, alpha, X.data(), incx, Y.data(), incy, A.data(), A.mRows());

		output = A;
		return MATRIX_SUCCESS;
	}


	template<class T>
	void distance2_nomul(LaspMatrix<T>& output, LaspMatrix<T> a, LaspMatrix<T>  b, LaspMatrix<T>  a_norm, LaspMatrix<T> b_norm){
		LaspMatrix<T> dist;

		a.multiply(b,dist,true,false, -2.0);
				
		int sizedAOnes = dist.cols();
		int sizedBOnes = dist.rows();
		
		LaspMatrix<T> onesA(1,sizedAOnes,1.0);
		LaspMatrix<T> onesB(1,sizedBOnes,1.0);

		dist.ger(onesB, b_norm);
		dist.ger(a_norm,onesA);

		output = dist;
	}


	template<class T>
	void calc_rbf(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T> X2Param, LaspMatrix<T> Xnorm1Param, LaspMatrix<T> Xnorm2Param, T gamma){
		distance2_nomul(out, X1Param, X2Param, Xnorm1Param, Xnorm2Param);
		out.exp(gamma);
	}

	template<class T>
	void calc_lin(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T> X2Param){
		X1Param.multiply(X2Param,out,true,false);
		
	}
	template<class T>
	void calc_pol(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T>X2Param, T a, T c, T d){
		LaspMatrix<T> out1(X1Param.cols(), X2Param.cols(), c);
		X1Param.multiply(X2Param, out1, true, false, a, 1.0);
		out1.eWiseOp(out,0,1,d);
	}
	template<class T>
	void calc_sigmoid(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T>X2Param, T a, T c){
		LaspMatrix<T> out1(X1Param.cols(), X2Param.cols(), c);
		X1Param.multiply(X2Param, out1, true, false, a, 1.0);
		out1.tanh();
		out = out1;
	}

	template<class T>
	int LaspMatrix<T>::getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& Xnorm1, LaspMatrix<T>& X2, LaspMatrix<T>& Xnorm2, bool mult, bool transMult, bool useGPU){
		transferToHost();
		if(context().getNumDevices() > 0 && (useGPU || X1.device() || X2.device() || Xnorm1.device() || Xnorm2.device())){
			T *A, *Anorm, *B, *Bnorm, *out;
			int lda = X1.mRows(), aCols = X1.cols(), aRows = X1.rows();
			int ldb = X2.mRows(), bCols = X2.cols(), bRows = X2.rows();

			bool doKernel = !mult, trans = transMult, a_on_device = X1.device(), b_on_device = X2.device(), out_on_device = device();
			int aCPU = 1, bCPU = 1, aGPU = context().getNumDevices(), bGPU = 1, streams = 3, numDev = context().getNumDevices();

			if(!transMult){
				resize(bCols, aCols);
			} else {
				resize(bRows, aRows);
			}

			int ldOut = mRows();

			if(X1.device()){
				Xnorm1.transferToDevice();
				A = X1.dData();
				Anorm = Xnorm1.dData();
			} else {
				Xnorm1.transferToHost();
				A = X1.data();
				Anorm = Xnorm1.data();
			}

			if(X2.device()){
				Xnorm2.transferToDevice();
				B = X2.dData();
				Bnorm = Xnorm2.dData();
			} else {
				Xnorm2.transferToHost();
				B = X2.data();
				Bnorm = Xnorm2.data();
			}

			if(device()){
				out = dData();
			} else {
				out = data();
			}

			if(doKernel && kernelOptions.kernel == RBF && (Xnorm1.cols() != X1.cols() || Xnorm2.cols() != X2.cols() || Xnorm1.mRows() != 1 || Xnorm2.mRows() != 1)){
				cerr << "Improper inputs to getKernel!" << endl;
				return UNSPECIFIED_MATRIX_ERROR;
			}

			DeviceParams params = context().setupOperation(this);
			return pinned_kernel_multiply(params, A, lda, aCols, Anorm, aRows, B, ldb, bCols, Bnorm, bRows, out, ldOut, kernelOptions, doKernel, aCPU, bCPU, aGPU, bGPU, streams, numDev, a_on_device, b_on_device, out_on_device, trans);

		} else {
			  if(mult){
				bool trans1 = !transMult, trans2 = transMult;
				X1.multiply(X2, *this, trans1, trans2);

				return MATRIX_SUCCESS;
			  }

			  if (kernelOptions.kernel == RBF){
			   	 calc_rbf(*this, X1, X2, Xnorm1, Xnorm2, static_cast<T>(kernelOptions.gamma));
			  }
			  if (kernelOptions.kernel == LINEAR){
			    calc_lin(*this, X1, X2);
			  }
			  if (kernelOptions.kernel == POLYNOMIAL){
			    calc_pol(*this, X1, X2, static_cast<T>(kernelOptions.gamma), static_cast<T>(kernelOptions.coef), static_cast<T>(kernelOptions.degree));
			  }
			  if (kernelOptions.kernel == SIGMOID){
			    calc_sigmoid(*this, X1, X2, static_cast<T>(kernelOptions.gamma), static_cast<T>(kernelOptions.coef));
  			  }
		}

		return MATRIX_SUCCESS;
	}

	
#ifndef WUCUDA
	
	template<class T>
	int LaspMatrix<T>::transfer(){
	#ifndef NDEBUG
		cerr << "Warning: Transfer not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::transferToHost(){
	#ifndef NDEBUG
		cerr << "Warning: Transfer to host not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::transferToDevice(){
	#ifndef NDEBUG
		cerr << "Warning: Transfer to device not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::registerHost(){
	#ifndef NDEBUG
		cerr << "Warning: Transfer to device not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::deviceSetRow(int row, LaspMatrix<T>& other){
	#ifndef NDEBUG
		cerr << "Warning: Device set row not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::deviceSetRow(int row, LaspMatrix<T>& other, int otherRow){
	#ifndef NDEBUG
		cerr << "Warning: Device set row not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::deviceSetCol(int col, LaspMatrix<T>& other){
	#ifndef NDEBUG
		cerr << "Warning: Device set col not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::deviceSetCol(int col, LaspMatrix<T>& other, int otherCol){
	#ifndef NDEBUG
		cerr << "Warning: Device set col not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::deviceResize(int newCols, int newRows, bool copy, bool fill, T val){
	#ifndef NDEBUG
		cerr << "Warning: Device resize not supported" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	int LaspMatrix<T>::deviceCopy(LaspMatrix<T> &other){
	#ifndef NDEBUG
		cerr << "Warning: Copy on device not supported, leaving data on host" << endl;
	#endif
		return CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	LaspMatrix<T> LaspMatrix<T>::deviceCopy(){
	#ifndef NDEBUG
		cerr << "Warning: Copy on device not supported, leaving data on host" << endl;
	#endif
		throw CANNOT_COMPLETE_OPERATION;
	}

	template<class T>
	void LaspMatrix<T>::freeData(){
		if (_data() != 0){
			delete [] _data();
		}
	}

#else

    template<class T>
				int LaspMatrix<T>::transfer(){
					if(isSubMatrix()){
			#ifndef NDEBUG
						cerr << "Warning: Transferring a sub-matrix" << endl;
			#endif
					}

					context().setupMemTransfer(this);
					if(device()) {
						if(_registered()){
								CUDA_CHECK(cudaHostUnregister(_data()));
								_registered() = false;
						} else {
							_data() = new T[_mRows() * _mCols()];
							if(dData() != 0){
									CUDA_CHECK(cudaMemcpy((void*)_data(), (void*)_dData(), _mRows() * _mCols() * sizeof(T), cudaMemcpyDeviceToHost));
									CUDA_CHECK(cudaFree((void*)_dData()));
							}
						}

						_device() = false;

					} else {
						if(mSize() != 0){
							if((static_cast<size_t>(mSize()) * sizeof(T)) >= context().getAvailableMemory()){
								return UNSPECIFIED_MATRIX_ERROR;
							}
							CUDA_CHECK(cudaMalloc((void**)dData_, _mRows() * _mCols() * sizeof(T)));
							CUDA_CHECK(cudaMemcpy((void*)_dData(), (void*)_data(), _mRows() * _mCols() * sizeof(T), cudaMemcpyHostToDevice));
						} else {
							_dData() = 0;
						}
						delete [] data();
						_data() = 0;
						_device() = true;

					}
					return MATRIX_SUCCESS;
				}

    template<class T>
				int LaspMatrix<T>::transferToDevice(){
					if(!device()){
						return transfer();
					} else{
						return MATRIX_SUCCESS;
					}
				}

    template<class T>
				int LaspMatrix<T>::transferToHost(){
					if(device()){
						return transfer();
					} else{
						return MATRIX_SUCCESS;
					}
				}

    template<class T>
				int LaspMatrix<T>::registerHost(){
					if(!device()){
						CUDA_CHECK(cudaHostRegister((void*)_data(), _mRows() * _mCols() * sizeof(T), cudaHostRegisterMapped));
						CUDA_CHECK(cudaHostGetDevicePointer((void**)&(_dData()), (void*)_data(), 0));  
						_device() = true;
						_registered() = true;
					} else{
    	#ifndef NDEBUG
						cerr << "Warning: Data already on device or registered" << endl;
		#endif
						return MATRIX_SUCCESS;
					}
				}

    template<class T>
				int LaspMatrix<T>::deviceResize(int newCols, int newRows, bool copy, bool fill, T val){
					if(!device()){
						return CANNOT_COMPLETE_OPERATION;
					}

					if ((newCols == 1 && rows() == 1 && cols() != 1) || (newRows == 1 && cols() == 1 && rows() != 1)){
						std::swap(_rows(), _cols());
						std::swap(_mRows(), _mCols());
					}

					if(mRows() < newRows || mCols() < newCols){
						context().setupMemTransfer(this);

						if ((max(newCols, cols()) * max(newRows, rows()) * 8) / 1000000000.0 > 1.0){
						#ifndef NDEBUG
							cerr << "Allocating size: " << (max(newCols, cols()) * max(newRows, rows()) * sizeof(T)) / 1000000000.0 << " GB" << endl;
						#endif
						}

						T* newptr = 0;
						CUDA_CHECK(cudaMalloc((void**)&newptr, max(newCols, cols()) * max(newRows, rows()) * sizeof(T)));

						if(copy && rows() > 0 && cols() > 0 && newRows > 0 && newCols > 0 && newptr != 0 && dData() != 0){
							CUDA_CHECK(cudaMemcpy((void*)newptr, (void*)dData(), mSize() * sizeof(T), cudaMemcpyDeviceToDevice));
						}

						if(fill){
							CUDA_CHECK(cudaMemset(newptr, 0, sizeof(T) * newRows * newCols));
						}

						if(dData() != 0){
							if(_registered()){
								CUDA_CHECK(cudaHostUnregister(_data()));
								delete [] _data();
							} else {
								CUDA_CHECK(cudaFree((void*)dData()));
							}
						}

						_dData() = newptr;
						_mCols() = newCols;
						_mRows() = newRows;

					} 
					_cols() = newCols;
					_rows() = newRows;

					if(fill){
						eWiseOp(*this, val, 0, 0);
					}

					return MATRIX_SUCCESS;
				}

	//Type agnostic wrapper for cublas<T>copy()
	template<class T>
				int setRowHelper(cublasHandle_t& handle, int n, const T* x, int incx, T* y, int incy){
	#ifndef NDEBUG
					cerr << "Device to device row copy not supported for this type!" << endl;
	#endif
					return	CUBLAS_STATUS_INVALID_VALUE;
				}

	template<>
				inline int setRowHelper(cublasHandle_t& handle, int n, const float* x, int incx, float* y, int incy){
					return cublasScopy(handle, n, x, incx, y, incy);
				}

	template<>
				inline int setRowHelper(cublasHandle_t& handle, int n, const double* x, int incx, double* y, int incy){
					return cublasDcopy(handle, n, x, incx, y, incy);
				}

	template<class T>
				int LaspMatrix<T>::deviceSetRow(int row, LaspMatrix<T>& other){
					if(!(other.cols() == 1 || (other.rows() == 1 && other.mRows() == 1))){
			#ifndef NDEBUG
						cerr << "Warning: Device set row requires a contiguous memory vector" << endl;
			#endif
						return CANNOT_COMPLETE_OPERATION;
					}

					DeviceParams params = context().setupMemTransfer(&other, this);

					if(device() && !other.device()){
						T* rowPtr = dData() + row;
						T* vecPtr = other.data();
						CUBLAS_CHECK(cublasSetVector(cols(), sizeof(T), vecPtr, 1, rowPtr, mRows()));
					} else if(!device() && other.device()){
						T* rowPtr = data() + mRows() * row;
						T* vecPtr = other.dData();
						CUBLAS_CHECK(cublasGetVector(cols(), sizeof(T), vecPtr, 1, rowPtr, mRows()));
					} else if(device() && other.device()){
						T* rowPtr = dData() + mRows() * row;
						T* vecPtr = other.dData();
						CUBLAS_CHECK(setRowHelper(context().getCuBlasHandle(), cols(), vecPtr, 1, rowPtr, mRows()));
					}

					return MATRIX_SUCCESS;
				}

	template<class T>
				int LaspMatrix<T>::deviceSetRow(int row, LaspMatrix<T>& other, int otherRow){

					DeviceParams params = context().setupMemTransfer(&other, this);

					if(device() && !other.device()){
						T* rowPtr = dData() + row;
						T* vecPtr = other.data() + otherRow;
						CUBLAS_CHECK(cublasSetVector(cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(!device() && other.device()){
						T* rowPtr = data() + mRows() * row;
						T* vecPtr = other.dData() + otherRow;
						CUBLAS_CHECK(cublasGetVector(cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(device() && other.device()){
						T* rowPtr = dData() + mRows() * row;
						T* vecPtr = other.dData() + otherRow;
						CUBLAS_CHECK(setRowHelper(context().getCuBlasHandle(), cols(), vecPtr, other.mRows(), rowPtr, mRows()));
					}

					return MATRIX_SUCCESS;
				}

	template<class T>
				int LaspMatrix<T>::deviceSetCol(int col, LaspMatrix<T>& other){
					if(!(other.cols() == 1 || (other.rows() == 1 && other.mRows() == 1))){
			#ifndef NDEBUG
						cerr << "Warning: Device set col requires a contiguous memory vector" << endl;
			#endif
						return CANNOT_COMPLETE_OPERATION;
					}

					context().setupMemTransfer(&other, this);

					if(device() && !other.device()){
						T* colPtr = dData() + mRows() * col;
						T* vecPtr = other.data();
						CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyHostToDevice));
					} else if(!device() && other.device()){
						T* colPtr = data() + mRows() * col;
						T* vecPtr = other.dData();
						CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToHost));
					} else if(device() && other.device()){
						T* colPtr = dData() + mRows() * col;
						T* vecPtr = other.dData();
						CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToDevice));
					}

					return MATRIX_SUCCESS;
				}


	template<class T>
				int LaspMatrix<T>::deviceSetCol(int col, LaspMatrix<T>& other, int otherCol){

					context().setupMemTransfer(&other, this);

					if(device() && !other.device()){
						T* colPtr = dData() + mRows() * col;
						T* vecPtr = other.data() + other.mRows() * otherCol;
						CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyHostToDevice));
					} else if(!device() && other.device()){
						T* colPtr = data() + mRows() * col;
						T* vecPtr = other.dData() + other.mRows() * otherCol;
						CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToHost));
					} else if(device() && other.device()){
						T* colPtr = dData() + mRows() * col;
						T* vecPtr = other.dData() + other.mRows() * otherCol;
						CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToDevice));
					}

					return MATRIX_SUCCESS;
				}

	template<class T>
				int LaspMatrix<T>::deviceCopy(LaspMatrix<T> &other){
					if(!device() && !other.device()){
						return CANNOT_COMPLETE_OPERATION;
					}

					context().setupMemTransfer(&other, this);

					if(device() && !other.device()){
						T* rowPtr = dData();
						T* vecPtr = other.data();
						CUBLAS_CHECK(cublasSetMatrix(rows(), cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(!device() && other.device()){
						T* rowPtr = data();
						T* vecPtr = other.dData();
						CUBLAS_CHECK(cublasGetMatrix(rows(), cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(device() && other.device()){
						T* rowPtr = dData();
						T* vecPtr = other.dData();
						DeviceParams params = context().setupOperation(&other, this);
						return device_ewiseOp(params, vecPtr, rowPtr, other.size(), 0, 1, 1, other.rows(), other.mRows(), mRows());
					}

					return MATRIX_SUCCESS;
				}

				template<>
				inline int LaspMatrix<float>::deviceCopy(LaspMatrix<float> &other){
					if(!device() && !other.device()){
						return CANNOT_COMPLETE_OPERATION;
					}

					context().setupMemTransfer(&other, this);

					if(device() && !other.device()){
						float* rowPtr = dData();
						float* vecPtr = other.data();
						CUBLAS_CHECK(cublasSetMatrix(rows(), cols(), sizeof(float), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(!device() && other.device()){
						float* rowPtr = data();
						float* vecPtr = other.dData();
						CUBLAS_CHECK(cublasGetMatrix(rows(), cols(), sizeof(float), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(device() && other.device()){
						float* rowPtr = dData();
						float* vecPtr = other.dData();
						float alpha = 1, beta = 0;
						DeviceParams params = context().setupOperation(&other, this);
						CUBLAS_CHECK(cublasSgeam(params.context.getCuBlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, rows(), cols(), &alpha, other.dData(), other.mRows(), &beta, other.dData(), other.mRows(), dData(), mRows()));
					}

					return MATRIX_SUCCESS;
				}

				template<>
				inline int LaspMatrix<double>::deviceCopy(LaspMatrix<double> &other){
					if(!device() && !other.device()){
						return CANNOT_COMPLETE_OPERATION;
					}

					context().setupMemTransfer(&other, this);

					if(device() && !other.device()){
						double* rowPtr = dData();
						double* vecPtr = other.data();
						CUBLAS_CHECK(cublasSetMatrix(rows(), cols(), sizeof(double), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(!device() && other.device()){
						double* rowPtr = data();
						double* vecPtr = other.dData();
						CUBLAS_CHECK(cublasGetMatrix(rows(), cols(), sizeof(double), vecPtr, other.mRows(), rowPtr, mRows()));
					} else if(device() && other.device()){
						double* rowPtr = dData();
						double* vecPtr = other.dData();
						double alpha = 1, beta = 0;
						DeviceParams params = context().setupOperation(&other, this);
						CUBLAS_CHECK(cublasDgeam(params.context.getCuBlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, rows(), cols(), &alpha, other.dData(), other.mRows(), &beta, other.dData(), other.mRows(), dData(), mRows()));
					}

					return MATRIX_SUCCESS;
				}

	template<class T>
				 LaspMatrix<T> LaspMatrix<T>::deviceCopy(){
					if(!device()){
						throw CANNOT_COMPLETE_OPERATION;
					}

					LaspMatrix<T> result;

					result._rc() = 1;
					result._subrc() = 1;
					result._rowOffset() = 0;
					result._colOffset() = 0;
					result._rowEnd() = 0;
					result._colEnd() = 0;
					result._rows() = rows();
					result._cols() = cols();
					result._mRows() = mRows();
					result._mCols() = mCols();
					result._data() = 0;
					result._device() = true;
					result._key() = context().getNextKey();

					context().setupMemTransfer(this, &result);
					CUDA_CHECK_THROW(cudaMalloc((void**)result.dData_, mSize() * sizeof(T)));
					CUDA_CHECK_THROW(cudaMemcpy((void*)result._dData(), (void*)_dData(), mSize() * sizeof(T), cudaMemcpyDeviceToDevice));

					return result;
				}

	template<class T>
				void LaspMatrix<T>::freeData(){
					if (!device() && _data() != 0){
						delete [] _data();
					} else if (dData() != 0){
						if(_registered()){
							CUDA_CHECK_THROW(cudaHostUnregister(_data()));
							delete [] _data();
						} else {
							CUDA_CHECK_THROW(cudaFree((void*)_dData()));
						}
					}
				}

#endif

			}

#endif
