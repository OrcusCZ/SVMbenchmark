/*
	Copyright (C) 2011  Andrew Cotter

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


/**
	\file gtsvm.h
	\brief definition of C interface to SVM class
*/




#ifndef __GTSVM_H__
#define __GTSVM_H__




#ifdef __cplusplus
extern "C" {
#endif    /* __cplusplus */




#include <stdlib.h>




/*============================================================================
	GTSVM_Context typedef
============================================================================*/


typedef unsigned int GTSVM_Context;




/*============================================================================
	GTSVM_Type enumeration
============================================================================*/


typedef enum {

	GTSVM_TYPE_UNKNOWN = 0,

	GTSVM_TYPE_BOOL,

	GTSVM_TYPE_FLOAT,
	GTSVM_TYPE_DOUBLE,

	GTSVM_TYPE_INT8,
	GTSVM_TYPE_INT16,
	GTSVM_TYPE_INT32,
	GTSVM_TYPE_INT64,

	GTSVM_TYPE_UINT8,
	GTSVM_TYPE_UINT16,
	GTSVM_TYPE_UINT32,
	GTSVM_TYPE_UINT64

} GTSVM_Type;




/*============================================================================
	GTSVM_Kernel enumeration
============================================================================*/


typedef enum {

	GTSVM_KERNEL_UNKNOWN = 0,

	GTSVM_KERNEL_GAUSSIAN,      /* K( x, y ) = exp( -p1 * || x - y ||^2 ) */
	GTSVM_KERNEL_POLYNOMIAL,    /* K( x, y ) = ( p1 * <x,y> + p2 )^p3     */
	GTSVM_KERNEL_SIGMOID        /* K( x, y ) = tanh( p1 * <x,y> + p2 )    */

} GTSVM_Kernel;




/*============================================================================
	GTSVM_Error function
============================================================================*/


extern char const* GTSVM_Error();




/*============================================================================
	GTSVM_Create function
============================================================================*/


extern bool GTSVM_Create( GTSVM_Context* const pContext );




/*============================================================================
	GTSVM_Destroy function
============================================================================*/


extern bool GTSVM_Destroy( GTSVM_Context const context );




/*============================================================================
	GTSVM_InitializeSparse function
============================================================================*/


extern bool GTSVM_InitializeSparse(
	GTSVM_Context const context,
	void const* const trainingVectors,    /* order depends on the columnMajor flag */
	size_t const* const trainingVectorIndices,
	size_t const* const trainingVectorOffsets,
	GTSVM_Type const trainingVectorsType,
	void const* const trainingLabels,
	GTSVM_Type const trainingLabelsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor,
	bool const multiclass,
	float const regularization,
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3,
	bool const biased,
	bool const smallClusters,
	unsigned int const activeClusters
);




/*============================================================================
	GTSVM_InitializeDense function
============================================================================*/


extern bool GTSVM_InitializeDense(
	GTSVM_Context const context,
	void const* const trainingVectors,    /* order depends on the columnMajor flag */
	GTSVM_Type const trainingVectorsType,
	void const* const trainingLabels,
	GTSVM_Type const trainingLabelsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor,
	bool const multiclass,
	float const regularization,
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3,
	bool const biased,
	bool const smallClusters,
	unsigned int const activeClusters
);




/*============================================================================
	GTSVM_Load function
============================================================================*/


extern bool GTSVM_Load(
	GTSVM_Context const context,
	char const* const filename,
	bool const smallClusters,
	unsigned int const activeClusters
);




/*============================================================================
	GTSVM_Save function
============================================================================*/


extern bool GTSVM_Save(
	GTSVM_Context const context,
	char const* const filename
);




/*============================================================================
	GTSVM_Shrink function
============================================================================*/


extern bool GTSVM_Shrink(
	GTSVM_Context const context,
	bool const smallClusters,
	unsigned int const activeClusters
);




/*============================================================================
	GTSVM_DeinitializeDevice function
============================================================================*/


extern bool GTSVM_DeinitializeDevice( GTSVM_Context const context );




/*============================================================================
	GTSVM_Deinitialize function
============================================================================*/


extern bool GTSVM_Deinitialize( GTSVM_Context const context );




/*============================================================================
	GTSVM_GetRows function
============================================================================*/


extern bool GTSVM_GetRows(
	GTSVM_Context const context,
	unsigned int* const result
);




/*============================================================================
	GTSVM_GetColumns function
============================================================================*/


extern bool GTSVM_GetColumns(
	GTSVM_Context const context,
	unsigned int* const result
);




/*============================================================================
	GTSVM_GetClasses function
============================================================================*/


extern bool GTSVM_GetClasses(
	GTSVM_Context const context,
	unsigned int* const result
);




/*============================================================================
	GTSVM_GetNonzeros function
============================================================================*/


extern bool GTSVM_GetNonzeros(
	GTSVM_Context const context,
	unsigned int* const result
);




/*============================================================================
	GTSVM_GetRegularization function
============================================================================*/


extern bool GTSVM_GetRegularization(
	GTSVM_Context const context,
	float* const result
);




/*============================================================================
	GTSVM_GetKernel function
============================================================================*/


extern bool GTSVM_GetKernel(
	GTSVM_Context const context,
	GTSVM_Kernel* const result
);




/*============================================================================
	GTSVM_GetKernelParameter1 function
============================================================================*/


extern bool GTSVM_GetKernelParameter1(
	GTSVM_Context const context,
	float* const result
);




/*============================================================================
	GTSVM_GetKernelParameter2 function
============================================================================*/


extern bool GTSVM_GetKernelParameter2(
	GTSVM_Context const context,
	float* const result
);




/*============================================================================
	GTSVM_GetKernelParameter3 function
============================================================================*/


extern bool GTSVM_GetKernelParameter3(
	GTSVM_Context const context,
	float* const result
);




/*============================================================================
	GTSVM_GetBiased function
============================================================================*/


extern bool GTSVM_GetBiased(
	GTSVM_Context const context,
	bool* const result
);




/*============================================================================
	GTSVM_GetBias function
============================================================================*/


extern bool GTSVM_GetBias(
	GTSVM_Context const context,
	double* const result
);




/*============================================================================
	GTSVM_GetTrainingVectorsSparse function
============================================================================*/


extern bool GTSVM_GetTrainingVectorsSparse(
	GTSVM_Context const context,
	void* const trainingVectors,    /* order depends on the columnMajor flag */
	size_t* const trainingVectorIndices,
	size_t* const trainingVectorOffsets,
	GTSVM_Type const trainingVectorsType,
	bool const columnMajor
);




/*============================================================================
	GTSVM_GetTrainingVectorsDense function
============================================================================*/


extern bool GTSVM_GetTrainingVectorsDense(
	GTSVM_Context const context,
	void* const trainingVectors,    /* order depends on the columnMajor flag */
	GTSVM_Type const trainingVectorsType,
	bool const columnMajor
);




/*============================================================================
	GTSVM_GetTrainingLabels function
============================================================================*/


extern bool GTSVM_GetTrainingLabels(
	GTSVM_Context const context,
	void* const trainingLabels,
	GTSVM_Type const trainingLabelsType
);




/*============================================================================
	GTSVM_GetTrainingResponses function
============================================================================*/


extern bool GTSVM_GetTrainingResponses(
	GTSVM_Context const context,
	void* const trainingResponses,
	GTSVM_Type const trainingResponsesType,
	bool const columnMajor
);




/*============================================================================
	GTSVM_GetAlphas function
============================================================================*/


extern bool GTSVM_GetAlphas(
	GTSVM_Context const context,
	void* const trainingAlphas,
	GTSVM_Type const trainingAlphasType,
	bool const columnMajor
);




/*============================================================================
	GTSVM_SetAlphas function
============================================================================*/


extern bool GTSVM_SetAlphas(
	GTSVM_Context const context,
	void const* const trainingAlphas,
	GTSVM_Type const trainingAlphasType,
	bool const columnMajor
);




/*============================================================================
	GTSVM_Recalculate function
============================================================================*/


extern bool GTSVM_Recalculate( GTSVM_Context const context );




/*============================================================================
	GTSVM_Restart function
============================================================================*/


extern bool GTSVM_Restart(
	GTSVM_Context const context,
	float const regularization,
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3,
	bool const biased
);




/*============================================================================
	GTSVM_Optimize function
============================================================================*/


extern bool GTSVM_Optimize(
	GTSVM_Context const context,
	double* const pPrimal,
	double* const pDual,
	unsigned int const iterations
);




/*============================================================================
	GTSVM_ClassifySparse function
============================================================================*/


extern bool GTSVM_ClassifySparse(
	GTSVM_Context const context,
	void* const result,
	GTSVM_Type const resultType,
	void const* const vectors,    /* order depends on columnMajor flag */
	size_t const* const vectorIndices,
	size_t const* const vectorOffsets,
	GTSVM_Type const vectorsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor
);




/*============================================================================
	GTSVM_ClassifyDense function
============================================================================*/


extern bool GTSVM_ClassifyDense(
	GTSVM_Context const context,
	void* const result,
	GTSVM_Type const resultType,
	void const* const vectors,    /* order depends on columnMajor flag */
	GTSVM_Type const vectorsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor
);




#ifdef __cplusplus
}    /* extern "C" */
#endif    /* __cplusplus */




#endif    /* __GTSVM_H__ */
