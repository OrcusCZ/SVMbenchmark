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
	\file gtsvm.cpp
	\brief implementation of C interface to SVM class
*/




#include "headers.hpp"




namespace {




//============================================================================
//    global variables
//============================================================================


typedef std::map< GTSVM_Context, boost::shared_ptr< GTSVM::SVM > > ContextMap;

ContextMap g_contextMap;
GTSVM_Context g_nextContext = 0;

bool g_error = false;
std::string g_errorString;




//============================================================================
//    SAVE_EXCEPTIONS macros
//============================================================================


template< typename t_Exception >
inline void CatchSaveExceptionsFunction( t_Exception const& error ) {

	g_error = true;
	g_errorString = error.What();
}


#define TRY_SAVE_EXCEPTIONS  \
	try {


#define CATCH_SAVE_EXCEPTIONS  \
	}  \
	catch( std::exception& error ) {  \
		g_error = true;  \
		g_errorString = error.what();  \
	}  \
	catch( ... ) {  \
		g_error = true;  \
		g_errorString = "Unknown error";  \
	}




}    // anonymous namespace




//============================================================================
//    GTSVM_Error function
//============================================================================


extern "C" char const* GTSVM_Error() {

	char const* result = "No error";
	if ( g_error )
		result = g_errorString.c_str();
	return result;
}




//============================================================================
//    GTSVM_Create function
//============================================================================


extern "C" bool GTSVM_Create( GTSVM_Context* const pContext ) {

	g_error = false;

	TRY_SAVE_EXCEPTIONS

		if ( g_nextContext + 1 == 0 )
			throw std::runtime_error( "Too many contexts created" );

		*pContext = g_nextContext;
		g_contextMap.insert( std::pair< GTSVM_Context, boost::shared_ptr< GTSVM::SVM > >( g_nextContext, boost::shared_ptr< GTSVM::SVM >( new GTSVM::SVM ) ) );
		++g_nextContext;

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Destroy function
//============================================================================


extern "C" bool GTSVM_Destroy( GTSVM_Context const context ) {

	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );
		g_contextMap.erase( pContext );

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_InitializeSparse function
//============================================================================


extern "C" bool GTSVM_InitializeSparse(
	GTSVM_Context const context,
	void const* const trainingVectors,    // order depends on the columnMajor flag
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
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->InitializeSparse(
			trainingVectors,
			trainingVectorIndices,
			trainingVectorOffsets,
			trainingVectorsType,
			trainingLabels,
			trainingLabelsType,
			rows,
			columns,
			columnMajor,
			multiclass,
			regularization,
			kernel,
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_InitializeDense function
//============================================================================


extern "C" bool GTSVM_InitializeDense(
	GTSVM_Context const context,
	void const* const trainingVectors,    // order depends on the columnMajor flag
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
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->InitializeDense(
			trainingVectors,
			trainingVectorsType,
			trainingLabels,
			trainingLabelsType,
			rows,
			columns,
			columnMajor,
			multiclass,
			regularization,
			kernel,
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Load function
//============================================================================


extern "C" bool GTSVM_Load(
	GTSVM_Context const context,
	char const* const filename,
	bool const smallClusters,
	unsigned int const activeClusters
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->Load(
			filename,
			smallClusters,
			activeClusters
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Save function
//============================================================================


extern "C" bool GTSVM_Save(
	GTSVM_Context const context,
	char const* const filename
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->Save( filename );

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Shrink function
//============================================================================


extern "C" bool GTSVM_Shrink(
	GTSVM_Context const context,
	bool const smallClusters,
	unsigned int const activeClusters
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->Shrink(
			smallClusters,
			activeClusters
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_DeinitializeDevice function
//============================================================================


extern "C" bool GTSVM_DeinitializeDevice( GTSVM_Context const context ) {

	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->DeinitializeDevice();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Deinitialize function
//============================================================================


extern "C" bool GTSVM_Deinitialize( GTSVM_Context const context ) {

	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->Deinitialize();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetRows function
//============================================================================


extern "C" bool GTSVM_GetRows(
	GTSVM_Context const context,
	unsigned int* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetRows();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetColumns function
//============================================================================


extern "C" bool GTSVM_GetColumns(
	GTSVM_Context const context,
	unsigned int* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetColumns();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetClasses function
//============================================================================


extern "C" bool GTSVM_GetClasses(
	GTSVM_Context const context,
	unsigned int* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetClasses();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetNonzeros function
//============================================================================


extern "C" bool GTSVM_GetNonzeros(
	GTSVM_Context const context,
	unsigned int* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetNonzeros();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetRegularization function
//============================================================================


extern "C" bool GTSVM_GetRegularization(
	GTSVM_Context const context,
	float* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetRegularization();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetKernel function
//============================================================================


extern "C" bool GTSVM_GetKernel (
	GTSVM_Context const context,
	GTSVM_Kernel* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetKernel();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetKernelParameter1 function
//============================================================================


extern "C" bool GTSVM_GetKernelParameter1 (
	GTSVM_Context const context,
	float* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetKernelParameter1();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetKernelParameter2 function
//============================================================================


extern "C" bool GTSVM_GetKernelParameter2 (
	GTSVM_Context const context,
	float* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetKernelParameter2();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetKernelParameter3 function
//============================================================================


extern "C" bool GTSVM_GetKernelParameter3 (
	GTSVM_Context const context,
	float* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetKernelParameter3();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetBiased function
//============================================================================


extern "C" bool GTSVM_GetBiased(
	GTSVM_Context const context,
	bool* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetBiased();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetBias function
//============================================================================


extern "C" bool GTSVM_GetBias(
	GTSVM_Context const context,
	double* const result
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		*result = pContext->second->GetBias();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetTrainingVectorsSparse function
//============================================================================


extern "C" bool GTSVM_GetTrainingVectorsSparse(
	GTSVM_Context const context,
	void* const trainingVectors,    // order depends on the columnMajor flag
	size_t* const trainingVectorIndices,
	size_t* const trainingVectorOffsets,
	GTSVM_Type const trainingVectorsType,
	bool const columnMajor
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->GetTrainingVectorsSparse(
			trainingVectors,
			trainingVectorIndices,
			trainingVectorOffsets,
			trainingVectorsType,
			columnMajor
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetTrainingVectorsDense function
//============================================================================


extern "C" bool GTSVM_GetTrainingVectorsDense(
	GTSVM_Context const context,
	void* const trainingVectors,    // order depends on the columnMajor flag
	GTSVM_Type const trainingVectorsType,
	bool const columnMajor
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->GetTrainingVectorsDense(
			trainingVectors,
			trainingVectorsType,
			columnMajor
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetTrainingLabels function
//============================================================================


extern "C" bool GTSVM_GetTrainingLabels(
	GTSVM_Context const context,
	void* const trainingLabels,
	GTSVM_Type const trainingLabelsType
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->GetTrainingLabels(
			trainingLabels,
			trainingLabelsType
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetTrainingResponses function
//============================================================================


extern "C" bool GTSVM_GetTrainingResponses(
	GTSVM_Context const context,
	void* const trainingResponses,
	GTSVM_Type const trainingResponsesType,
	bool const columnMajor
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->GetTrainingResponses(
			trainingResponses,
			trainingResponsesType,
			columnMajor
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_GetAlphas function
//============================================================================


extern "C" bool GTSVM_GetAlphas(
	GTSVM_Context const context,
	void* const trainingAlphas,
	GTSVM_Type const trainingAlphasType,
	bool const columnMajor
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->GetAlphas(
			trainingAlphas,
			trainingAlphasType,
			columnMajor
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_SetAlphas function
//============================================================================


extern "C" bool GTSVM_SetAlphas(
	GTSVM_Context const context,
	void const* const trainingAlphas,
	GTSVM_Type const trainingAlphasType,
	bool const columnMajor
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->SetAlphas(
			trainingAlphas,
			trainingAlphasType,
			columnMajor
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Recalculate function
//============================================================================


extern "C" bool GTSVM_Recalculate( GTSVM_Context const context ) {

	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->Recalculate();

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Restart function
//============================================================================


extern "C" bool GTSVM_Restart(
	GTSVM_Context const context,
	float const regularization,
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3,
	bool const biased
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->Restart(
			regularization,
			kernel,
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_Optimize function
//============================================================================


extern "C" bool GTSVM_Optimize(
	GTSVM_Context const context,
	double* const pPrimal,
	double* const pDual,
	unsigned int const iterations
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		std::pair< CUDA_FLOAT_DOUBLE, CUDA_FLOAT_DOUBLE > const result = pContext->second->Optimize( iterations );
		*pPrimal = result.first;
		*pDual   = result.second;

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_ClassifySparse function
//============================================================================


extern "C" bool GTSVM_ClassifySparse(
	GTSVM_Context const context,
	void* const result,
	GTSVM_Type const resultType,
	void const* const vectors,    // order depends on columnMajor flag
	size_t const* const vectorIndices,
	size_t const* const vectorOffsets,
	GTSVM_Type const vectorsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->ClassifySparse(
			result,
			resultType,
			vectors,
			vectorIndices,
			vectorOffsets,
			vectorsType,
			rows,
			columns,
			columnMajor
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}




//============================================================================
//    GTSVM_ClassifyDense function
//============================================================================


extern "C" bool GTSVM_ClassifyDense(
	GTSVM_Context const context,
	void* const result,
	GTSVM_Type const resultType,
	void const* const vectors,    // order depends on columnMajor flag
	GTSVM_Type const vectorsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor
)
{
	g_error = false;

	TRY_SAVE_EXCEPTIONS

		ContextMap::const_iterator pContext = g_contextMap.find( context );
		if ( pContext == g_contextMap.end() )
			throw std::runtime_error( "Context does not exist" );

		pContext->second->ClassifyDense(
			result,
			resultType,
			vectors,
			vectorsType,
			rows,
			columns,
			columnMajor
		);

	CATCH_SAVE_EXCEPTIONS

	return g_error;
}
