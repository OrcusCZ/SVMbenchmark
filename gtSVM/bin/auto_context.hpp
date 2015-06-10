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
	\file auto.hpp
	\brief implementation of AutoContext class
*/




#ifndef __AUTO_CONTEXT_HPP__
#define __AUTO_CONTEXT_HPP__

#ifdef __cplusplus




#include <gtsvm.h>

#include <stdexcept>




//============================================================================
//    AutoContext class
//============================================================================


struct AutoContext {

	inline AutoContext();

	inline ~AutoContext();


	inline operator GTSVM_Context const() const;


private:

	GTSVM_Context m_context;


	inline AutoContext( AutoContext const& other );
	inline AutoContext const& operator=( AutoContext const& other );
};




//============================================================================
//    AutoContext inline methods
//============================================================================


AutoContext::AutoContext() {

	if ( GTSVM_Create( &m_context ) )
		throw std::runtime_error( GTSVM_Error() );
}


AutoContext::~AutoContext() {

	if ( GTSVM_Destroy( m_context ) )
		throw std::runtime_error( GTSVM_Error() );    // this will actually abort, since it's inside a destructor
}


AutoContext::operator GTSVM_Context const() const {

	return m_context;
}




#endif    /* __cplusplus */

#endif    /* __AUTO_CONTEXT_HPP__ */
