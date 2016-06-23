#pragma once

#include "boost_helpers.h"

#ifdef USE_BOOST

#include <boost/math/special_functions.hpp>
#include <boost/type_traits.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/version.hpp>

#else

#include <memory> //for std::shared_ptr

namespace boost
{
    template<typename T>
    class shared_array : public std::shared_ptr<T>  //there is no shared_array in C++11
    {
    public:
        shared_array() : std::shared_ptr<T>() {}
        shared_array(T * p) : std::shared_ptr<T>(p, std::default_delete<T[]>()) {}
        T & operator[](int i) { return this->get()[i]; }
        const T & operator[](int i) const { return this->get()[i]; }
    };

    namespace math
    {
        using namespace std;  //isnan, isinf
    }
}

#endif
