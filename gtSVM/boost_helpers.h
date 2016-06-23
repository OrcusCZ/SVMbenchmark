#pragma once

#ifdef USE_BOOST

#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>

#else

#include <cassert>
#if __GNUC__ && __cplusplus <= 199711L
#include <stdint.h>
namespace std
{
    typedef int8_t int8_t;
    typedef uint8_t uint8_t;
    typedef int16_t int16_t;
    typedef uint16_t uint16_t;
    typedef int32_t int32_t;
    typedef uint32_t uint32_t;
    typedef int64_t int64_t;
    typedef uint64_t uint64_t;
}
#define BOOST_STATIC_ASSERT(x)
#else
#include <cstdint>
#define BOOST_STATIC_ASSERT(x) static_assert(x, "")
#endif

#define BOOST_ASSERT(x) assert(x)

namespace boost
{
    using namespace std;  //for types from cstdint
}

#endif
