/*
 * utilities.cpp
 *
 *  Created on: Jul 23, 2009
 *      Author: sherrero
 */


#include "../../include/utilities.h"


/**
 * Determines the next number power of 2 greater than the argument
 * @param x the number to be checked
 */
unsigned int nextPow2( unsigned int x )
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/**
 * Determines whether the argument is power of 2
 * @param x the number to be checked
 */
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


