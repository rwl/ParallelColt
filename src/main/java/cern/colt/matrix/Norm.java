/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix;

/**
 * Types of matrix norms.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public enum Norm {

    /**
     * Maximum absolute row sum
     */
    One,

    /**
     * Euclidean norm
     */
    Two,

    /**
     * The root of sum of the sum of squares
     */
    Frobenius,

    /**
     * Maximum column sum
     */
    Infinity,

}