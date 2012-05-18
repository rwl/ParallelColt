package cern.colt.function.tfloat;

/*
 Copyright (C) 1999 CERN - European Organization for Nuclear Research.
 Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
 is hereby granted without fee, provided that the above copyright notice appear in all copies and 
 that both that copyright notice and this permission notice appear in supporting documentation. 
 CERN makes no representations about the suitability of this software for any purpose. 
 It is provided "as is" without expressed or implied warranty.
 */
/**
 * Interface that represents a function object: a function that takes 9
 * arguments and returns a single value.
 */
public interface Float9Function {
    /**
     * Applies a function to nine arguments.
     * 
     * @return the result of the function.
     */
    abstract public float apply(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21,
            float a22);
}
