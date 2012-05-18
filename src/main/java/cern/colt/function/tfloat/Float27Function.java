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
 * Interface that represents a function object: a function that takes 27
 * arguments and returns a single value.
 */
public interface Float27Function {
    /**
     * Applies a function to 27 arguments.
     * 
     * @return the result of the function.
     */
    abstract public float apply(float a000, float a001, float a002, float a010, float a011, float a012, float a020,
            float a021, float a022,

            float a100, float a101, float a102, float a110, float a111, float a112, float a120, float a121, float a122,

            float a200, float a201, float a202, float a210, float a211, float a212, float a220, float a221, float a222);
}
