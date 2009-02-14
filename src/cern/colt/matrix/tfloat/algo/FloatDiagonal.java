/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.algo;

import cern.colt.matrix.tfloat.FloatMatrix2D;

/**
 * For diagonal matrices we can often do better.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
class FloatDiagonal {
    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected FloatDiagonal() {
        throw new RuntimeException("Non instantiable");
    }

    /**
     * Modifies A to hold its inverse.
     * 
     * @param x
     *            the first vector.
     * @param y
     *            the second vector.
     * @return isNonSingular.
     * @throws IllegalArgumentException
     *             if <tt>x.size() != y.size()</tt>.
     */
    public static boolean inverse(FloatMatrix2D A) {
        FloatProperty.DEFAULT.checkSquare(A);
        boolean isNonSingular = true;
        for (int i = A.rows(); --i >= 0;) {
            float v = A.getQuick(i, i);
            isNonSingular &= (v != 0);
            A.setQuick(i, i, 1 / v);
        }
        return isNonSingular;
    }
}
