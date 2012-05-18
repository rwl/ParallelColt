/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.math.tfloat;

/**
 * Only for performance tuning of compute intensive linear algebraic
 * computations. Constructs functions that return one of
 * <ul>
 * <li><tt>a*constant + b</tt>
 * <li><tt>a*constant - b</tt>
 * <li><tt>a/constant + b</tt>
 * <li><tt>a/constant - b</tt>
 * </ul>
 * <tt>a</tt> and <tt>b</tt> are variables, <tt>constant</tt> is fixed, but for
 * performance reasons publicly accessible. Intended to be passed to
 * <tt>matrix.assign(otherMatrix,function)</tt> methods.
 */
public final class FloatPlusMultFirst implements cern.colt.function.tfloat.FloatFloatFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public float multiplicator;

    /**
     * Insert the method's description here. Creation date: (8/10/99 19:12:09)
     */
    protected FloatPlusMultFirst(final float multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final float apply(float a, float b) {
        return a * multiplicator + b;
    }

    /**
     * <tt>a/constant - b</tt>.
     */
    public static FloatPlusMultFirst minusDiv(final float constant) {
        return new FloatPlusMultFirst(-1 / constant);
    }

    /**
     * <tt>a*constant - b</tt>.
     */
    public static FloatPlusMultFirst minusMult(final float constant) {
        return new FloatPlusMultFirst(-constant);
    }

    /**
     * <tt>a/constant + b</tt>.
     */
    public static FloatPlusMultFirst plusDiv(final float constant) {
        return new FloatPlusMultFirst(1 / constant);
    }

    /**
     * <tt>a*constant + b</tt>.
     */
    public static FloatPlusMultFirst plusMult(final float constant) {
        return new FloatPlusMultFirst(constant);
    }
}
