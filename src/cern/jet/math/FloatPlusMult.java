/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.math;

/**
 * Only for performance tuning of compute intensive linear algebraic
 * computations. Constructs functions that return one of
 * <ul>
 * <li><tt>a + b*constant</tt>
 * <li><tt>a - b*constant</tt>
 * <li><tt>a + b/constant</tt>
 * <li><tt>a - b/constant</tt>
 * </ul>
 * <tt>a</tt> and <tt>b</tt> are variables, <tt>constant</tt> is fixed,
 * but for performance reasons publicly accessible. Intended to be passed to
 * <tt>matrix.assign(otherMatrix,function)</tt> methods.
 */
public final class FloatPlusMult implements cern.colt.function.FloatFloatFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public float multiplicator;

    /**
     * Insert the method's description here. Creation date: (8/10/99 19:12:09)
     */
    protected FloatPlusMult(final float multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final float apply(float a, float b) {
        return a + b * multiplicator;
    }

    /**
     * <tt>a - b/constant</tt>.
     */
    public static FloatPlusMult minusDiv(final float constant) {
        return new FloatPlusMult(-1 / constant);
    }

    /**
     * <tt>a - b*constant</tt>.
     */
    public static FloatPlusMult minusMult(final float constant) {
        return new FloatPlusMult(-constant);
    }

    /**
     * <tt>a + b/constant</tt>.
     */
    public static FloatPlusMult plusDiv(final float constant) {
        return new FloatPlusMult(1 / constant);
    }

    /**
     * <tt>a + b*constant</tt>.
     */
    public static FloatPlusMult plusMult(final float constant) {
        return new FloatPlusMult(constant);
    }
}
