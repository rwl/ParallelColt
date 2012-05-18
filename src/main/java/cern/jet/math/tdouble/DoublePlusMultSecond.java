/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.math.tdouble;

/**
 * Only for performance tuning of compute intensive linear algebraic
 * computations. Constructs functions that return one of
 * <ul>
 * <li><tt>a + b*constant</tt>
 * <li><tt>a - b*constant</tt>
 * <li><tt>a + b/constant</tt>
 * <li><tt>a - b/constant</tt>
 * </ul>
 * <tt>a</tt> and <tt>b</tt> are variables, <tt>constant</tt> is fixed, but for
 * performance reasons publicly accessible. Intended to be passed to
 * <tt>matrix.assign(otherMatrix,function)</tt> methods.
 */
public final class DoublePlusMultSecond implements cern.colt.function.tdouble.DoubleDoubleFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public double multiplicator;

    /**
     * Insert the method's description here. Creation date: (8/10/99 19:12:09)
     */
    protected DoublePlusMultSecond(final double multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final double apply(double a, double b) {
        return a + b * multiplicator;
    }

    /**
     * <tt>a - b/constant</tt>.
     */
    public static DoublePlusMultSecond minusDiv(final double constant) {
        return new DoublePlusMultSecond(-1 / constant);
    }

    /**
     * <tt>a - b*constant</tt>.
     */
    public static DoublePlusMultSecond minusMult(final double constant) {
        return new DoublePlusMultSecond(-constant);
    }

    /**
     * <tt>a + b/constant</tt>.
     */
    public static DoublePlusMultSecond plusDiv(final double constant) {
        return new DoublePlusMultSecond(1 / constant);
    }

    /**
     * <tt>a + b*constant</tt>.
     */
    public static DoublePlusMultSecond plusMult(final double constant) {
        return new DoublePlusMultSecond(constant);
    }
}
