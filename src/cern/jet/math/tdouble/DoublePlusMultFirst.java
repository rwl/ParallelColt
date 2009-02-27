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
 * <li><tt>a*constant + b</tt>
 * <li><tt>a*constant - b</tt>
 * <li><tt>a/constant + b</tt>
 * <li><tt>a/constant - b</tt>
 * </ul>
 * <tt>a</tt> and <tt>b</tt> are variables, <tt>constant</tt> is fixed, but for
 * performance reasons publicly accessible. Intended to be passed to
 * <tt>matrix.assign(otherMatrix,function)</tt> methods.
 */
public final class DoublePlusMultFirst implements cern.colt.function.tdouble.DoubleDoubleFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public double multiplicator;

    protected DoublePlusMultFirst(final double multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final double apply(double a, double b) {
        return a * multiplicator + b;
    }

    /**
     * <tt>a - b/constant</tt>.
     */
    public static DoublePlusMultFirst minusDiv(final double constant) {
        return new DoublePlusMultFirst(-1 / constant);
    }

    /**
     * <tt>a - b*constant</tt>.
     */
    public static DoublePlusMultFirst minusMult(final double constant) {
        return new DoublePlusMultFirst(-constant);
    }

    /**
     * <tt>a + b/constant</tt>.
     */
    public static DoublePlusMultFirst plusDiv(final double constant) {
        return new DoublePlusMultFirst(1 / constant);
    }

    /**
     * <tt>a + b*constant</tt>.
     */
    public static DoublePlusMultFirst plusMult(final double constant) {
        return new DoublePlusMultFirst(constant);
    }
}
