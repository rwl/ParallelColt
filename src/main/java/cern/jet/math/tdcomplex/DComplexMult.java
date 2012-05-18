/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.math.tdcomplex;

/**
 * Only for performance tuning of compute intensive linear algebraic
 * computations. Constructs functions that return one of
 * <ul>
 * <li><tt>a * constant</tt>
 * <li><tt>a / constant</tt>
 * </ul>
 * <tt>a</tt> is variable, <tt>constant</tt> is fixed, but for performance
 * reasons publicly accessible. Intended to be passed to
 * <tt>matrix.assign(function)</tt> methods.
 */
public final class DComplexMult implements cern.colt.function.tdcomplex.DComplexDComplexFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public double[] multiplicator;

    protected DComplexMult(final double[] multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final double[] apply(double[] a) {
        double[] z = new double[2];
        z[0] = a[0] * multiplicator[0] - a[1] * multiplicator[1];
        z[1] = a[1] * multiplicator[0] + a[0] * multiplicator[1];
        return z;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final double[] apply(double re, double im) {
        double[] z = new double[2];
        z[0] = re * multiplicator[0] - im * multiplicator[1];
        z[1] = im * multiplicator[0] + re * multiplicator[1];
        return z;
    }

    /**
     * <tt>a / constant</tt>.
     */
    public static DComplexMult div(final double[] constant) {
        return mult(DComplex.inv(constant));
    }

    /**
     * <tt>a * constant</tt>.
     */
    public static DComplexMult mult(final double[] constant) {
        return new DComplexMult(constant);
    }
}
