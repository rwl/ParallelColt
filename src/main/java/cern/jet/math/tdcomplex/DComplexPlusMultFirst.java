package cern.jet.math.tdcomplex;

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

public class DComplexPlusMultFirst implements cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public double[] multiplicator;

    /**
     * Insert the method's description here. Creation date: (8/10/99 19:12:09)
     */
    protected DComplexPlusMultFirst(final double[] multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final double[] apply(double[] a, double[] b) {
        double[] z = new double[2];
        z[0] = a[0] * multiplicator[0] - a[1] * multiplicator[1];
        z[1] = a[1] * multiplicator[0] + a[0] * multiplicator[1];
        z[0] += b[0];
        z[1] += b[1];
        return z;
    }

    /**
     * <tt>a - b/constant</tt>.
     */
    public static DComplexPlusMultFirst minusDiv(final double[] constant) {
        return new DComplexPlusMultFirst(DComplex.neg(DComplex.inv(constant)));
    }

    /**
     * <tt>a - b*constant</tt>.
     */
    public static DComplexPlusMultFirst minusMult(final double[] constant) {
        return new DComplexPlusMultFirst(DComplex.neg(constant));
    }

    /**
     * <tt>a + b/constant</tt>.
     */
    public static DComplexPlusMultFirst plusDiv(final double[] constant) {
        return new DComplexPlusMultFirst(DComplex.inv(constant));
    }

    /**
     * <tt>a + b*constant</tt>.
     */
    public static DComplexPlusMultFirst plusMult(final double[] constant) {
        return new DComplexPlusMultFirst(constant);
    }
}
