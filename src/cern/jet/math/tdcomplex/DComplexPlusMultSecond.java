package cern.jet.math.tdcomplex;

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

public class DComplexPlusMultSecond implements cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public double[] multiplicator;

    /**
     * Insert the method's description here. Creation date: (8/10/99 19:12:09)
     */
    protected DComplexPlusMultSecond(final double[] multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final double[] apply(double[] a, double[] b) {
        double[] z = new double[2];
        z[0] = b[0] * multiplicator[0] - b[1] * multiplicator[1];
        z[1] = b[1] * multiplicator[0] + b[0] * multiplicator[1];
        z[0] += a[0];
        z[1] += a[1];
        return z;
    }

    /**
     * <tt>a - b/constant</tt>.
     */
    public static DComplexPlusMultSecond minusDiv(final double[] constant) {
        return new DComplexPlusMultSecond(DComplex.neg(DComplex.inv(constant)));
    }

    /**
     * <tt>a - b*constant</tt>.
     */
    public static DComplexPlusMultSecond minusMult(final double[] constant) {
        return new DComplexPlusMultSecond(DComplex.neg(constant));
    }

    /**
     * <tt>a + b/constant</tt>.
     */
    public static DComplexPlusMultSecond plusDiv(final double[] constant) {
        return new DComplexPlusMultSecond(DComplex.inv(constant));
    }

    /**
     * <tt>a + b*constant</tt>.
     */
    public static DComplexPlusMultSecond plusMult(final double[] constant) {
        return new DComplexPlusMultSecond(constant);
    }
}
