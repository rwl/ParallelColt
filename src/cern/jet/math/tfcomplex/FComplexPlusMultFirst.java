package cern.jet.math.tfcomplex;

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

public class FComplexPlusMultFirst implements cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction {
    /**
     * Public read/write access to avoid frequent object construction.
     */
    public float[] multiplicator;

    /**
     * Insert the method's description here. Creation date: (8/10/99 19:12:09)
     */
    protected FComplexPlusMultFirst(final float[] multiplicator) {
        this.multiplicator = multiplicator;
    }

    /**
     * Returns the result of the function evaluation.
     */
    public final float[] apply(float[] a, float[] b) {
        float[] z = new float[2];
        z[0] = a[0] * multiplicator[0] - a[1] * multiplicator[1];
        z[1] = a[1] * multiplicator[0] + a[0] * multiplicator[1];
        z[0] += b[0];
        z[1] += b[1];
        return z;
    }

    /**
     * <tt>a - b/constant</tt>.
     */
    public static FComplexPlusMultFirst minusDiv(final float[] constant) {
        return new FComplexPlusMultFirst(FComplex.neg(FComplex.inv(constant)));
    }

    /**
     * <tt>a - b*constant</tt>.
     */
    public static FComplexPlusMultFirst minusMult(final float[] constant) {
        return new FComplexPlusMultFirst(FComplex.neg(constant));
    }

    /**
     * <tt>a + b/constant</tt>.
     */
    public static FComplexPlusMultFirst plusDiv(final float[] constant) {
        return new FComplexPlusMultFirst(FComplex.inv(constant));
    }

    /**
     * <tt>a + b*constant</tt>.
     */
    public static FComplexPlusMultFirst plusMult(final float[] constant) {
        return new FComplexPlusMultFirst(constant);
    }
}
