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
 * Arithmetic functions.
 */
public class FloatArithmetic extends FloatConstants {
    // for method stirlingCorrection(...)
    private static final float[] stirlingCorrection = { 0.0f, 8.1061467e-02f, 4.1340696e-02f, 2.7677926e-02f,
            2.0790672e-02f, 1.6644691e-02f, 1.3876129e-02f, 1.1896710e-02f, 1.0411265e-02f, 9.2554622e-03f,
            8.3305634e-03f, 7.5736755e-03f, 6.9428401e-03f, 6.4089942e-03f, 5.9513701e-03f, 5.5547336e-03f,
            5.2076559e-03f, 4.9013959e-03f, 4.6291537e-03f, 4.3855602e-03f, 4.1663197e-03f, 3.9679542e-03f,
            3.7876181e-03f, 3.6229602e-03f, 3.4720214e-03f, 3.3331556e-03f, 3.2049702e-03f, 3.0862787e-03f,
            2.9760640e-03f, 2.8734494e-03f, 2.7776749e-03f, };

    // for method logFactorial(...)
    // log(k!) for k = 0, ..., 29
    protected static final float[] logFactorials = { 0.0000000f, 0.0000000f, 0.69314718f, 1.7917595f, 3.1780538f,
            4.7874917f, 6.5792512f, 8.5251613f, 10.6046029f, 12.8018275f, 15.1044126f, 17.5023078f, 19.9872145f,
            22.5521639f, 25.1912212f, 27.8992714f, 30.6718601f, 33.5050735f, 36.3954452f, 39.3398842f, 42.3356165f,
            45.3801389f, 48.4711814f, 51.6066756f, 54.7847294f, 58.0036052f, 61.2617018f, 64.5575386f, 67.8897431f,
            71.2570390f };

    // k! for k = 0, ..., 20
    protected static final long[] longFactorials = { 1L, 1L, 2L, 6L, 24L, 120L, 720L, 5040L, 40320L, 362880L, 3628800L,
            39916800L, 479001600L, 6227020800L, 87178291200L, 1307674368000L, 20922789888000L, 355687428096000L,
            6402373705728000L, 121645100408832000L, 2432902008176640000L };

    // k! for k = 21, ..., 34
    protected static final float[] floatFactorials = { 5.1090942E19f, 1.1240007E21f, 2.5852017E22f, 6.2044840E23f,
            1.5511210E25f, 4.0329146E26f, 1.0888869E28f, 3.0488834E29f, 8.8417620E30f, 2.6525286E32f, 8.2228387E33f,
            2.6313084E35f, 8.6833176E36f, 2.9523280E38f };

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected FloatArithmetic() {
    }

    /**
     * Efficiently returns the binomial coefficient, often also referred to as
     * "n over k" or "n choose k". The binomial coefficient is defined as
     * <tt>(n * n-1 * ... * n-k+1 ) / ( 1 * 2 * ... * k )</tt>.
     * <ul>
     * <li>k<0<tt>: <tt>0</tt>.
     * <li>k==0<tt>: <tt>1</tt>.
     * <li>k==1<tt>: <tt>n</tt>.
     * <li>else: <tt>(n * n-1 * ... * n-k+1 ) / ( 1 * 2 * ... * k )</tt>.
     * </ul>
     * 
     * @return the binomial coefficient.
     */
    public static float binomial(float n, long k) {
        if (k < 0)
            return 0;
        if (k == 0)
            return 1;
        if (k == 1)
            return n;

        // binomial(n,k) = (n * n-1 * ... * n-k+1 ) / ( 1 * 2 * ... * k )
        float a = n - k + 1;
        float b = 1;
        float binomial = 1;
        for (long i = k; i-- > 0;) {
            binomial *= (a++) / (b++);
        }
        return binomial;
    }

    /**
     * Efficiently returns the binomial coefficient, often also referred to as
     * "n over k" or "n choose k". The binomial coefficient is defined as
     * <ul>
     * <li>k<0<tt>: <tt>0</tt>.
     * <li>k==0 || k==n<tt>: <tt>1</tt>.
     * <li>k==1 || k==n-1<tt>: <tt>n</tt>.
     * <li>else: <tt>(n * n-1 * ... * n-k+1 ) / ( 1 * 2 * ... * k )</tt>.
     * </ul>
     * 
     * @return the binomial coefficient.
     */
    public static float binomial(long n, long k) {
        if (k < 0)
            return 0;
        if (k == 0 || k == n)
            return 1;
        if (k == 1 || k == n - 1)
            return n;

        // try quick version and see whether we get numeric overflows.
        // factorial(..) is O(1); requires no loop; only a table lookup.
        if (n > k) {
            int max = longFactorials.length + floatFactorials.length;
            if (n < max) { // if (n! < inf && k! < inf)
                float n_fac = factorial((int) n);
                float k_fac = factorial((int) k);
                float n_minus_k_fac = factorial((int) (n - k));
                float nk = n_minus_k_fac * k_fac;
                if (nk != Float.POSITIVE_INFINITY) { // no numeric overflow?
                    // now this is completely safe and accurate
                    return n_fac / nk;
                }
            }
            if (k > n / 2)
                k = n - k; // quicker
        }

        // binomial(n,k) = (n * n-1 * ... * n-k+1 ) / ( 1 * 2 * ... * k )
        long a = n - k + 1;
        long b = 1;
        float binomial = 1;
        for (long i = k; i-- > 0;) {
            binomial *= ((float) (a++)) / (b++);
        }
        return binomial;
    }

    /**
     * Returns the smallest <code>long &gt;= value</code>. <dt>Examples:
     * <code>1.0 -> 1, 1.2 -> 2, 1.9 -> 2</code>. This method is safer than
     * using (long) Math.ceil(value), because of possible rounding error.
     */
    public static long ceil(float value) {
        return Math.round(Math.ceil(value));
    }

    /**
     * Evaluates the series of Chebyshev polynomials Ti at argument x/2. The
     * series is given by
     * 
     * <pre>
     *        N-1
     *         - '
     *  y  =   &gt;   coef[i] T (x/2)
     *         -            i
     *        i=0
     * </pre>
     * 
     * Coefficients are stored in reverse order, i.e. the zero order term is
     * last in the array. Note N is the number of coefficients, not the order.
     * <p>
     * If coefficients are for the interval a to b, x must have been transformed
     * to x -> 2(2x - b - a)/(b-a) before entering the routine. This maps x from
     * (a, b) to (-1, 1), over which the Chebyshev polynomials are defined.
     * <p>
     * If the coefficients are for the inverted interval, in which (a, b) is
     * mapped to (1/b, 1/a), the transformation required is x -> 2(2ab/x - b -
     * a)/(b-a). If b is infinity, this becomes x -> 4a/x - 1.
     * <p>
     * SPEED:
     * <p>
     * Taking advantage of the recurrence properties of the Chebyshev
     * polynomials, the routine requires one more addition per loop than
     * evaluating a nested polynomial of the same degree.
     * 
     * @param x
     *            argument to the polynomial.
     * @param coef
     *            the coefficients of the polynomial.
     * @param N
     *            the number of coefficients.
     */
    public static float chbevl(float x, float coef[], int N) throws ArithmeticException {
        float b0, b1, b2;

        int p = 0;
        int i;

        b0 = coef[p++];
        b1 = 0.0f;
        i = N - 1;

        do {
            b2 = b1;
            b1 = b0;
            b0 = x * b1 - b2 + coef[p++];
        } while (--i > 0);

        return (0.5f * (b0 - b2));
    }

    /**
     * Returns the factorial of the argument.
     */
    static private long fac1(int j) {
        long i = j;
        if (j < 0)
            i = Math.abs(j);
        if (i > longFactorials.length)
            throw new IllegalArgumentException("Overflow");

        long d = 1;
        while (i > 1)
            d *= i--;

        if (j < 0)
            return -d;
        else
            return d;
    }

    /**
     * Returns the factorial of the argument.
     */
    static private float fac2(int j) {
        long i = j;
        if (j < 0)
            i = Math.abs(j);

        float d = 1.0f;
        while (i > 1)
            d *= i--;

        if (j < 0)
            return -d;
        else
            return d;
    }

    /**
     * Instantly returns the factorial <tt>k!</tt>.
     * 
     * @param k
     *            must hold <tt>k &gt;= 0</tt>.
     */
    static public float factorial(int k) {
        if (k < 0)
            throw new IllegalArgumentException();

        int length1 = longFactorials.length;
        if (k < length1)
            return longFactorials[k];

        int length2 = floatFactorials.length;
        if (k < length1 + length2)
            return floatFactorials[k - length1];
        else
            return Float.POSITIVE_INFINITY;
    }

    /**
     * Returns the largest <code>long &lt;= value</code>. <dt>Examples: <code>
     * 1.0 -> 1, 1.2 -> 1, 1.9 -> 1 <dt>
     * 2.0 -> 2, 2.2 -> 2, 2.9 -> 2 </code> <dt>This method is safer than using
     * (long) Math.floor(value), because of possible rounding error.
     */
    public static long floor(float value) {
        return Math.round(Math.floor(value));
    }

    /**
     * Returns <tt>log<sub>base</sub>value</tt>.
     */
    public static float log(float base, float value) {
        return (float) (Math.log(value) / Math.log(base));
    }

    /**
     * Returns <tt>log<sub>10</sub>value</tt>.
     */
    static public float log10(float value) {
        // 1.0 / Math.log(10) == 0.43429448190325176
        return (float) (Math.log(value) * 0.43429448190325176);
    }

    /**
     * Returns <tt>log<sub>2</sub>value</tt>.
     */
    static public float log2(float value) {
        // 1.0 / Math.log(2) == 1.4426950408889634
        return (float) (Math.log(value) * 1.4426950408889634);
    }

    /**
     * Returns <tt>log(k!)</tt>. Tries to avoid overflows. For <tt>k<30</tt>
     * simply looks up a table in O(1). For <tt>k>=30</tt> uses stirlings
     * approximation.
     * 
     * @param k
     *            must hold <tt>k &gt;= 0</tt>.
     */
    public static float logFactorial(int k) {
        if (k >= 30) {
            float r, rr;
            final float C0 = 9.1893853e-01f;
            final float C1 = 8.3333333e-02f;
            final float C3 = -2.7777778e-03f;
            final float C5 = 7.9365079e-04f;
            final float C7 = -5.9523810e-04f;

            r = 1.0f / k;
            rr = r * r;
            return (float) ((k + 0.5) * Math.log(k) - k + C0 + r * (C1 + rr * (C3 + rr * (C5 + rr * C7))));
        } else
            return logFactorials[k];
    }

    /**
     * Instantly returns the factorial <tt>k!</tt>.
     * 
     * @param k
     *            must hold <tt>k &gt;= 0 && k &lt; 21</tt>.
     */
    static public long longFactorial(int k) throws IllegalArgumentException {
        if (k < 0)
            throw new IllegalArgumentException("Negative k");

        if (k < longFactorials.length)
            return longFactorials[k];
        throw new IllegalArgumentException("Overflow");
    }

    /**
     * Returns the StirlingCorrection.
     * <p>
     * Correction term of the Stirling approximation for <tt>log(k!)</tt>
     * (series in 1/k, or table values for small k) with int parameter k.
     * <p>
     * <tt>
     * log k! = (k + 1/2)log(k + 1) - (k + 1) + (1/2)log(2Pi) +
     *          stirlingCorrection(k + 1)                                    
     * <p>                                                                      
     * log k! = (k + 1/2)log(k)     -  k      + (1/2)log(2Pi) +              
     *          stirlingCorrection(k)
     * </tt>
     */
    public static float stirlingCorrection(int k) {
        final float C1 = 8.3333333e-02f; // +1/12
        final float C3 = -2.7777778e-03f; // -1/360
        final float C5 = 7.9365079e-04f; // +1/1260
        final float C7 = -5.9523810e-04f; // -1/1680

        float r, rr;

        if (k > 30) {
            r = 1.0f / k;
            rr = r * r;
            return r * (C1 + rr * (C3 + rr * (C5 + rr * C7)));
        } else
            return stirlingCorrection[k];
    }

    /**
     * Equivalent to <tt>Math.round(binomial(n,k))</tt>.
     */
    private static long xlongBinomial(long n, long k) {
        return Math.round(binomial(n, k));
    }
}
