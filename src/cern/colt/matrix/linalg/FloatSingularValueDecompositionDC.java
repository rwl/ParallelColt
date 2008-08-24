/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.linalg;

import org.netlib.lapack.Sgesdd;

import cern.colt.matrix.FloatMatrix2D;
import cern.colt.matrix.impl.DenseFloatMatrix2D;

/**
 * 
 * For an <tt>m x n</tt> matrix <tt>A</tt>, the singular value
 * decomposition is an <tt>m x m</tt> orthogonal matrix <tt>U</tt>, an
 * <tt>m x n</tt> diagonal matrix <tt>S</tt>, and an <tt>n x n</tt>
 * orthogonal matrix <tt>V</tt> so that <tt>A = U*S*V'</tt>.
 * <P>
 * The singular values, <tt>sigma[k] = S[k][k]</tt>, are ordered so that
 * <tt>sigma[0] >= sigma[1] >= ... >= sigma[min(m-1,n-1)]</tt>.
 * <P>
 * 
 * This implementation uses the divide-and-conquer algorithm (Dgesdd) from
 * JLAPACK.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class FloatSingularValueDecompositionDC {

    private float[] Ut;

    private float[] V;

    private float[] s;

    private org.netlib.util.intW info;

    private int m;

    private int n;

    private int minmn;

    private boolean wantWholeUV;

    private boolean wantUV;

    /**
     * Constructs and returns a new singular value decomposition object; The
     * decomposed matrices can be retrieved via instance methods of the returned
     * decomposition object.
     * 
     * @param Arg
     *            rectangular matrix
     * 
     * @param wantUV
     *            <br>
     *            true: all matrices (U, S, V') are computed<br>
     *            false: only S is computed
     * @param wantWholeUV
     *            <br>
     *            true: all m columns of U and all n rows of V' are computed<br>
     *            false: the first min(m,n) columns of U and the first min(m,n)
     *            rows of V' are computed
     */
    public FloatSingularValueDecompositionDC(FloatMatrix2D Arg, boolean wantUV, boolean wantWholeUV) {
        this.wantUV = wantUV;
        this.wantWholeUV = wantWholeUV;
        m = Arg.rows();
        n = Arg.columns();
        float[] elems = (float[]) Arg.viewDice().copy().elements();
        minmn = Math.min(m, n);
        int maxmn = Math.max(m, n);
        int lwork;
        float[] work;
        info = new org.netlib.util.intW(2);
        int[] iwork = new int[8 * minmn];
        s = new float[minmn];
        if (wantUV = true) {
            if (wantWholeUV) { // JOBZ='A'
                Ut = new float[m * m];
                V = new float[n * n];
                lwork = 3 * minmn * minmn + Math.max(maxmn, 4 * minmn * minmn + 4 * minmn) + maxmn;
                work = new float[lwork];
                Sgesdd.sgesdd("A", m, n, elems, 0, m, s, 0, Ut, 0, m, V, 0, n, work, 0, lwork, iwork, 0, info);
            } else { // JOBZ='S'
                Ut = new float[m * minmn];
                V = new float[minmn * n];
                lwork = 3 * minmn * minmn + Math.max(maxmn, 4 * minmn * minmn + 4 * minmn) + maxmn;
                work = new float[lwork];
                Sgesdd.sgesdd("S", m, n, elems, 0, m, s, 0, Ut, 0, m, V, 0, minmn, work, 0, lwork, iwork, 0, info);
            }
        } else {// JOBZ='N'
            lwork = 3 * minmn + Math.max(maxmn, 6 * minmn) + maxmn;
            work = new float[lwork];
            Sgesdd.sgesdd("N", m, n, elems, 0, m, s, 0, null, 0, m, null, 0, n, work, 0, lwork, iwork, 0, info);
        }
    }

    /**
     * Returns the two norm condition number, which is <tt>max(S) / min(S)</tt>.
     */
    public float cond() {
        return s[0] / s[Math.min(m, n) - 1];
    }

    /**
     * Returns the diagonal matrix of singular values.
     * 
     * @return S
     */
    public FloatMatrix2D getS() {
        FloatMatrix2D S;
        if (wantWholeUV == false) {
            S = new DenseFloatMatrix2D(minmn, minmn);
        } else {
            S = new DenseFloatMatrix2D(m, n);
        }
        for (int i = 0; i < s.length; i++) {
            S.setQuick(i, i, s[i]);
        }
        return S;
    }

    /**
     * Returns the diagonal of <tt>S</tt>, which is a one-dimensional array
     * of singular values
     * 
     * @return diagonal of <tt>S</tt>.
     */
    public float[] getSingularValues() {
        return s;
    }

    /**
     * Returns the left singular vectors <tt>U</tt>.
     * 
     * @return <tt>U</tt>
     */
    public FloatMatrix2D getU() {
        if (wantUV == false) {
            throw new IllegalAccessError("Matrix U was not computed");
        } else {
            if (wantWholeUV == false) {
                return new DenseFloatMatrix2D(minmn, m).assign(Ut).viewDice().copy();
            } else {
                return new DenseFloatMatrix2D(m, m).assign(Ut).viewDice().copy();
            }
        }
    }

    /**
     * Returns the transpose of the left singular vectors <tt>U</tt>.
     * 
     * @return <tt>U'</tt>
     */
    public FloatMatrix2D getUt() {
        if (wantUV == false) {
            throw new IllegalAccessError("Matrix Ut was not computed");
        } else {
            if (wantWholeUV == false) {
                return new DenseFloatMatrix2D(m, minmn).assign(Ut);
            } else {
                return new DenseFloatMatrix2D(m, m).assign(Ut);
            }
        }

    }

    /**
     * Returns the transpose of the right singular vectors <tt>V</tt>.
     * 
     * @return <tt>V'</tt>
     */
    public FloatMatrix2D getVt() {
        if (wantUV == false) {
            throw new IllegalAccessError("Matrix Vt was not computed");
        } else {
            if (wantWholeUV == false) {
                return new DenseFloatMatrix2D(minmn, n).assign(V).viewDice().copy();
            } else {
                return new DenseFloatMatrix2D(n, n).assign(V).viewDice().copy();
            }
        }

    }

    /**
     * Returns the right singular vectors <tt>V</tt>.
     * 
     * @return <tt>V</tt>
     */
    public FloatMatrix2D getV() {
        if (wantUV == false) {
            throw new IllegalAccessError("Matrix V was not computed");
        } else {
            if (wantWholeUV == false) {
                return new DenseFloatMatrix2D(n, minmn).assign(V);
            } else {
                return new DenseFloatMatrix2D(n, n).assign(V);
            }
        }
    }

    /**
     * Returns the output flag
     * 
     * @return 0: successful exit<br> < 0: if INFO = -i, the i-th argument had
     *         an illegal value<br> > 0: process did not converge.
     */
    public org.netlib.util.intW getInfo() {
        return info;
    }

    /**
     * Returns the two norm, which is <tt>max(S)</tt>.
     */
    public float norm2() {
        return s[0];
    }

    /**
     * Returns the effective numerical matrix rank, which is the number of
     * nonnegligible singular values.
     */
    public int rank() {
        float eps = (float) Math.pow(2.0, -23.0);
        float tol = Math.max(m, n) * s[0] * eps;
        int r = 0;
        for (int i = 0; i < s.length; i++) {
            if (s[i] > tol) {
                r++;
            }
        }
        return r;
    }

    /**
     * Returns a String with (propertyName, propertyValue) pairs. Useful for
     * debugging or to quickly get the rough picture. For example,
     * 
     * <pre>
     * 	 rank          : 3
     * 	 trace         : 0
     * 	
     * </pre>
     */
    public String toString() {
        StringBuffer buf = new StringBuffer();
        String unknown = "Illegal operation or error: ";

        buf.append("---------------------------------------------------------------------\n");
        buf.append("SingularValueDecomposition(A) --> cond(A), rank(A), norm2(A), U, S, V\n");
        buf.append("---------------------------------------------------------------------\n");

        buf.append("cond = ");
        try {
            buf.append(String.valueOf(this.cond()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\nrank = ");
        try {
            buf.append(String.valueOf(this.rank()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\nnorm2 = ");
        try {
            buf.append(String.valueOf(this.norm2()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nU = ");
        try {
            buf.append(String.valueOf(this.getU()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nS = ");
        try {
            buf.append(String.valueOf(this.getS()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nV = ");
        try {
            buf.append(String.valueOf(this.getV()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        return buf.toString();
    }
}
