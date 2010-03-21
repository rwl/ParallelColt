/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.algo.decomposition;

import org.netlib.lapack.LAPACK;

import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.FloatProperty;
import cern.colt.matrix.tfloat.impl.DenseColumnFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DiagonalFloatMatrix2D;

/**
 * 
 * For an <tt>m x n</tt> matrix <tt>A</tt>, the singular value decomposition is
 * an <tt>m x m</tt> orthogonal matrix <tt>U</tt>, an <tt>m x n</tt> diagonal
 * matrix <tt>S</tt>, and an <tt>n x n</tt> orthogonal matrix <tt>V</tt> so that
 * <tt>A = U*S*V'</tt>.
 * <P>
 * The singular values, <tt>sigma[k] = S[k][k]</tt>, are ordered so that
 * <tt>sigma[0] >= sigma[1] >= ... >= sigma[min(m-1,n-1)]</tt>.
 * <P>
 * 
 * This implementation uses the divide-and-conquer algorithm (dgesdd) from
 * LAPACK.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseFloatSingularValueDecomposition {

    private FloatMatrix2D U;

    private FloatMatrix2D V;

    private FloatMatrix2D S;

    private float[] elementsU;

    private float[] elementsVt;

    private float[] elementsS;

    private org.netlib.util.intW info;

    private int m;

    private int n;

    private int mn;

    private boolean wantWholeUV;

    private boolean wantUV;

    private boolean columnMatrix = false;

    /**
     * Constructs and returns a new singular value decomposition object; The
     * decomposed matrices can be retrieved via instance methods of the returned
     * decomposition object.
     * 
     * @param A
     *            rectangular matrix
     * 
     * @param wantUV
     *            if true then all matrices (U, S, V') are computed; otherwise
     *            only S is computed
     * @param wantWholeUV
     *            if true then all m columns of U and all n rows of V' are
     *            computed; otherwise only the first min(m,n) columns of U and
     *            the first min(m,n) rows of V' are computed
     */
    public DenseFloatSingularValueDecomposition(FloatMatrix2D A, boolean wantUV, boolean wantWholeUV) {
        FloatProperty.DEFAULT.checkDense(A);
        this.wantUV = wantUV;
        this.wantWholeUV = wantWholeUV;
        m = A.rows();
        n = A.columns();
        float[] elementsA;
        if (A instanceof DenseColumnFloatMatrix2D) {
            elementsA = (float[]) A.copy().elements();
            columnMatrix = true;
        } else {
            elementsA = (float[]) A.viewDice().copy().elements();
        }
        mn = Math.min(m, n);
        int maxmn = Math.max(m, n);
        int lwork;
        float[] work;
        info = new org.netlib.util.intW(2);
        int[] iwork = new int[8 * mn];
        elementsS = new float[mn];
        if (wantUV = true) {
            if (wantWholeUV) { // JOBZ='A'
                elementsU = new float[m * m];
                elementsVt = new float[n * n];
                lwork = 3 * mn * mn + Math.max(maxmn, 4 * mn * mn + 4 * mn) + maxmn;
                work = new float[lwork];
                LAPACK.getInstance().sgesdd("A", m, n, elementsA, m, elementsS, elementsU, m, elementsVt, n, work,
                        lwork, iwork, info);
            } else { // JOBZ='S'
                elementsU = new float[m * mn];
                elementsVt = new float[mn * n];
                lwork = 3 * mn * mn + Math.max(maxmn, 4 * mn * mn + 4 * mn) + maxmn;
                work = new float[lwork];
                LAPACK.getInstance().sgesdd("S", m, n, elementsA, m, elementsS, elementsU, m, elementsVt, mn, work,
                        lwork, iwork, info);
            }
        } else {// JOBZ='N'
            lwork = 3 * mn + Math.max(maxmn, 6 * mn) + maxmn;
            work = new float[lwork];
            LAPACK.getInstance().sgesdd("N", m, n, elementsA, m, elementsS, null, m, null, n, work, lwork, iwork, info);
        }
        if (info.val != 0) {
            throw new IllegalArgumentException("Error occured while computing SVD decomposition: " + info);
        }
    }

    /**
     * Returns the two norm condition number, which is <tt>max(S) / min(S)</tt>.
     */
    public float cond() {
        return elementsS[0] / elementsS[mn - 1];
    }

    /**
     * Returns the diagonal matrix of singular values.
     * 
     * @return S
     */
    public FloatMatrix2D getS() {
        if (S == null) {
            if (wantWholeUV == false) {
                S = new DiagonalFloatMatrix2D(mn, mn, 0);
            } else {
                S = new DiagonalFloatMatrix2D(m, n, 0);
            }
            for (int i = 0; i < mn; i++) {
                S.setQuick(i, i, elementsS[i]);
            }
        }
        return S.copy();
    }

    /**
     * Returns the diagonal of <tt>S</tt>, which is a one-dimensional array of
     * singular values
     * 
     * @return diagonal of <tt>S</tt>.
     */
    public float[] getSingularValues() {
        return elementsS;
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
            if (U == null) {
                if (wantWholeUV == false) {
                    if (columnMatrix) {
                        U = new DenseColumnFloatMatrix2D(m, mn).assign(elementsU);
                    } else {
                        U = new DenseFloatMatrix2D(mn, m).assign(elementsU).viewDice();
                    }
                } else {
                    if (columnMatrix) {
                        U = new DenseColumnFloatMatrix2D(m, m).assign(elementsU);
                    } else {
                        U = new DenseFloatMatrix2D(m, m).assign(elementsU).viewDice();
                    }
                }
            }
            return U.copy();
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
            if (V == null) {
                if (wantWholeUV == false) {
                    if (columnMatrix) {
                        V = new DenseColumnFloatMatrix2D(mn, n).assign(elementsVt).viewDice();
                    } else {
                        V = new DenseFloatMatrix2D(n, mn).assign(elementsVt);
                    }
                } else {
                    if (columnMatrix) {
                        V = new DenseColumnFloatMatrix2D(n, n).assign(elementsVt).viewDice();
                    } else {
                        V = new DenseFloatMatrix2D(n, n).assign(elementsVt);
                    }
                }
            }
            return V.copy();
        }
    }

    /**
     * Returns the output flag
     * 
     * @return 0: successful exit<br>
     *         < 0: if INFO = -i, the i-th argument had an illegal value<br>
     *         > 0: process did not converge.
     */
    public org.netlib.util.intW getInfo() {
        return info;
    }

    /**
     * Returns the two norm, which is <tt>max(S)</tt>.
     */
    public float norm2() {
        return elementsS[0];
    }

    /**
     * Returns the effective numerical matrix rank, which is the number of
     * nonnegligible singular values.
     */
    public int rank() {
        float eps = (float) Math.pow(2.0, -23.0);
        float tol = Math.max(m, n) * elementsS[0] * eps;
        int r = 0;
        for (int i = 0; i < elementsS.length; i++) {
            if (elementsS[i] > tol) {
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
