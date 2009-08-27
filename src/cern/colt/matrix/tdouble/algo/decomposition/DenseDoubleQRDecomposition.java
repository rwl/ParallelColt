/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.algo.decomposition;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import edu.emory.mathcs.jplasma.tdouble.Dplasma;

/**
 * For an <tt>m x n</tt> matrix <tt>A</tt> with <tt>m >= n</tt>, the QR
 * decomposition is an <tt>m x n</tt> orthogonal matrix <tt>Q</tt> and an
 * <tt>n x n</tt> upper triangular matrix <tt>R</tt> so that <tt>A = Q*R</tt>.
 * <P>
 * The QR decompostion always exists, even if the matrix does not have full
 * rank, so the constructor will never fail. The primary use of the QR
 * decomposition is in the least squares solution of nonsquare systems of
 * simultaneous linear equations. This will fail if <tt>isFullRank()</tt>
 * returns <tt>false</tt>.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DenseDoubleQRDecomposition implements java.io.Serializable {
    static final long serialVersionUID = 1020;

    /**
     * Array for internal storage of decomposition.
     * 
     * @serial internal array storage.
     */
    private double[] elementsA;

    private double[] T;

    private boolean columnMatrix = false;

    private DoubleMatrix2D R;

    private DoubleMatrix2D Q;

    /**
     * Row and column dimensions.
     * 
     * @serial column dimension.
     * @serial row dimension.
     */
    private int m, n;

    /**
     * Constructs and returns a new QR decomposition object; computed by
     * Householder reflections; The decomposed matrices can be retrieved via
     * instance methods of the returned decomposition object.
     * 
     * @param A
     *            A rectangular matrix.
     * 
     * @throws IllegalArgumentException
     *             if <tt>A.rows() < A.columns()</tt>.
     */

    public DenseDoubleQRDecomposition(DoubleMatrix2D A) {
        DoubleProperty.DEFAULT.checkRectangular(A);
        DoubleProperty.DEFAULT.checkDense(A);
        if (A instanceof DenseDoubleMatrix2D) {
            elementsA = (double[]) A.viewDice().copy().elements();
        } else {
            columnMatrix = true;
            elementsA = (double[]) A.copy().elements();
        }
        m = A.rows();
        n = A.columns();
        int lda = m;
        Dplasma.plasma_Init(m, n, 1);
        T = Dplasma.plasma_Allocate_T(m, n);
        int info = Dplasma.plasma_DGEQRF(m, n, elementsA, 0, lda, T, 0);
        Dplasma.plasma_Finalize();
        if (info != 0) {
            throw new IllegalArgumentException("Error occured while computing QR decomposition: " + info);
        }
    }

    /**
     * Generates and returns a copy of the orthogonal factor <tt>Q</tt>.
     * 
     * @param economySize
     *            if true, then Q is m-by-n, otherwise, Q is m-by-m
     * 
     * @return <tt>Q</tt>
     */
    public DoubleMatrix2D getQ(boolean economySize) {
        if (Q == null) {
            Dplasma.plasma_Init(m, n, 1);
            Q = new DenseColumnDoubleMatrix2D(m, m);
            double[] elementsQ = (double[]) Q.elements();
            for (int i = 0; i < m; i++)
                elementsQ[m * i + i] = 1.0;
            int info = Dplasma.plasma_DORMQR(Dplasma.PlasmaLeft, Dplasma.PlasmaNoTrans, m, m, n, elementsA, 0, m, T, 0,
                    elementsQ, 0, m);
            Dplasma.plasma_Finalize();
            if (info != 0) {
                throw new IllegalArgumentException("Error occured while computing matrix Q: " + info);
            }
            Q = Q.viewDice().copy();
        }
        if (!columnMatrix) {
            if (economySize) {
                return ((DenseColumnDoubleMatrix2D) Q.viewPart(0, 0, m, n)).getRowMajor();
            } else {
                return ((DenseColumnDoubleMatrix2D) Q).getRowMajor();
            }
        } else {
            if (economySize) {
                return Q.viewPart(0, 0, m, n).copy();
            } else {
                return Q.copy();
            }
        }
    }

    /**
     * Returns a copy of the upper triangular factor, <tt>R</tt>.
     * 
     * @param economySize
     *            if true, then R is n-by-n, otherwise, R is m-by-n
     * 
     * @return <tt>R</tt>
     */
    public DoubleMatrix2D getR(boolean economySize) {
        if (R == null) {
            R = new DenseColumnDoubleMatrix2D(m, n);
            double[] elementsR = (double[]) R.elements();
            for (int c = 0; c < n; c++) {
                for (int r = 0; r < m; r++) {
                    if (r <= c)
                        elementsR[c * m + r] = elementsA[c * m + r];
                }
            }
        }
        if (!columnMatrix) {
            if (economySize) {
                return ((DenseColumnDoubleMatrix2D) R.viewPart(0, 0, n, n)).getRowMajor();
            } else {
                return ((DenseColumnDoubleMatrix2D) R).getRowMajor();
            }
        } else {
            if (economySize) {
                return ((DenseColumnDoubleMatrix2D) R.viewPart(0, 0, n, n)).copy();
            } else {
                return R.copy();
            }
        }
    }

    /**
     * Returns whether the matrix <tt>A</tt> has full rank.
     * 
     * @return true if <tt>R</tt>, and hence <tt>A</tt>, has full rank.
     */
    public boolean hasFullRank() {
        for (int j = 0; j < n; j++) {
            if (elementsA[j * m + j] == 0)
                return false;
        }
        return true;
    }

    /**
     * Least squares solution of <tt>A*x = b</tt> (in-place). Upon return
     * <tt>b</tt> is overridden with the result <tt>x</tt>.
     * 
     * @param b
     *            right-hand side.
     * @exception IllegalArgumentException
     *                if <tt>b.size() != A.rows()</tt>.
     * @exception IllegalArgumentException
     *                if <tt>!this.hasFullRank()</tt> (<tt>A</tt> is rank
     *                deficient).
     */
    public void solve(DoubleMatrix1D b) {
        DoubleProperty.DEFAULT.checkDense(b);
        if (b.size() != m) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!this.hasFullRank()) {
            throw new IllegalArgumentException("Matrix is rank deficient.");
        }
        double[] elementsX;
        if (b.isView()) {
            elementsX = (double[]) b.copy().elements();
        } else {
            elementsX = (double[]) b.elements();
        }
        Dplasma.plasma_Init(m, n, 1);
        int info = Dplasma.plasma_DORMQR(Dplasma.PlasmaLeft, Dplasma.PlasmaNoTrans, m, 1, n, elementsA, 0, m, T, 0,
                elementsX, 0, m);
        if (info != 0) {
            throw new IllegalArgumentException(
                    "Error occured while solving the system of equation using QR decomposition: " + info);
        }
        info = Dplasma.plasma_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaUpper, Dplasma.PlasmaNoTrans,
                Dplasma.PlasmaNonUnit, n, 1, elementsA, 0, m, elementsX, 0, m);
        Dplasma.plasma_Finalize();
        if (info != 0) {
            throw new IllegalArgumentException(
                    "Error occured while solving the system of equation using QR decomposition: " + info);
        }
        if (b.isView()) {
            b.assign(elementsX);
        }
    }

    /**
     * Least squares solution of <tt>A*X = B</tt>(in-place). Upon return
     * <tt>B</tt> is overridden with the result <tt>X</tt>.
     * 
     * @param B
     *            A matrix with as many rows as <tt>A</tt> and any number of
     *            columns.
     * @exception IllegalArgumentException
     *                if <tt>B.rows() != A.rows()</tt>.
     * @exception IllegalArgumentException
     *                if <tt>!this.hasFullRank()</tt> (<tt>A</tt> is rank
     *                deficient).
     */
    public void solve(DoubleMatrix2D B) {
        if (B.rows() != m) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!this.hasFullRank()) {
            throw new IllegalArgumentException("Matrix is rank deficient.");
        }
        DoubleProperty.DEFAULT.checkDense(B);
        double[] elementsX;
        if (B instanceof DenseDoubleMatrix2D) {
            elementsX = (double[]) B.viewDice().copy().elements();
        } else {
            if (B.isView()) {
                elementsX = (double[]) B.copy().elements();
            } else {
                elementsX = (double[]) B.elements();
            }
        }
        int nrhs = B.columns();
        Dplasma.plasma_Init(m, n, nrhs);
        int info = Dplasma.plasma_DORMQR(Dplasma.PlasmaLeft, Dplasma.PlasmaNoTrans, m, nrhs, n, elementsA, 0, m, T, 0,
                elementsX, 0, m);
        if (info != 0) {
            throw new IllegalArgumentException(
                    "Error occured while solving the system of equation using QR decomposition: " + info);
        }
        info = Dplasma.plasma_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaUpper, Dplasma.PlasmaNoTrans,
                Dplasma.PlasmaNonUnit, n, nrhs, elementsA, 0, m, elementsX, 0, m);
        Dplasma.plasma_Finalize();
        if (info != 0) {
            throw new IllegalArgumentException(
                    "Error occured while solving the system of equation using QR decomposition: " + info);
        }
        if (B instanceof DenseDoubleMatrix2D) {
            B.viewDice().assign(elementsX);
        } else {
            if (B.isView()) {
                B.assign(elementsX);
            }
        }
    }

    /**
     * Returns a String with (propertyName, propertyValue) pairs. Useful for
     * debugging or to quickly get the rough picture. For example,
     * 
     * <pre>
     *   rank          : 3
     *   trace         : 0
     * 
     * </pre>
     */

    public String toString() {
        StringBuffer buf = new StringBuffer();
        String unknown = "Illegal operation or error: ";

        buf.append("-----------------------------------------------------------------\n");
        buf.append("QRDecomposition(A) --> hasFullRank(A), Q, R, pseudo inverse(A)\n");
        buf.append("-----------------------------------------------------------------\n");

        buf.append("hasFullRank = ");
        try {
            buf.append(String.valueOf(this.hasFullRank()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nQ = ");
        try {
            buf.append(String.valueOf(this.getQ(false)));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nR = ");
        try {
            buf.append(String.valueOf(this.getR(false)));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\npseudo inverse(A) = ");
        try {
            DoubleMatrix2D X = cern.colt.matrix.tdouble.DoubleFactory2D.dense.identity(m);
            this.solve(X);
            buf.append(String.valueOf(X));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        return buf.toString();
    }
}
