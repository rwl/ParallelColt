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
 * For a symmetric, positive definite matrix <tt>A</tt>, the Cholesky
 * decomposition is a lower triangular matrix <tt>L</tt> so that <tt>A = L*L'</tt>; If
 * the matrix is not symmetric positive definite, the IllegalArgumentException
 * is thrown.
 */
public class DenseDoubleCholeskyDecomposition implements java.io.Serializable {
    static final long serialVersionUID = 1020;

    private DoubleMatrix2D Lt;
    private double[] elementsA;
    private boolean columnMatrix = false;

    /**
     * Row and column dimension (square matrix).
     */
    private int n;

    /**
     * Constructs and returns a new Cholesky decomposition object for a
     * symmetric and positive definite matrix; The decomposed matrices can be
     * retrieved via instance methods of the returned decomposition object.
     * 
     * @param A
     *            Square, symmetric positive definite matrix .
     * @throws IllegalArgumentException
     *             if <tt>A</tt> is not square or is not a symmetric positive
     *             definite.
     */
    public DenseDoubleCholeskyDecomposition(DoubleMatrix2D A) {
        DoubleProperty.DEFAULT.checkSquare(A);
        DoubleProperty.DEFAULT.checkDense(A);
        if (A instanceof DenseDoubleMatrix2D) {
            elementsA = (double[]) A.viewDice().copy().elements();
        } else {
            columnMatrix = true;
            elementsA = (double[]) A.copy().elements();
        }
        n = A.rows();
        Dplasma.plasma_Init(n, n, 1);
        int info = Dplasma.plasma_DPOTRF(Dplasma.PlasmaUpper, n, elementsA, 0, n);
        Dplasma.plasma_Finalize();
        if (info > 0) {
            throw new IllegalArgumentException("Matrix is not symmetric positive definite.");
        }
        if (info < 0) {
            throw new IllegalArgumentException("Error occured while computing Cholesky decomposition: " + info);
        }
    }

    /**
     * Returns the triangular factor, <tt>L</tt>.
     * 
     * @return <tt>L</tt>
     */
    public DoubleMatrix2D getL() {
        if (Lt != null) {
            return Lt.viewDice().copy();
        } else {
            if (columnMatrix) {
                Lt = new DenseColumnDoubleMatrix2D(n, n);
                double[] Lelems = (double[]) Lt.elements();
                for (int c = n; --c >= 0;) {
                    for (int r = n; --r >= c;) {
                        Lelems[r * n + c] = elementsA[r * n + c];
                    }
                }
            } else {
                Lt = new DenseDoubleMatrix2D(n, n);
                double[] Lelems = (double[]) Lt.elements();
                for (int c = n; --c >= 0;) {
                    for (int r = n; --r >= c;) {
                        Lelems[c * n + r] = elementsA[r * n + c];
                    }
                }
            }
            return Lt.viewDice().copy();
        }
    }

    public DoubleMatrix2D getLtranspose() {
        if (Lt != null) {
            return Lt;
        } else {
            if (columnMatrix) {
                Lt = new DenseColumnDoubleMatrix2D(n, n);
                double[] Lelems = (double[]) Lt.elements();
                for (int c = n; --c >= 0;) {
                    for (int r = n; --r >= c;) {
                        Lelems[r * n + c] = elementsA[r * n + c];
                    }
                }
            } else {
                Lt = new DenseDoubleMatrix2D(n, n);
                double[] Lelems = (double[]) Lt.elements();
                for (int c = n; --c >= 0;) {
                    for (int r = n; --r >= c;) {
                        Lelems[c * n + r] = elementsA[r * n + c];
                    }
                }
            }
            return Lt;
        }
    }

    /**
     * Solves <tt>A*X = B</tt>(in-place). Upon return <tt>B</tt> is overridden
     * with the result <tt>X</tt>.
     * 
     * @param B
     *            A Matrix with as many rows as <tt>A</tt> and any number of
     *            columns.
     * @exception IllegalArgumentException
     *                if <tt>B.rows() != A.rows()</tt>.
     */
    public void solve(DoubleMatrix2D B) {
        if (B.rows() != n) {
            throw new IllegalArgumentException("B.rows() != A.rows()");
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
        Dplasma.plasma_Init(n, n, nrhs);
        int info = Dplasma.plasma_DPOTRS(Dplasma.PlasmaUpper, n, nrhs, elementsA, 0, n, elementsX, 0, n);
        Dplasma.plasma_Finalize();
        if (info != 0) {
            throw new IllegalArgumentException(
                    "Error occured while solving the system of equation using Cholesky decomposition: " + info);
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
     * Solves <tt>A*x = b</tt>(in-place). Upon return <tt>b</tt> is overridden
     * with the result <tt>x</tt>.
     * 
     * @param b
     *            A vector with of size A.rows();
     * @exception IllegalArgumentException
     *                if <tt>b.size() != A.rows()</tt>.
     */
    public void solve(DoubleMatrix1D b) {
        if (b.size() != n) {
            throw new IllegalArgumentException("b.size() != A.rows()");
        }
        DoubleProperty.DEFAULT.checkDense(b);
        double[] elementsX;
        if (b.isView()) {
            elementsX = (double[]) b.copy().elements();
        } else {
            elementsX = (double[]) b.elements();
        }
        Dplasma.plasma_Init(n, n, 1);
        int info = Dplasma.plasma_DPOTRS(Dplasma.PlasmaUpper, n, 1, elementsA, 0, n, elementsX, 0, n);
        Dplasma.plasma_Finalize();
        if (info != 0) {
            throw new IllegalArgumentException(
                    "Error occured while solving the system of equation using Cholesky decomposition: " + info);
        }
        if (b.isView()) {
            b.assign(elementsX);
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

        buf.append("--------------------------------------------------------------------------\n");
        buf.append("CholeskyDecomposition(A) --> L, inverse(A)\n");
        buf.append("--------------------------------------------------------------------------\n");

        buf.append("\nL = ");
        try {
            buf.append(String.valueOf(this.getL()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\ninverse(A) = ");
        try {
            DoubleMatrix2D X = cern.colt.matrix.tdouble.DoubleFactory2D.dense.identity(n);
            this.solve(X);
            buf.append(String.valueOf(X));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        return buf.toString();
    }
}
