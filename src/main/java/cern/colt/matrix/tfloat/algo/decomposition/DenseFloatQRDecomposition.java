/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.algo.decomposition;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;
import cern.colt.matrix.tfloat.algo.FloatProperty;
import cern.jet.math.tfloat.FloatFunctions;

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
 */
public class DenseFloatQRDecomposition implements java.io.Serializable {
    static final long serialVersionUID = 1020;

    /**
     * Array for internal storage of decomposition.
     * 
     * @serial internal array storage.
     */
    private FloatMatrix2D QR;

    // private float[][] QR;

    /**
     * Row and column dimensions.
     * 
     * @serial column dimension.
     * @serial row dimension.
     */
    private int m, n;

    /**
     * Array for internal storage of diagonal of R.
     * 
     * @serial diagonal of R.
     */
    private FloatMatrix1D Rdiag;

    /**
     * Constructs and returns a new QR decomposition object; computed by
     * Householder reflections; The decomposed matrices can be retrieved via
     * instance methods of the returned decomposition object.
     * 
     * @param A
     *            A rectangular matrix.
     * @throws IllegalArgumentException
     *             if <tt>A.rows() < A.columns()</tt>.
     */

    public DenseFloatQRDecomposition(FloatMatrix2D A) {
        FloatProperty.DEFAULT.checkRectangular(A);

        cern.jet.math.tfloat.FloatFunctions F = cern.jet.math.tfloat.FloatFunctions.functions;
        // Initialize.
        QR = A.copy();
        m = A.rows();
        n = A.columns();
        Rdiag = A.like1D(n);
        // Rdiag = new float[n];
        cern.colt.function.tfloat.FloatFloatFunction hypot = DenseFloatAlgebra.hypotFunction();

        // precompute and cache some views to avoid regenerating them time and
        // again
        FloatMatrix1D[] QRcolumns = new FloatMatrix1D[n];
        FloatMatrix1D[] QRcolumnsPart = new FloatMatrix1D[n];
        for (int k = 0; k < n; k++) {
            QRcolumns[k] = QR.viewColumn(k);
            QRcolumnsPart[k] = QR.viewColumn(k).viewPart(k, m - k);
        }

        // Main loop.
        for (int k = 0; k < n; k++) {
            // FloatMatrix1D QRcolk = QR.viewColumn(k).viewPart(k,m-k);
            // Compute 2-norm of k-th column without under/overflow.
            float nrm = 0;
            // if (k<m) nrm = QRcolumnsPart[k].aggregate(hypot,F.identity);

            for (int i = k; i < m; i++) { // fixes bug reported by
                // hong.44@osu.edu
                nrm = DenseFloatAlgebra.hypot(nrm, QR.getQuick(i, k));
            }

            if (nrm != 0.0) {
                // Form k-th Householder vector.
                if (QR.getQuick(k, k) < 0)
                    nrm = -nrm;
                QRcolumnsPart[k].assign(cern.jet.math.tfloat.FloatFunctions.div(nrm));
                /*
                 * for (int i = k; i < m; i++) { QR[i][k] /= nrm; }
                 */

                QR.setQuick(k, k, QR.getQuick(k, k) + 1);

                // Apply transformation to remaining columns.
                for (int j = k + 1; j < n; j++) {
                    FloatMatrix1D QRcolj = QR.viewColumn(j).viewPart(k, m - k);
                    float s = QRcolumnsPart[k].zDotProduct(QRcolj);
                    /*
                     * // fixes bug reported by John Chambers FloatMatrix1D
                     * QRcolj = QR.viewColumn(j).viewPart(k,m-k); float s =
                     * QRcolumnsPart[k].zDotProduct(QRcolumns[j]); float s =
                     * 0.0; for (int i = k; i < m; i++) { s +=
                     * QR[i][k]*QR[i][j]; }
                     */
                    s = -s / QR.getQuick(k, k);
                    // QRcolumnsPart[j].assign(QRcolumns[k], F.plusMult(s));

                    for (int i = k; i < m; i++) {
                        QR.setQuick(i, j, QR.getQuick(i, j) + s * QR.getQuick(i, k));
                    }

                }
            }
            Rdiag.setQuick(k, -nrm);
        }
    }

    /**
     * Returns the Householder vectors <tt>H</tt>.
     * 
     * @return A lower trapezoidal matrix whose columns define the householder
     *         reflections.
     */
    public FloatMatrix2D getH() {
        return DenseFloatAlgebra.DEFAULT.trapezoidalLower(QR.copy());
    }

    /**
     * Generates and returns the (economy-sized) orthogonal factor <tt>Q</tt>.
     * 
     * @return <tt>Q</tt>
     */
    public FloatMatrix2D getQ() {
        cern.jet.math.tfloat.FloatFunctions F = cern.jet.math.tfloat.FloatFunctions.functions;
        FloatMatrix2D Q = QR.like();
        // float[][] Q = X.getArray();
        for (int k = n - 1; k >= 0; k--) {
            FloatMatrix1D QRcolk = QR.viewColumn(k).viewPart(k, m - k);
            Q.setQuick(k, k, 1);
            for (int j = k; j < n; j++) {
                if (QR.getQuick(k, k) != 0) {
                    FloatMatrix1D Qcolj = Q.viewColumn(j).viewPart(k, m - k);
                    float s = QRcolk.zDotProduct(Qcolj);
                    s = -s / QR.getQuick(k, k);
                    Qcolj.assign(QRcolk, FloatFunctions.plusMultSecond(s));
                }
            }
        }
        return Q;
    }

    /**
     * Returns the upper triangular factor, <tt>R</tt>.
     * 
     * @return <tt>R</tt>
     */
    public FloatMatrix2D getR() {
        FloatMatrix2D R = QR.like(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i < j)
                    R.setQuick(i, j, QR.getQuick(i, j));
                else if (i == j)
                    R.setQuick(i, j, Rdiag.getQuick(i));
                else
                    R.setQuick(i, j, 0);
            }
        }
        return R;
    }

    /**
     * Returns whether the matrix <tt>A</tt> has full rank.
     * 
     * @return true if <tt>R</tt>, and hence <tt>A</tt>, has full rank.
     */
    public boolean hasFullRank() {
        for (int j = 0; j < n; j++) {
            if (Rdiag.getQuick(j) == 0)
                return false;
        }
        return true;
    }

    /**
     * Least squares solution of <tt>A*x = b</tt>; <tt>returns x</tt>.
     * 
     * @param b
     *            right-hand side.
     * @return <tt>x</tt> that minimizes the two norm of <tt>Q*R*x - b</tt>.
     * @exception IllegalArgumentException
     *                if <tt>b.size() != A.rows()</tt>.
     * @exception IllegalArgumentException
     *                if <tt>!this.hasFullRank()</tt> (<tt>A</tt> is rank
     *                deficient).
     */
    public FloatMatrix1D solve(FloatMatrix1D b) {
        cern.jet.math.tfloat.FloatFunctions F = cern.jet.math.tfloat.FloatFunctions.functions;
        if (b.size() != m) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!this.hasFullRank()) {
            throw new IllegalArgumentException("Matrix is rank deficient.");
        }

        // Copy right hand side
        FloatMatrix1D x = b.copy();

        // Compute y = transpose(Q)*b
        for (int k = 0; k < n; k++) {
            float s = 0.0f;
            for (int i = k; i < m; i++) {
                s += QR.getQuick(i, k) * x.getQuick(i);
            }
            s = -s / QR.getQuick(k, k);
            for (int i = k; i < m; i++) {
                x.setQuick(i, x.getQuick(i) + s * QR.getQuick(i, k));
            }
        }
        // Solve R*x = y;

        for (int k = n - 1; k >= 0; k--) {
            x.setQuick(k, x.getQuick(k) / Rdiag.getQuick(k));
            for (int i = 0; i < k; i++) {
                x.setQuick(i, x.getQuick(i) - x.getQuick(k) * QR.getQuick(i, k));
            }
        }
        return x.viewPart(0, n).copy();
    }

    /**
     * Least squares solution of <tt>A*X = B</tt>; <tt>returns X</tt>.
     * 
     * @param B
     *            A matrix with as many rows as <tt>A</tt> and any number of
     *            columns.
     * @return <tt>X</tt> that minimizes the two norm of <tt>Q*R*X - B</tt>.
     * @exception IllegalArgumentException
     *                if <tt>B.rows() != A.rows()</tt>.
     * @exception IllegalArgumentException
     *                if <tt>!this.hasFullRank()</tt> (<tt>A</tt> is rank
     *                deficient).
     */
    public FloatMatrix2D solve(FloatMatrix2D B) {
        cern.jet.math.tfloat.FloatFunctions F = cern.jet.math.tfloat.FloatFunctions.functions;
        if (B.rows() != m) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!this.hasFullRank()) {
            throw new IllegalArgumentException("Matrix is rank deficient.");
        }

        // Copy right hand side
        int nx = B.columns();
        FloatMatrix2D X = B.copy();

        // Compute Y = transpose(Q)*B
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < nx; j++) {
                float s = 0.0f;
                for (int i = k; i < m; i++) {
                    s += QR.getQuick(i, k) * X.getQuick(i, j);
                }
                s = -s / QR.getQuick(k, k);
                for (int i = k; i < m; i++) {
                    X.setQuick(i, j, X.getQuick(i, j) + s * QR.getQuick(i, k));
                }
            }
        }
        // Solve R*X = Y;
        for (int k = n - 1; k >= 0; k--) {
            for (int j = 0; j < nx; j++) {
                X.setQuick(k, j, X.getQuick(k, j) / Rdiag.getQuick(k));
            }
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < nx; j++) {
                    X.setQuick(i, j, X.getQuick(i, j) - X.getQuick(k, j) * QR.getQuick(i, k));
                }
            }
        }
        return X.viewPart(0, 0, n, nx);
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
        buf.append("QRDecomposition(A) --> hasFullRank(A), H, Q, R, pseudo inverse(A)\n");
        buf.append("-----------------------------------------------------------------\n");

        buf.append("hasFullRank = ");
        try {
            buf.append(String.valueOf(this.hasFullRank()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nH = ");
        try {
            buf.append(String.valueOf(this.getH()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nQ = ");
        try {
            buf.append(String.valueOf(this.getQ()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\nR = ");
        try {
            buf.append(String.valueOf(this.getR()));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        buf.append("\n\npseudo inverse(A) = ");
        try {
            buf.append(String.valueOf(this.solve(cern.colt.matrix.tfloat.FloatFactory2D.dense.identity(QR.rows()))));
        } catch (IllegalArgumentException exc) {
            buf.append(unknown + exc.getMessage());
        }

        return buf.toString();
    }
}
