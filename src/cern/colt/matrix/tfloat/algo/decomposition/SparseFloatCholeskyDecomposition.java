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
import cern.colt.matrix.tfloat.algo.FloatProperty;
import cern.colt.matrix.tfloat.impl.SparseCCFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.SparseRCFloatMatrix2D;
import edu.emory.mathcs.csparsej.tfloat.Scs_chol;
import edu.emory.mathcs.csparsej.tfloat.Scs_ipvec;
import edu.emory.mathcs.csparsej.tfloat.Scs_lsolve;
import edu.emory.mathcs.csparsej.tfloat.Scs_ltsolve;
import edu.emory.mathcs.csparsej.tfloat.Scs_pvec;
import edu.emory.mathcs.csparsej.tfloat.Scs_schol;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scs;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scsn;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scss;

/**
 * For a symmetric, positive definite matrix <tt>A</tt>, the Cholesky
 * decomposition is a lower triangular matrix <tt>L</tt> so that <tt>A = L*L'</tt>; If
 * the matrix is not symmetric positive definite, the IllegalArgumentException
 * is thrown.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseFloatCholeskyDecomposition {
    private Scss S;
    private Scsn N;
    private FloatMatrix2D L;
    private boolean rcMatrix = false;

    /**
     * Row and column dimension (square matrix).
     */
    private int n;

    /**
     * Constructs and returns a new Cholesky decomposition object for a sparse
     * symmetric and positive definite matrix; The decomposed matrices can be
     * retrieved via instance methods of the returned decomposition object.
     * 
     * @param A
     *            Square, symmetric positive definite matrix .
     * @param order
     *            ordering option (0 or 1); 0: natural ordering, 1: amd(A+A')
     * @throws IllegalArgumentException
     *             if <tt>A</tt> is not square or is not sparse or is not a
     *             symmetric positive definite.
     * @throws IllegalArgumentException
     *             if <tt>order != 0 || order != 1</tt>
     */
    public SparseFloatCholeskyDecomposition(FloatMatrix2D A, int order) {
        FloatProperty.DEFAULT.checkSquare(A);
        FloatProperty.DEFAULT.checkSparse(A);
        if (order < 0 || order > 1) {
            throw new IllegalArgumentException("order must be equal 0 or 1");
        }
        Scs dcs;
        if (A instanceof SparseRCFloatMatrix2D) {
            rcMatrix = true;
            dcs = ((SparseRCFloatMatrix2D) A).getColumnCompressed().elements();
        } else {
            dcs = (Scs) A.elements();
        }
        n = A.rows();
        S = Scs_schol.cs_schol(order, dcs);
        if (S == null) {
            throw new IllegalArgumentException("Exception occured in cs_schol()");
        }
        N = Scs_chol.cs_chol(dcs, S);
        if (N == null) {
            throw new IllegalArgumentException("Matrix is not symmetric positive definite");
        }
    }

    /**
     * Returns the triangular factor, <tt>L</tt>.
     * 
     * @return <tt>L</tt>
     */
    public FloatMatrix2D getL() {
        if (L == null) {
            L = new SparseCCFloatMatrix2D(N.L);
            if (rcMatrix) {
                L = ((SparseCCFloatMatrix2D) L).getRowCompressed();
            }
        }
        return L.copy();
    }

    /**
     * 
     * Returns the triangular factor, <tt>L'</tt>.
     * 
     * @return <tt>L'</tt>
     */
    public FloatMatrix2D getLtranspose() {
        if (L == null) {
            L = new SparseCCFloatMatrix2D(N.L);
            if (rcMatrix) {
                L = ((SparseCCFloatMatrix2D) L).getRowCompressed();
            }
        }
        if (rcMatrix) {
            return ((SparseRCFloatMatrix2D) L).getTranspose();
        } else {
            return ((SparseCCFloatMatrix2D) L).getTranspose();
        }
    }

    /**
     * Returns a copy of the symbolic Cholesky analysis object
     * 
     * @return symbolic Cholesky analysis
     */
    public Scss getSymbolicAnalysis() {
        Scss S2 = new Scss();
        S2.cp = S.cp != null ? S.cp.clone() : null;
        S2.leftmost = S.leftmost != null ? S.leftmost.clone() : null;
        S2.lnz = S.lnz;
        S2.m2 = S.m2;
        S2.parent = S.parent != null ? S.parent.clone() : null;
        S2.pinv = S.pinv != null ? S.pinv.clone() : null;
        S2.q = S.q != null ? S.q.clone() : null;
        S2.unz = S.unz;
        return S2;
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
    public void solve(FloatMatrix1D b) {
        if (b.size() != n) {
            throw new IllegalArgumentException("b.size() != A.rows()");
        }
        FloatProperty.DEFAULT.checkDense(b);
        float[] y = new float[n];
        float[] x;
        if (b.isView()) {
            x = (float[]) b.copy().elements();
        } else {
            x = (float[]) b.elements();
        }
        Scs_ipvec.cs_ipvec(S.pinv, x, y, n); /* y = P*b */
        Scs_lsolve.cs_lsolve(N.L, y); /* y = L\y */
        Scs_ltsolve.cs_ltsolve(N.L, y); /* y = L'\y */
        Scs_pvec.cs_pvec(S.pinv, y, x, n); /* x = P'*y */

        if (b.isView()) {
            b.assign(x);
        }
    }
}
