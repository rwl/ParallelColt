package cern.colt.matrix.tfloat.algo.decomposition;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.FloatProperty;
import cern.colt.matrix.tfloat.impl.SparseCCFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.SparseRCFloatMatrix2D;
import edu.emory.mathcs.csparsej.tfloat.Scs_happly;
import edu.emory.mathcs.csparsej.tfloat.Scs_ipvec;
import edu.emory.mathcs.csparsej.tfloat.Scs_pvec;
import edu.emory.mathcs.csparsej.tfloat.Scs_qr;
import edu.emory.mathcs.csparsej.tfloat.Scs_sqr;
import edu.emory.mathcs.csparsej.tfloat.Scs_usolve;
import edu.emory.mathcs.csparsej.tfloat.Scs_utsolve;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scs;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scsn;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scss;

/**
 * For an <tt>m x n</tt> matrix <tt>A</tt> with <tt>m >= n</tt>, the QR
 * decomposition is an <tt>m x n</tt> orthogonal matrix <tt>Q</tt> and an
 * <tt>n x n</tt> upper triangular matrix <tt>R</tt> so that <tt>A = Q*R</tt>.
 * <P>
 * The QR decompostion always exists, even if the matrix does not have full
 * rank. The primary use of the QR decomposition is in the least squares
 * solution of nonsquare systems of simultaneous linear equations. This will
 * fail if <tt>isFullRank()</tt> returns <tt>false</tt>.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseFloatQRDecomposition {
    private Scss S;
    private Scsn N;
    private FloatMatrix2D R;
    private FloatMatrix2D V;
    private int m, n;
    private boolean rcMatrix = false;

    /**
     * Constructs and returns a new QR decomposition object; computed by
     * Householder reflections; If m < n then then the QR of A' is computed. The
     * decomposed matrices can be retrieved via instance methods of the returned
     * decomposition object.
     * 
     * @param A
     *            A rectangular matrix.
     * @param order
     *            ordering option (0 to 3); 0: natural ordering, 1: amd(A+A'),
     *            2: amd(S'*S), 3: amd(A'*A)
     * @throws IllegalArgumentException
     *             if <tt>A</tt> is not sparse
     * @throws IllegalArgumentException
     *             if <tt>order</tt> is not in [0,3]
     */
    public SparseFloatQRDecomposition(FloatMatrix2D A, int order) {
        FloatProperty.DEFAULT.checkSparse(A);
        if (order < 0 || order > 3) {
            throw new IllegalArgumentException("order must be a number between 0 and 3");
        }
        m = A.rows();
        n = A.columns();
        Scs dcs;
        if (A instanceof SparseRCFloatMatrix2D) {
            rcMatrix = true;
            if (m >= n) {
                dcs = ((SparseRCFloatMatrix2D) A).getColumnCompressed().elements();
            } else {
                dcs = ((SparseRCFloatMatrix2D) A).getColumnCompressed().getTranspose().elements();
            }
        } else {
            if (m >= n) {
                dcs = (Scs) A.elements();
            } else {
                dcs = ((SparseCCFloatMatrix2D) A).getTranspose().elements();
            }
        }
        S = Scs_sqr.cs_sqr(order, dcs, true);
        if (S == null) {
            throw new IllegalArgumentException("Exception occured in cs_sqr()");
        }
        N = Scs_qr.cs_qr(dcs, S);
        if (N == null) {
            throw new IllegalArgumentException("Exception occured in cs_qr()");
        }
    }

    /**
     * Returns a copy of the Householder vectors v, from the Householder
     * reflections H = I - beta*v*v'.
     * 
     * @return the Householder vectors.
     */
    public FloatMatrix2D getV() {
        if (V == null) {
            V = new SparseCCFloatMatrix2D(N.L);
            if (rcMatrix) {
                V = ((SparseCCFloatMatrix2D) V).getRowCompressed();
            }
        }
        return V.copy();
    }

    /**
     * Returns a copy of the beta factors, from the Householder reflections H =
     * I - beta*v*v'.
     * 
     * @return the beta factors.
     */
    public float[] getBeta() {
        if (N.B == null) {
            return null;
        }
        float[] beta = new float[N.B.length];
        System.arraycopy(N.B, 0, beta, 0, N.B.length);
        return beta;
    }

    /**
     * Returns a copy of the upper triangular factor, <tt>R</tt>.
     * 
     * @return <tt>R</tt>
     */
    public FloatMatrix2D getR() {
        if (R == null) {
            R = new SparseCCFloatMatrix2D(N.U);
            if (rcMatrix) {
                R = ((SparseCCFloatMatrix2D) R).getRowCompressed();
            }

        }
        return R.copy();
    }

    /**
     * Returns a copy of the symbolic QR analysis object
     * 
     * @return symbolic QR analysis
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
     * Returns whether the matrix <tt>A</tt> has full rank.
     * 
     * @return true if <tt>R</tt>, and hence <tt>A</tt>, has full rank.
     */
    public boolean hasFullRank() {
        if (R == null) {
            R = new SparseCCFloatMatrix2D(N.U);
            if (rcMatrix) {
                R = ((SparseCCFloatMatrix2D) R).getRowCompressed();
            }
        }
        int mn = Math.min(m, n);
        for (int j = 0; j < mn; j++) {
            if (R.getQuick(j, j) == 0)
                return false;
        }
        return true;
    }

    /**
     * Solve a least-squares problem (min ||Ax-b||_2, where A is m-by-n with m
     * >= n) or underdetermined system (Ax=b, where m < n). Upon return
     * <tt>b</tt> is overridden with the result <tt>x</tt>.
     * 
     * @param b
     *            right-hand side.
     * @exception IllegalArgumentException
     *                if <tt>b.size() != max(A.rows(), A.columns())</tt>.
     * @exception IllegalArgumentException
     *                if <tt>!this.hasFullRank()</tt> (<tt>A</tt> is rank
     *                deficient).
     */
    public void solve(FloatMatrix1D b) {
        if (b.size() != Math.max(m, n)) {
            throw new IllegalArgumentException("The size b must be equal to max(A.rows(), A.columns()).");
        }
        if (!this.hasFullRank()) {
            throw new IllegalArgumentException("Matrix is rank deficient.");
        }
        float[] x;
        if (b.isView()) {
            x = (float[]) b.copy().elements();
        } else {
            x = (float[]) b.elements();
        }
        if (m >= n) {
            float[] y = new float[S != null ? S.m2 : 1]; /* get workspace */
            Scs_ipvec.cs_ipvec(S.pinv, x, y, m); /* y(0:m-1) = b(p(0:m-1) */
            for (int k = 0; k < n; k++) /* apply Householder refl. to x */
            {
                Scs_happly.cs_happly(N.L, k, N.B[k], y);
            }
            Scs_usolve.cs_usolve(N.U, y); /* y = R\y */
            Scs_ipvec.cs_ipvec(S.q, y, x, n); /* x(q(0:n-1)) = y(0:n-1) */
        } else {
            float[] y = new float[S != null ? S.m2 : 1]; /* get workspace */
            Scs_pvec.cs_pvec(S.q, x, y, m); /* y(q(0:m-1)) = b(0:m-1) */
            Scs_utsolve.cs_utsolve(N.U, y); /* y = R'\y */
            for (int k = m - 1; k >= 0; k--) /* apply Householder refl. to x */
            {
                Scs_happly.cs_happly(N.L, k, N.B[k], y);
            }
            Scs_pvec.cs_pvec(S.pinv, y, x, n); /* x(0:n-1) = y(p(0:n-1)) */
        }
    }

}
