package cern.colt.matrix.tdouble.algo.decomposition;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.impl.SparseCCDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2D;
import edu.emory.mathcs.csparsej.tdouble.Dcs_dmperm;
import edu.emory.mathcs.csparsej.tdouble.Dcs_ipvec;
import edu.emory.mathcs.csparsej.tdouble.Dcs_lsolve;
import edu.emory.mathcs.csparsej.tdouble.Dcs_lu;
import edu.emory.mathcs.csparsej.tdouble.Dcs_sqr;
import edu.emory.mathcs.csparsej.tdouble.Dcs_usolve;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcsd;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcsn;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcss;

/**
 * For a square matrix <tt>A</tt>, the LU decomposition is an unit lower
 * triangular matrix <tt>L</tt>, an upper triangular matrix <tt>U</tt>, and a
 * permutation vector <tt>piv</tt> so that <tt>A(piv,:) = L*U</tt>
 * <P>
 * The LU decomposition with pivoting always exists, even if the matrix is
 * singular. The primary use of the LU decomposition is in the solution of
 * square systems of simultaneous linear equations. This will fail if
 * <tt>isNonsingular()</tt> returns false.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseDoubleLUDecomposition {
    private Dcss S;
    private Dcsn N;
    private DoubleMatrix2D L;
    private DoubleMatrix2D U;
    private boolean rcMatrix = false;
    private boolean isNonSingular = true;
    /**
     * Row and column dimension (square matrix).
     */
    private int n;

    /**
     * Constructs and returns a new LU Decomposition object; The decomposed
     * matrices can be retrieved via instance methods of the returned
     * decomposition object.
     * 
     * @param A
     *            Square matrix
     * @param order
     *            ordering option (0 to 3); 0: natural ordering, 1: amd(A+A'),
     *            2: amd(S'*S), 3: amd(A'*A)
     * @param checkIfSingular
     *            if true, then the singularity test (based on
     *            Dulmage-Mendelsohn decomposition) is performed.
     * @throws IllegalArgumentException
     *             if <tt>A</tt> is not square or is not sparse.
     * @throws IllegalArgumentException
     *             if <tt>order</tt> is not in [0,3]
     */
    public SparseDoubleLUDecomposition(DoubleMatrix2D A, int order, boolean checkIfSingular) {
        DoubleProperty.DEFAULT.checkSquare(A);
        DoubleProperty.DEFAULT.checkSparse(A);

        if (order < 0 || order > 3) {
            throw new IllegalArgumentException("order must be a number between 0 and 3");
        }
        Dcs dcs;
        if (A instanceof SparseRCDoubleMatrix2D) {
            rcMatrix = true;
            dcs = ((SparseRCDoubleMatrix2D) A).getColumnCompressed().elements();
        } else {
            dcs = (Dcs) A.elements();
        }
        n = A.rows();

        S = Dcs_sqr.cs_sqr(order, dcs, false);
        if (S == null) {
            throw new IllegalArgumentException("Exception occured in cs_sqr()");
        }
        N = Dcs_lu.cs_lu(dcs, S, 1);
        if (N == null) {
            throw new IllegalArgumentException("Exception occured in cs_lu()");
        }
        if (checkIfSingular) {
            Dcsd D = Dcs_dmperm.cs_dmperm(dcs, 1); /* check if matrix is singular */
            if (D != null && D.rr[3] < n) {
                isNonSingular = false;
            }
        }
    }

    /**
     * Returns the determinant, <tt>det(A)</tt>.
     * 
     */
    public double det() {
        if (!isNonsingular())
            return 0; // avoid rounding errors
        int pivsign = 1;
        for (int i = 0; i < n; i++) {
            if (N.pinv[i] != i) {
                pivsign = -pivsign;
            }
        }
        if (U == null) {
            U = new SparseCCDoubleMatrix2D(N.U);
            if (rcMatrix) {
                U = ((SparseCCDoubleMatrix2D) U).getRowCompressed();
            }
        }
        double det = pivsign;
        for (int j = 0; j < n; j++) {
            det *= U.getQuick(j, j);
        }
        return det;
    }

    /**
     * Returns the lower triangular factor, <tt>L</tt>.
     * 
     * @return <tt>L</tt>
     */
    public DoubleMatrix2D getL() {
        if (L == null) {
            L = new SparseCCDoubleMatrix2D(N.L);
            if (rcMatrix) {
                L = ((SparseCCDoubleMatrix2D) L).getRowCompressed();
            }
        }
        return L.copy();
    }

    /**
     * Returns a copy of the pivot permutation vector.
     * 
     * @return piv
     */
    public int[] getPivot() {
        if (N.pinv == null)
            return null;
        int[] pinv = new int[N.pinv.length];
        System.arraycopy(N.pinv, 0, pinv, 0, pinv.length);
        return pinv;
    }

    /**
     * Returns the upper triangular factor, <tt>U</tt>.
     * 
     * @return <tt>U</tt>
     */
    public DoubleMatrix2D getU() {
        if (U == null) {
            U = new SparseCCDoubleMatrix2D(N.U);
            if (rcMatrix) {
                U = ((SparseCCDoubleMatrix2D) U).getRowCompressed();
            }
        }
        return U.copy();
    }

    /**
     * Returns a copy of the symbolic LU analysis object
     * 
     * @return symbolic LU analysis
     */
    public Dcss getSymbolicAnalysis() {
        Dcss S2 = new Dcss();
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
     * Returns whether the matrix is nonsingular (has an inverse).
     * 
     * @return true if <tt>U</tt>, and hence <tt>A</tt>, is nonsingular; false
     *         otherwise.
     */
    public boolean isNonsingular() {
        return isNonSingular;
    }

    /**
     * Solves <tt>A*x = b</tt>(in-place). Upon return <tt>b</tt> is overridden
     * with the result <tt>x</tt>.
     * 
     * @param b
     *            A vector with of size A.rows();
     * @exception IllegalArgumentException
     *                if <tt>b.size() != A.rows()</tt> or if A is singular.
     */
    public void solve(DoubleMatrix1D b) {
        if (b.size() != n) {
            throw new IllegalArgumentException("b.size() != A.rows()");
        }
        if (!isNonsingular()) {
            throw new IllegalArgumentException("A is singular");
        }
        DoubleProperty.DEFAULT.checkDense(b);
        double[] y = new double[n];
        double[] x;
        if (b.isView()) {
            x = (double[]) b.copy().elements();
        } else {
            x = (double[]) b.elements();
        }
        Dcs_ipvec.cs_ipvec(N.pinv, x, y, n); /* y = b(p) */
        Dcs_lsolve.cs_lsolve(N.L, y); /* y = L\y */
        Dcs_usolve.cs_usolve(N.U, y); /* y = U\y */
        Dcs_ipvec.cs_ipvec(S.q, y, x, n); /* b(q) = x */

        if (b.isView()) {
            b.assign(x);
        }
    }
}
