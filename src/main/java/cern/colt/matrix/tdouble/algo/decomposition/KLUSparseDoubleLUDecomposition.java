package cern.colt.matrix.tdouble.algo.decomposition;

import static edu.ufl.cise.klu.tdouble.Dklu_analyze.klu_analyze;
import static edu.ufl.cise.klu.tdouble.Dklu_defaults.klu_defaults;
import static edu.ufl.cise.klu.tdouble.Dklu_factor.klu_factor;
import static edu.ufl.cise.klu.tdouble.Dklu_solve.klu_solve;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.impl.SparseCCDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2D;

import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;
import edu.ufl.cise.klu.common.KLU_common;
import edu.ufl.cise.klu.common.KLU_numeric;
import edu.ufl.cise.klu.common.KLU_symbolic;

/**
 * LU decomposition implemented using JKLU.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @author Richard Lincoln (r.w.lincoln@gmail.com)
 */
public class KLUSparseDoubleLUDecomposition implements SparseDoubleLUDecomposition {
    private KLU_symbolic S;
    private KLU_numeric N;
    private KLU_common Common;
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
     *            ordering option (0 to 1); 0: AMD, 1: COLAMD
     * @param checkIfSingular
     *            if true, then the singularity test (based on
     *            BTFJ) is performed.
     * @param preOrder use BTF pre-ordering, or not
     * @throws IllegalArgumentException
     *             if <tt>A</tt> is not square or is not sparse.
     * @throws IllegalArgumentException
     *             if <tt>order</tt> is not in [0,1]
     */
    public KLUSparseDoubleLUDecomposition(DoubleMatrix2D A, int order, boolean checkIfSingular, boolean preOrder) {
        DoubleProperty.DEFAULT.checkSquare(A);
        DoubleProperty.DEFAULT.checkSparse(A);

        if (order < 0 || order > 3) {
            throw new IllegalArgumentException("order must be a number between 0 and 3");
        }

		Common = new KLU_common();
		klu_defaults (Common);
		Common.ordering = order;
		Common.btf = preOrder ? 1 : 0;
		
        Dcs dcs;
        if (A instanceof SparseRCDoubleMatrix2D) {
            rcMatrix = true;
            dcs = ((SparseRCDoubleMatrix2D) A).getColumnCompressed().elements();
        } else {
            dcs = (Dcs) A.elements();
        }
        n = A.rows();
        int[] Ap = dcs.p, Ai = dcs.i;
        double[] Ax = dcs.x;

        S = klu_analyze(n, Ap, Ai, Common);
        if (S == null) {
            throw new IllegalArgumentException("Exception occured in klu_analyze()");
        }
		N = klu_factor(Ap, Ai, Ax, S, Common);
        if (N == null) {
            throw new IllegalArgumentException("Exception occured in klu_factor()");
        }
        if (checkIfSingular) {
            Dcsd D = Dcs_dmperm.cs_dmperm(dcs, 1); /* check if matrix is singular */
            if (D != null && D.rr[3] < n) {
                isNonSingular = false;
            }
        }
    }
    
    public KLUSparseDoubleLUDecomposition(DoubleMatrix2D A, int order, boolean checkIfSingular) {
    	this(A, order, checkIfSingular, true);
    }

    /* (non-Javadoc)
	 * @see cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition#det()
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

    /* (non-Javadoc)
	 * @see cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition#getL()
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

    /* (non-Javadoc)
	 * @see cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition#getPivot()
	 */
    public int[] getPivot() {
        if (N.pinv == null)
            return null;
        int[] pinv = new int[N.pinv.length];
        System.arraycopy(N.pinv, 0, pinv, 0, pinv.length);
        return pinv;
    }

    /* (non-Javadoc)
	 * @see cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition#getU()
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

    /* (non-Javadoc)
	 * @see cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition#getSymbolicAnalysis()
	 */
    public KLU_symbolic getSymbolicAnalysis() {
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

    /* (non-Javadoc)
	 * @see cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition#isNonsingular()
	 */
    public boolean isNonsingular() {
        return isNonSingular;
    }

    /* (non-Javadoc)
	 * @see cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition#solve(cern.colt.matrix.tdouble.DoubleMatrix1D)
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
		klu_solve(S, N, n, 1, x, 0, Common);

        if (b.isView()) {
            b.assign(x);
        }
    }
}
