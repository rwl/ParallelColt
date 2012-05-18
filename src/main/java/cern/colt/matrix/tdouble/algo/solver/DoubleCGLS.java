package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleIdentity;
import cern.jet.math.tdouble.DoubleFunctions;

/**
 * CGLS is Conjugate Gradient for Least Squares method used for solving
 * large-scale, ill-posed inverse problems of the form: b = A*x + noise.
 * 
 * <p>
 * Reference:<br>
 * <p>
 * A. Bjorck, "Numerical Methods for Least Squares Problems" SIAM, 1996, pg.
 * 289.
 * </p>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DoubleCGLS extends AbstractDoubleIterativeSolver {

    private static final DenseDoubleAlgebra alg = DenseDoubleAlgebra.DEFAULT;
    public static final double sqrteps = Math.sqrt(Math.pow(2, -52));

    public DoubleCGLS() {
        iter = new CGLSDoubleIterationMonitor();
        ((CGLSDoubleIterationMonitor) iter).setRelativeTolerance(-1);
    }

    public DoubleMatrix1D solve(DoubleMatrix2D A, DoubleMatrix1D b, DoubleMatrix1D x)
            throws IterativeSolverDoubleNotConvergedException {
        DoubleMatrix1D p, q, r, s;
        double alpha;
        double beta;
        double gamma;
        double oldgamma = 0;
        double rnrm;
        double nq;

        if (((CGLSDoubleIterationMonitor) iter).getRelativeTolerance() == -1.0) {
            ((CGLSDoubleIterationMonitor) iter).setRelativeTolerance(sqrteps * alg.norm2(A.zMult(b, null, 1, 0, true)));
        }

        s = A.zMult(x, null);
        s.assign(b, DoubleFunctions.plusMultFirst(-1));
        r = A.zMult(s, null, 1, 0, true);
        rnrm = alg.norm2(r);
        if (!(M instanceof DoubleIdentity)) {
            r = M.transApply(r, null);
            gamma = alg.norm2(r);
            gamma *= gamma;
        } else {
            gamma = rnrm;
            gamma *= gamma;
        }
        p = r.copy();
        for (iter.setFirst(); !iter.converged(rnrm, x); iter.next()) {
            if (!iter.isFirst()) {
                beta = gamma / oldgamma;
                p.assign(r, DoubleFunctions.plusMultFirst(beta));
            }
            if (!(M instanceof DoubleIdentity)) {
                r = M.apply(p, null);
                q = A.zMult(r, null);
            } else {
                q = A.zMult(p, null);
            }
            nq = alg.norm2(q);
            nq = nq * nq;
            alpha = gamma / nq;
            if (!(M instanceof DoubleIdentity)) {
                x.assign(r, DoubleFunctions.plusMultSecond(alpha));
            } else {
                x.assign(p, DoubleFunctions.plusMultSecond(alpha));
            }
            s.assign(q, DoubleFunctions.plusMultSecond(-alpha));
            r = A.zMult(s, null, 1, 0, true);
            rnrm = alg.norm2(r);
            if (!(M instanceof DoubleIdentity)) {
                r = M.transApply(r, null);
                oldgamma = gamma;
                gamma = alg.norm2(r);
                gamma *= gamma;
            } else {
                oldgamma = gamma;
                gamma = rnrm;
                gamma *= gamma;
            }
        }
        return x;
    }

}
