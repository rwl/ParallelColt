package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DoubleAlgebra;
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

    private static final DoubleAlgebra alg = DoubleAlgebra.DEFAULT;
    public static final double sqrteps = (double) Math.sqrt(Math.pow(2, -52));

    public DoubleCGLS() {
        iter = new CGLSDoubleIterationMonitor();
        ((CGLSDoubleIterationMonitor) iter).setRelativeTolerance(-1);
    }

    @Override
    public DoubleMatrix1D solve(DoubleMatrix2D A, DoubleMatrix1D b, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
        DoubleMatrix1D p, q, r, s;
        double alpha;
        double beta;
        double gamma;
        double oldgamma = 0;
        double rnrm;
        double nq;

        double nrm_trAb = alg.norm2(A.zMult(b, null, 1, 0, true));
        if (((CGLSDoubleIterationMonitor) iter).getRelativeTolerance() == -1.0) {
            ((CGLSDoubleIterationMonitor) iter).setRelativeTolerance(sqrteps * nrm_trAb);
        }

        s = A.zMult(x, null);
        s.assign(b, DoubleFunctions.plusMultFirst(-1));
        r = A.zMult(s, null, 1, 0, true);
        if (!(M instanceof DoubleIdentity)) {
            r = M.transApply(r, null);
        }
        gamma = alg.norm2(r);
        rnrm = gamma;
        gamma *= gamma;
        p = r.copy();
        for (iter.setFirst(); !iter.converged(rnrm, x); iter.next()) {
            if (!iter.isFirst()) {
                beta = gamma / oldgamma;
                p.assign(r, DoubleFunctions.plusMultFirst(beta));
            }
            if (!(M instanceof DoubleIdentity)) {
                p = M.apply(p, null);
            }
            q = A.zMult(p, null);
            nq = alg.norm2(q);
            nq = nq * nq;
            alpha = gamma / nq;
            x.assign(p, DoubleFunctions.plusMultSecond(alpha));
            s.assign(q, DoubleFunctions.plusMultSecond(-alpha));
            r = A.zMult(s, null, 1, 0, true);
            if (!(M instanceof DoubleIdentity)) {
                r = M.transApply(r, null);
            }
            oldgamma = gamma;
            gamma = alg.norm2(r);
            rnrm = gamma;
            gamma *= gamma;
        }
        return x;
    }

}
