package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatIdentity;
import cern.jet.math.tfloat.FloatFunctions;

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
public class FloatCGLS extends AbstractFloatIterativeSolver {

    private static final DenseFloatAlgebra alg = DenseFloatAlgebra.DEFAULT;
    public static final float sqrteps = (float) Math.sqrt(Math.pow(2, -52));

    public FloatCGLS() {
        iter = new CGLSFloatIterationMonitor();
        ((CGLSFloatIterationMonitor) iter).setRelativeTolerance(-1);
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        FloatMatrix1D p, q, r, s;
        float alpha;
        float beta;
        float gamma;
        float oldgamma = 0;
        float rnrm;
        float nq;

        if (((CGLSFloatIterationMonitor) iter).getRelativeTolerance() == -1.0) {
            ((CGLSFloatIterationMonitor) iter).setRelativeTolerance(sqrteps * alg.norm2(A.zMult(b, null, 1, 0, true)));
        }

        s = A.zMult(x, null);
        s.assign(b, FloatFunctions.plusMultFirst(-1));
        r = A.zMult(s, null, 1, 0, true);
        rnrm = alg.norm2(r);
        if (!(M instanceof FloatIdentity)) {
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
                p.assign(r, FloatFunctions.plusMultFirst(beta));
            }
            if (!(M instanceof FloatIdentity)) {
                r = M.apply(p, null);
                q = A.zMult(r, null);
            } else {
                q = A.zMult(p, null);
            }
            nq = alg.norm2(q);
            nq = nq * nq;
            alpha = gamma / nq;
            if (!(M instanceof FloatIdentity)) {
                x.assign(r, FloatFunctions.plusMultSecond(alpha));
            } else {
                x.assign(p, FloatFunctions.plusMultSecond(alpha));
            }
            s.assign(q, FloatFunctions.plusMultSecond(-alpha));
            r = A.zMult(s, null, 1, 0, true);
            rnrm = alg.norm2(r);
            if (!(M instanceof FloatIdentity)) {
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
