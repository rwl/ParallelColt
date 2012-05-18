package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.FloatMatrix1D;

public class MRNSDFloatIterationMonitor extends DefaultFloatIterationMonitor {

    protected boolean convergedI(float r) throws IterativeSolverFloatNotConvergedException {
        // Store initial residual
        if (isFirst())
            initR = r;

        if (r <= rtol)
            return true;

        // Check for divergence
        if (r > dtol * initR)
            throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Divergence, this);
        if (iter >= maxIter)
            throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Iterations, this);
        if (Float.isNaN(r))
            throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Divergence, this);

        // Neither convergence nor divergence
        return false;
    }

    protected boolean convergedI(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
        return convergedI(r);
    }
}
