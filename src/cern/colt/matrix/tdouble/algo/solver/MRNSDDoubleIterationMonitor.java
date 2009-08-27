package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.DoubleMatrix1D;

public class MRNSDDoubleIterationMonitor extends DefaultDoubleIterationMonitor {

    protected boolean convergedI(double r) throws IterativeSolverDoubleNotConvergedException {
        // Store initial residual
        if (isFirst())
            initR = r;

        if (r <= rtol)
            return true;

        // Check for divergence
        if (r > dtol * initR)
            throw new IterativeSolverDoubleNotConvergedException(DoubleNotConvergedException.Reason.Divergence, this);
        if (iter >= maxIter)
            throw new IterativeSolverDoubleNotConvergedException(DoubleNotConvergedException.Reason.Iterations, this);
        if (Double.isNaN(r))
            throw new IterativeSolverDoubleNotConvergedException(DoubleNotConvergedException.Reason.Divergence, this);

        // Neither convergence nor divergence
        return false;
    }

    protected boolean convergedI(double r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
        return convergedI(r);
    }
}
