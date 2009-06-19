package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.FloatMatrix1D;

public class HyBRFloatIterationMonitor extends AbstractFloatIterationMonitor {

    protected enum HyBRStoppingCondition {
        FLAT_GCV_CURVE, MIN_OF_GCV_CURVE_WITHIN_WINDOW_OF_4_ITERATIONS
    }

    protected HyBRStoppingCondition stoppingCondition;
    protected int maxIter;
    protected float dtol;
    protected float initR;

    /**
     * Constructor for HyBRFloatIterationMonitor. Default is 100 iterations at
     * most, and a divergence tolerance of 1e+3.
     */
    public HyBRFloatIterationMonitor() {
        this.maxIter = 100;
        this.dtol = 1e+3f;
    }

    /**
     * Constructor for HyBRFloatIterationMonitor
     * 
     * @param maxIter
     *            Maximum number of iterations
     * @param dtol
     *            Relative divergence tolerance (to initial residual)
     */
    public HyBRFloatIterationMonitor(int maxIter, float dtol) {
        this.maxIter = maxIter;
        this.dtol = dtol;
    }

    @Override
    public boolean converged(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, x, iter);
        }
        this.residual = r;
        return convergedI(r, x);
    }

    @Override
    public boolean converged(float r) throws IterativeSolverFloatNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, iter);
        }
        this.residual = r;
        return convergedI(r);
    }

    @Override
    protected boolean convergedI(float r) throws IterativeSolverFloatNotConvergedException {
        // Store initial residual
        if (isFirst())
            initR = r;

        // Check for divergence
        if (initR != -1.0) {
            if (r > dtol * initR)
                throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Divergence, this);
        }
        if (iter >= (maxIter + 1))
            throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Iterations, this);
        if (Float.isNaN(r))
            throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Divergence, this);

        // Neither convergence nor divergence
        return false;
    }

    @Override
    protected boolean convergedI(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
        return convergedI(r);
    }

    @Override
    public int getMaxIterations() {
        return maxIter;
    }

    @Override
    public void setMaxIterations(int maxIter) {
        this.maxIter = maxIter;
    }

    /**
     * Sets the relative divergence tolerance
     * 
     * @param dtol
     *            relative divergence tolerance (to initial residual)
     */
    public void setDivergenceTolerance(float dtol) {
        this.dtol = dtol;
    }

    /**
     * Returns the relative divergence tolerance
     * 
     * @return relative divergence tolerance (to initial residual)
     */
    public float getDivergenceTolerance() {
        return dtol;
    }

    public void setStoppingCondition(HyBRStoppingCondition stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
    }

    public HyBRStoppingCondition getStoppingCondition() {
        return stoppingCondition;
    }

}
