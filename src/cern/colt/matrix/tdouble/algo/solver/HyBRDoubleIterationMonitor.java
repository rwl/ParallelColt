package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.DoubleMatrix1D;

public class HyBRDoubleIterationMonitor extends AbstractDoubleIterationMonitor {

    protected enum HyBRStoppingCondition {
        FLAT_GCV_CURVE, MIN_OF_GCV_CURVE_WITHIN_WINDOW_OF_4_ITERATIONS
    }

    protected HyBRStoppingCondition stoppingCondition;
    protected int maxIter;
    protected double dtol;
    protected double initR;

    /**
     * Constructor for HyBRDoubleIterationMonitor. Default is 100 iterations at
     * most, and a divergence tolerance of 1e+5.
     */
    public HyBRDoubleIterationMonitor() {
        this.maxIter = 100;
        this.dtol = 1e+5;
    }

    /**
     * Constructor for HyBRDoubleIterationMonitor
     * 
     * @param maxIter
     *            Maximum number of iterations
     * @param dtol
     *            Relative divergence tolerance (to initial residual)
     */
    public HyBRDoubleIterationMonitor(int maxIter, double dtol) {
        this.maxIter = maxIter;
        this.dtol = dtol;
    }

    @Override
    public boolean converged(double r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, x, iter);
        }
        this.residual = r;
        return convergedI(r, x);
    }

    @Override
    public boolean converged(double r) throws IterativeSolverDoubleNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, iter);
        }
        this.residual = r;
        return convergedI(r);
    }

    @Override
    protected boolean convergedI(double r) throws IterativeSolverDoubleNotConvergedException {
        // Store initial residual
        if (isFirst())
            initR = r;

        // Check for divergence
        if (initR != -1.0) {
            if (r > dtol * initR)
                throw new IterativeSolverDoubleNotConvergedException(DoubleNotConvergedException.Reason.Divergence,
                        this);
        }
        if (iter >= (maxIter + 1))
            throw new IterativeSolverDoubleNotConvergedException(DoubleNotConvergedException.Reason.Iterations, this);
        if (Double.isNaN(r))
            throw new IterativeSolverDoubleNotConvergedException(DoubleNotConvergedException.Reason.Divergence, this);

        // Neither convergence nor divergence
        return false;
    }

    @Override
    protected boolean convergedI(double r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
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
    public void setDivergenceTolerance(double dtol) {
        this.dtol = dtol;
    }

    /**
     * Returns the relative divergence tolerance
     * 
     * @return relative divergence tolerance (to initial residual)
     */
    public double getDivergenceTolerance() {
        return dtol;
    }

    public void setStoppingCondition(HyBRStoppingCondition stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
    }

    public HyBRStoppingCondition getStoppingCondition() {
        return stoppingCondition;
    }

}
