package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.DoubleMatrix1D;

public class HyBRDoubleIterationMonitor extends AbstractDoubleIterationMonitor {

    protected enum HyBRStoppingCondition {
        FLAT_GCV_CURVE, MIN_OF_GCV_CURVE_WITHIN_WINDOW_OF_4_ITERATIONS, PERFORMED_MAX_NUMBER_OF_ITERATIONS
    }

    protected HyBRStoppingCondition stoppingCondition;
    protected int maxIter;
    protected double dtol;
    protected double initR;
    protected double regularizationParameter;

    /**
     * Constructor for HyBRDoubleIterationMonitor. Default is 100 iterations at
     * most, and a divergence tolerance of 1e+5.
     */
    public HyBRDoubleIterationMonitor() {
        this.maxIter = 100;
        this.dtol = 1e+5;
        this.stoppingCondition = HyBRStoppingCondition.PERFORMED_MAX_NUMBER_OF_ITERATIONS;
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
        this.stoppingCondition = HyBRStoppingCondition.PERFORMED_MAX_NUMBER_OF_ITERATIONS;
    }

    public boolean converged(double r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, x, iter);
        }
        this.residual = r;
        return convergedI(r, x);
    }

    public boolean converged(double r) throws IterativeSolverDoubleNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, iter);
        }
        this.residual = r;
        return convergedI(r);
    }

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

    protected boolean convergedI(double r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
        return convergedI(r);
    }

    public int getMaxIterations() {
        return maxIter;
    }

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

    /**
     * Returns the regularization parameter
     * 
     * @return regularization parameter
     */
    public double getRegularizationParameter() {
        return regularizationParameter;
    }

    /**
     * Sets the regularization parameter
     * 
     * @param regularizationParameter
     *            regularization parameter
     */
    public void setRegularizationParameter(double regularizationParameter) {
        this.regularizationParameter = regularizationParameter;
    }

    /**
     * Sets the stopping condition
     * 
     * @param stoppingCondition
     *            stopping condition
     */
    public void setStoppingCondition(HyBRStoppingCondition stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
    }

    /**
     * Returns the stopping condition
     * 
     * @return stopping condition
     */
    public HyBRStoppingCondition getStoppingCondition() {
        return stoppingCondition;
    }

    public int iterations() {
        return Math.min(iter, maxIter);
    }

}
