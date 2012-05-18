package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.FloatMatrix1D;

public class HyBRFloatIterationMonitor extends AbstractFloatIterationMonitor {

    protected enum HyBRStoppingCondition {
        FLAT_GCV_CURVE, MIN_OF_GCV_CURVE_WITHIN_WINDOW_OF_4_ITERATIONS, PERFORMED_MAX_NUMBER_OF_ITERATIONS
    }

    protected HyBRStoppingCondition stoppingCondition;
    protected int maxIter;
    protected float dtol;
    protected float initR;
    protected float regularizationParameter;

    /**
     * Constructor for HyBRFloatIterationMonitor. Default is 100 iterations at
     * most, and a divergence tolerance of 1e+5.
     */
    public HyBRFloatIterationMonitor() {
        this.maxIter = 100;
        this.dtol = 1e+5f;
        this.stoppingCondition = HyBRStoppingCondition.PERFORMED_MAX_NUMBER_OF_ITERATIONS;
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
        this.stoppingCondition = HyBRStoppingCondition.PERFORMED_MAX_NUMBER_OF_ITERATIONS;
    }

    public boolean converged(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, x, iter);
        }
        this.residual = r;
        return convergedI(r, x);
    }

    public boolean converged(float r) throws IterativeSolverFloatNotConvergedException {
        if (!isFirst()) {
            reporter.monitor(r, iter);
        }
        this.residual = r;
        return convergedI(r);
    }

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

    protected boolean convergedI(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
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

    /**
     * Returns the regularization parameter
     * 
     * @return regularization parameter
     */
    public float getRegularizationParameter() {
        return regularizationParameter;
    }

    /**
     * Sets the regularization parameter
     * 
     * @param regularizationParameter
     *            regularization parameter
     */
    public void setRegularizationParameter(float regularizationParameter) {
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
