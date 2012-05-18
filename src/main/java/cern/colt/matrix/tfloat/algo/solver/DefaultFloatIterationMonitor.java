/*
 * Copyright (C) 2003-2006 Bjørn-Ove Heimsund
 * 
 * This file is part of MTJ.
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation; either version 2.1 of the License, or (at your
 * option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.FloatMatrix1D;

/**
 * Default iteration monitor. This tester checks declares convergence if the
 * absolute value of the residual norm is sufficiently small, or if the relative
 * decrease is small. Divergence is decleared if too many iterations are spent,
 * or the residual has grown too much. NaNs will also cause divergence to be
 * flagged.
 */
public class DefaultFloatIterationMonitor extends AbstractFloatIterationMonitor {

    /**
     * Initial residual
     */
    float initR;

    /**
     * Relative tolerance
     */
    float rtol;

    /**
     * Absolute tolerance
     */
    float atol;

    /**
     * Divergence tolerance
     */
    float dtol;

    /**
     * Maximum number of iterations
     */
    int maxIter;

    /**
     * Constructor for DefaultIterationMonitor
     * 
     * @param maxIter
     *            Maximum number of iterations
     * @param rtol
     *            Relative convergence tolerance (to initial residual)
     * @param atol
     *            Absolute convergence tolerance
     * @param dtol
     *            Relative divergence tolerance (to initial residual)
     */
    public DefaultFloatIterationMonitor(int maxIter, float rtol, float atol, float dtol) {
        this.maxIter = maxIter;
        this.rtol = rtol;
        this.atol = atol;
        this.dtol = dtol;
    }

    /**
     * Constructor for DefaultIterationMonitor. Default is 100000 iterations at
     * most, relative tolerance of 1e-3, absolute tolerance of 1e-25 and a
     * divergence tolerance of 1e+3.
     */
    public DefaultFloatIterationMonitor() {
        this.maxIter = 100000;
        this.rtol = 1e-3f;
        this.atol = 1e-25f;
        this.dtol = 1e+3f;
    }

    /**
     * Sets maximum number of iterations to permit
     * 
     * @param maxIter
     *            Maximum number of iterations
     */
    public void setMaxIterations(int maxIter) {
        this.maxIter = maxIter;
    }

    /**
     * Returns maximum number of iterations to permit
     */
    public int getMaxIterations() {
        return this.maxIter;
    }

    /**
     * Sets the relative convergence tolerance
     * 
     * @param rtol
     *            relative convergence tolerance (to initial residual)
     */
    public void setRelativeTolerance(float rtol) {
        this.rtol = rtol;
    }

    /**
     * Returns the relative convergence tolerance
     * 
     * @return relative convergence tolerance (to initial residual)
     */
    public float getRelativeTolerance() {
        return rtol;
    }

    /**
     * Sets the absolute convergence tolerance
     * 
     * @param atol
     *            absolute convergence tolerance
     */
    public void setAbsoluteTolerance(float atol) {
        this.atol = atol;
    }

    /**
     * Returns the absolute convergence tolerance
     * 
     * @return absolute convergence tolerance
     */
    public float getAbsoluteTolerance() {
        return atol;
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

    protected boolean convergedI(float r) throws IterativeSolverFloatNotConvergedException {
        // Store initial residual
        if (isFirst())
            initR = r;

        // Check for convergence
        if (r < Math.max(rtol * initR, atol))
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
