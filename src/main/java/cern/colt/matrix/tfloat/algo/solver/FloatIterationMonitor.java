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

import cern.colt.matrix.Norm;
import cern.colt.matrix.tfloat.FloatMatrix1D;

/**
 * Monitors the iterative solution process for convergence and divergence. Can
 * also report the current progress.
 */
public interface FloatIterationMonitor {

    /**
     * Resets the iteration
     */
    void setFirst();

    /**
     * Returns true for the first iteration
     */
    boolean isFirst();

    /**
     * Increases iteration counter
     */
    void next();

    /**
     * Number of iterations performed
     */
    int iterations();

    /**
     * Returns current residual
     */
    float residual();

    /**
     * Checks for convergence
     * 
     * @param r
     *            Residual-vector
     * @param x
     *            State-vector
     * @return True if converged
     */
    boolean converged(FloatMatrix1D r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException;

    /**
     * Checks for convergence
     * 
     * @param r
     *            Residual-norm
     * @param x
     *            State-vector
     * @return True if converged
     */
    boolean converged(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException;

    /**
     * Checks for convergence
     * 
     * @param r
     *            Residual-norm
     * @return True if converged
     */
    boolean converged(float r) throws IterativeSolverFloatNotConvergedException;

    /**
     * Checks for convergence
     * 
     * @param r
     *            Residual-vector
     * @return True if converged
     */
    boolean converged(FloatMatrix1D r) throws IterativeSolverFloatNotConvergedException;

    /**
     * Sets new iteration reporter
     */
    void setIterationReporter(FloatIterationReporter monitor);

    /**
     * Returns current iteration reporter
     */
    FloatIterationReporter getIterationReporter();

    /**
     * Sets the vector-norm to calculate with
     */
    void setNormType(Norm normType);

    /**
     * Returns the vector-norm in use
     */
    Norm getNormType();

    /**
     * Sets maximum number of iterations to permit
     * 
     * @param maxIter
     *            Maximum number of iterations
     */
    public void setMaxIterations(int maxIter);

    /**
     * Returns the maximum number of iterations
     */
    public int getMaxIterations();
}
