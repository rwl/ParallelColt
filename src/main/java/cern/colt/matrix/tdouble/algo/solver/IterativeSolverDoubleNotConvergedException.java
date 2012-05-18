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

package cern.colt.matrix.tdouble.algo.solver;

/**
 * Exception for lack of convergence in a linear problem. Contains the final
 * computed residual.
 */
public class IterativeSolverDoubleNotConvergedException extends DoubleNotConvergedException {

    private static final long serialVersionUID = 1L;

    /**
     * Iteration count when this exception was thrown
     */
    private int iterations;

    /**
     * Final residual
     */
    private double r;

    /**
     * Constructor for IterativeSolverNotConvergedException
     * 
     * @param reason
     *            Reason for this exception
     * @param message
     *            A more detailed message
     * @param iter
     *            Associated iteration monitor, for extracting residual and
     *            iteration number
     */
    public IterativeSolverDoubleNotConvergedException(Reason reason, String message, DoubleIterationMonitor iter) {
        super(reason, message);
        this.r = iter.residual();
        this.iterations = iter.iterations();
    }

    /**
     * Constructor for IterativeSolverNotConvergedException
     * 
     * @param reason
     *            Reason for this exception
     * @param iter
     *            Associated iteration monitor, for extracting residual and
     *            iteration number
     */
    public IterativeSolverDoubleNotConvergedException(Reason reason, DoubleIterationMonitor iter) {
        super(reason);
        this.r = iter.residual();
        this.iterations = iter.iterations();
    }

    /**
     * Returns final computed residual
     */
    public double getResidual() {
        return r;
    }

    /**
     * Gets the number of iterations used when this exception was thrown
     */
    public int getIterations() {
        return iterations;
    }

}
