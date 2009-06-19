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

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoublePreconditioner;

/**
 * Iterative linear solver. Solves <code>Ax=b</code> for <code>x</code>, and it
 * supports preconditioning and convergence monitoring.
 */
public interface DoubleIterativeSolver {

    /**
     * Solves the given problem, writing result into the vector.
     * 
     * @param A
     *            Matrix of the problem
     * @param b
     *            Right hand side
     * @param x
     *            Solution is stored here. Also used as initial guess
     * @return The solution vector x
     */
    DoubleMatrix1D solve(DoubleMatrix2D A, DoubleMatrix1D b, DoubleMatrix1D x)
            throws IterativeSolverDoubleNotConvergedException;

    /**
     * Sets preconditioner
     * 
     * @param M
     *            Preconditioner to use
     */
    void setPreconditioner(DoublePreconditioner M);

    /**
     * Gets preconditioner
     * 
     * @return Current preconditioner
     */
    DoublePreconditioner getPreconditioner();

    /**
     * Sets iteration monitor
     * 
     * @param iter
     *            Iteration monitor
     */
    void setIterationMonitor(DoubleIterationMonitor iter);

    /**
     * Gets the iteration monitor
     * 
     * @return Current iteration monitor
     */
    DoubleIterationMonitor getIterationMonitor();

}
