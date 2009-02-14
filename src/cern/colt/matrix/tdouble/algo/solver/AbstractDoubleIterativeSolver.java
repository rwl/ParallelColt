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

import java.util.Vector;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleIdentity;
import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoublePreconditioner;

/**
 * Partial implementation of an iterative solver
 */
public abstract class AbstractDoubleIterativeSolver implements DoubleIterativeSolver {

    /**
     * Preconditioner to use
     */
    protected DoublePreconditioner M;

    /**
     * Iteration monitor
     */
    protected DoubleIterationMonitor iter;

    /**
     * Constructor for AbstractIterativeSolver. Does not use preconditioning,
     * and uses the default linear iteration object.
     */
    public AbstractDoubleIterativeSolver() {
        M = new DoubleIdentity();
        iter = new DefaultDoubleIterationMonitor();
    }

    public void setPreconditioner(DoublePreconditioner M) {
        this.M = M;
    }

    public DoublePreconditioner getPreconditioner() {
        return M;
    }

    public DoubleIterationMonitor getIterationMonitor() {
        return iter;
    }

    public void setIterationMonitor(DoubleIterationMonitor iter) {
        this.iter = iter;
    }

    /**
     * Checks sizes of input data for {@link #solve(Matrix, Vector, Vector)}.
     * Throws an exception if the sizes does not match.
     */
    protected void checkSizes(DoubleMatrix2D A, DoubleMatrix1D b, DoubleMatrix1D x) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not square");
        if (b.size() != A.rows())
            throw new IllegalArgumentException("b.size() != A.rows()");
        if (b.size() != x.size())
            throw new IllegalArgumentException("b.size() != x.size()");
    }
}
