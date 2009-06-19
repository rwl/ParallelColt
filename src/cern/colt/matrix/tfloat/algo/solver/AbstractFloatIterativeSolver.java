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

import java.util.Vector;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatIdentity;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatPreconditioner;

/**
 * Partial implementation of an iterative solver
 */
public abstract class AbstractFloatIterativeSolver implements FloatIterativeSolver {

    /**
     * Preconditioner to use
     */
    protected FloatPreconditioner M;

    /**
     * Iteration monitor
     */
    protected FloatIterationMonitor iter;

    /**
     * Constructor for AbstractIterativeSolver. Does not use preconditioning,
     * and uses the default linear iteration object.
     */
    public AbstractFloatIterativeSolver() {
        M = new FloatIdentity();
        iter = new DefaultFloatIterationMonitor();
    }

    public void setPreconditioner(FloatPreconditioner M) {
        this.M = M;
    }

    public FloatPreconditioner getPreconditioner() {
        return M;
    }

    public FloatIterationMonitor getIterationMonitor() {
        return iter;
    }

    public void setIterationMonitor(FloatIterationMonitor iter) {
        this.iter = iter;
    }

    /**
     * Checks sizes of input data for {@link #solve(Matrix, Vector, Vector)}.
     * Throws an exception if the sizes does not match.
     */
    protected void checkSizes(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not square");
        if (b.size() != A.rows())
            throw new IllegalArgumentException("b.size() != A.rows()");
        if (b.size() != x.size())
            throw new IllegalArgumentException("b.size() != x.size()");
    }
}
