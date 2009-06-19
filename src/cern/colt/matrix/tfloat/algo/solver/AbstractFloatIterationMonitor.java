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
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;

/**
 * Partial implementation of an iteration reporter
 */
public abstract class AbstractFloatIterationMonitor implements FloatIterationMonitor {

    /**
     * Iteration number
     */
    protected int iter;

    /**
     * Vector-norm
     */
    protected Norm normType;

    /**
     * Iteration reporter
     */
    protected FloatIterationReporter reporter;

    /**
     * Current residual
     */
    protected float residual;

    /**
     * Constructor for AbstractIterationMonitor. Default norm is the 2-norm with
     * no iteration reporting.
     */
    public AbstractFloatIterationMonitor() {
        normType = Norm.Two;
        reporter = new NoFloatIterationReporter();
    }

    public void setFirst() {
        iter = 0;
    }

    public boolean isFirst() {
        return iter == 0;
    }

    public void next() {
        iter++;
    }

    public int iterations() {
        return iter;
    }

    public boolean converged(FloatMatrix1D r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
        return converged(DenseFloatAlgebra.DEFAULT.norm(r, normType), x);
    }

    public boolean converged(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
        reporter.monitor(r, x, iter);
        this.residual = r;
        return convergedI(r, x);
    }

    public boolean converged(float r) throws IterativeSolverFloatNotConvergedException {
        reporter.monitor(r, iter);
        this.residual = r;
        return convergedI(r);
    }

    protected abstract boolean convergedI(float r, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException;

    protected abstract boolean convergedI(float r) throws IterativeSolverFloatNotConvergedException;

    public boolean converged(FloatMatrix1D r) throws IterativeSolverFloatNotConvergedException {
        return converged(DenseFloatAlgebra.DEFAULT.norm(r, normType));
    }

    public Norm getNormType() {
        return normType;
    }

    public void setNormType(Norm normType) {
        this.normType = normType;
    }

    public FloatIterationReporter getIterationReporter() {
        return reporter;
    }

    public void setIterationReporter(FloatIterationReporter monitor) {
        this.reporter = monitor;
    }

    public float residual() {
        return residual;
    }

}
