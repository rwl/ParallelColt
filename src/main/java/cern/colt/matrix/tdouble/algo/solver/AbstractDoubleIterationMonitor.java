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

import cern.colt.matrix.Norm;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;

/**
 * Partial implementation of an iteration reporter
 */
public abstract class AbstractDoubleIterationMonitor implements DoubleIterationMonitor {

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
    protected DoubleIterationReporter reporter;

    /**
     * Current residual
     */
    protected double residual;

    /**
     * Constructor for AbstractIterationMonitor. Default norm is the 2-norm with
     * no iteration reporting.
     */
    public AbstractDoubleIterationMonitor() {
        normType = Norm.Two;
        reporter = new NoDoubleIterationReporter();
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

    public boolean converged(DoubleMatrix1D r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
        return converged(DenseDoubleAlgebra.DEFAULT.norm(r, normType), x);
    }

    public boolean converged(double r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException {
        reporter.monitor(r, x, iter);
        this.residual = r;
        return convergedI(r, x);
    }

    public boolean converged(double r) throws IterativeSolverDoubleNotConvergedException {
        reporter.monitor(r, iter);
        this.residual = r;
        return convergedI(r);
    }

    protected abstract boolean convergedI(double r, DoubleMatrix1D x) throws IterativeSolverDoubleNotConvergedException;

    protected abstract boolean convergedI(double r) throws IterativeSolverDoubleNotConvergedException;

    public boolean converged(DoubleMatrix1D r) throws IterativeSolverDoubleNotConvergedException {
        return converged(DenseDoubleAlgebra.DEFAULT.norm(r, normType));
    }

    public Norm getNormType() {
        return normType;
    }

    public void setNormType(Norm normType) {
        this.normType = normType;
    }

    public DoubleIterationReporter getIterationReporter() {
        return reporter;
    }

    public void setIterationReporter(DoubleIterationReporter monitor) {
        this.reporter = monitor;
    }

    public double residual() {
        return residual;
    }

}
