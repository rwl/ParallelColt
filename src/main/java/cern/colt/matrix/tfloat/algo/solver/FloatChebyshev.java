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
/*
 * Derived from public domain software at http://www.netlib.org/templates
 */

package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.jet.math.tfloat.FloatFunctions;

/**
 * Chebyshev solver. Solves the symmetric positive definite linear system
 * <code>Ax = b</code> using the Preconditioned Chebyshev Method. Chebyshev
 * requires an acurate estimate on the bounds of the spectrum of the matrix.
 * 
 * @author Templates
 */
public class FloatChebyshev extends AbstractFloatIterativeSolver {

    /**
     * Estimates for the eigenvalue of the matrix
     */
    private float eigmin, eigmax;

    /**
     * Vectors for use in the iterative solution process
     */
    private FloatMatrix1D p, z, r, q;

    /**
     * Constructor for Chebyshev. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified. Eigenvalue estimates
     * must also be provided
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     * @param eigmin
     *            Smallest eigenvalue. Must be positive
     * @param eigmax
     *            Largest eigenvalue. Must be positive
     */
    public FloatChebyshev(FloatMatrix1D template, float eigmin, float eigmax) {
        p = template.copy();
        z = template.copy();
        r = template.copy();
        q = template.copy();
        setEigenvalues(eigmin, eigmax);
    }

    /**
     * Sets the eigenvalue estimates.
     * 
     * @param eigmin
     *            Smallest eigenvalue. Must be positive
     * @param eigmax
     *            Largest eigenvalue. Must be positive
     */
    public void setEigenvalues(float eigmin, float eigmax) {
        this.eigmin = eigmin;
        this.eigmax = eigmax;

        if (eigmin <= 0)
            throw new IllegalArgumentException("eigmin <= 0");
        if (eigmax <= 0)
            throw new IllegalArgumentException("eigmax <= 0");
        if (eigmin > eigmax)
            throw new IllegalArgumentException("eigmin > eigmax");
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        checkSizes(A, b, x);

        float alpha = 0, beta = 0, c = 0, d = 0;

        A.zMult(x, r.assign(b), -1, 1, false);

        c = (eigmax - eigmin) / 2;
        d = (eigmax + eigmin) / 2;

        for (iter.setFirst(); !iter.converged(r, x); iter.next()) {
            M.apply(r, z);

            if (iter.isFirst()) {
                p.assign(z);
                alpha = 2.0f / d;
            } else {
                beta = (alpha * c) / 2.0f;
                beta *= beta;
                alpha = 1.0f / (d - beta);
                p.assign(z, FloatFunctions.plusMultFirst(beta));
            }

            A.zMult(p, q);
            x.assign(p, FloatFunctions.plusMultSecond(alpha));
            r.assign(q, FloatFunctions.plusMultSecond(-alpha));
        }

        return x;
    }

}
