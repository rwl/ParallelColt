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
 * Iterative Refinement. IR solves the unsymmetric linear system
 * <code>Ax = b</code> using Iterative Refinement (preconditioned Richardson
 * iteration).
 * 
 * @author Templates
 */
public class FloatIR extends AbstractFloatIterativeSolver {

    /**
     * Vectors for use in the iterative solution process
     */
    private FloatMatrix1D z, r;

    /**
     * Constructor for IR. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     */
    public FloatIR(FloatMatrix1D template) {
        z = template.copy();
        r = template.copy();
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        checkSizes(A, b, x);

        A.zMult(x, r.assign(b), -1, 1, false);

        for (iter.setFirst(); !iter.converged(r, x); iter.next()) {
            M.apply(r, z);
            x.assign(z, FloatFunctions.plus);
            A.zMult(x, r.assign(b), -1, 1, false);
        }

        return x;
    }

}
