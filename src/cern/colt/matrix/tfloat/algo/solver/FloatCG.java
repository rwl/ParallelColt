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
 * Conjugate Gradients solver. CG solves the symmetric positive definite linear
 * system <code>Ax=b</code> using the Conjugate Gradient method.
 * 
 * @author Templates
 */
public class FloatCG extends AbstractFloatIterativeSolver {

    /**
     * Vectors for use in the iterative solution process
     */
    private FloatMatrix1D p, z, q, r;

    /**
     * Constructor for CG. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     */
    public FloatCG(FloatMatrix1D template) {
        p = template.copy();
        z = template.copy();
        q = template.copy();
        r = template.copy();
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        checkSizes(A, b, x);

        float alpha = 0, beta = 0, rho = 0, rho_1 = 0;

        A.zMult(x, r.assign(b), -1, 1, false);

        for (iter.setFirst(); !iter.converged(r, x); iter.next()) {
            M.apply(r, z);
            rho = r.zDotProduct(z);

            if (iter.isFirst())
                p.assign(z);
            else {
                beta = rho / rho_1;
                p.assign(z, FloatFunctions.plusMultFirst(beta));
            }

            A.zMult(p, q);
            alpha = rho / p.zDotProduct(q);

            x.assign(p, FloatFunctions.plusMultSecond(alpha));
            r.assign(q, FloatFunctions.plusMultSecond(-alpha));

            rho_1 = rho;
        }
        return x;
    }

}
