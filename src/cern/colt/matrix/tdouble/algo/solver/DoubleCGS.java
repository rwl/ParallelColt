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

package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

/**
 * Conjugate Gradients squared solver. CGS solves the unsymmetric linear system
 * <code>Ax = b</code> using the Conjugate Gradient Squared method
 * 
 * @author Templates
 */
public class DoubleCGS extends AbstractDoubleIterativeSolver {

    /**
     * Vectors for use in the iterative solution process
     */
    private DoubleMatrix1D p, q, u, phat, qhat, vhat, uhat, sum, r, rtilde;

    /**
     * Constructor for CGS. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     */
    public DoubleCGS(DoubleMatrix1D template) {
        p = template.copy();
        q = template.copy();
        u = template.copy();
        phat = template.copy();
        qhat = template.copy();
        vhat = template.copy();
        uhat = template.copy();
        sum = template.copy();
        r = template.copy();
        rtilde = template.copy();
    }

    public DoubleMatrix1D solve(DoubleMatrix2D A, DoubleMatrix1D b, DoubleMatrix1D x)
            throws IterativeSolverDoubleNotConvergedException {
        checkSizes(A, b, x);

        double rho_1 = 0, rho_2 = 0, alpha = 0, beta = 0;
        A.zMult(x, r.assign(b), -1, 1, false);
        rtilde.assign(r);

        for (iter.setFirst(); !iter.converged(r, x); iter.next()) {
            rho_1 = rtilde.zDotProduct(r);

            if (rho_1 == 0)
                throw new IterativeSolverDoubleNotConvergedException(DoubleNotConvergedException.Reason.Breakdown,
                        "rho", iter);

            if (iter.isFirst()) {
                u.assign(r);
                p.assign(u);
            } else {
                beta = rho_1 / rho_2;
                u.assign(r).assign(q, DoubleFunctions.plusMultSecond(beta));
                sum.assign(q).assign(p, DoubleFunctions.plusMultSecond(beta));
                p.assign(u).assign(sum, DoubleFunctions.plusMultSecond(beta));
            }

            M.apply(p, phat);
            A.zMult(phat, vhat);
            alpha = rho_1 / rtilde.zDotProduct(vhat);
            q.assign(vhat, DoubleFunctions.multSecond(-alpha)).assign(u, DoubleFunctions.plus);

            M.apply(sum.assign(u).assign(q, DoubleFunctions.plus), uhat);
            x.assign(uhat, DoubleFunctions.plusMultSecond(alpha));
            A.zMult(uhat, qhat);
            r.assign(qhat, DoubleFunctions.plusMultSecond(-alpha));
            rho_2 = rho_1;
        }

        return x;
    }

}