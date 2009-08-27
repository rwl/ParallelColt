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

import cern.colt.matrix.Norm;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatPreconditioner;
import cern.jet.math.tfloat.FloatFunctions;

/**
 * Quasi-Minimal Residual method. QMR solves the unsymmetric linear system
 * <code>Ax = b</code> using the Quasi-Minimal Residual method. QMR uses two
 * preconditioners, and by default these are the same preconditioner.
 * 
 * @author Templates
 */
public class FloatQMR extends AbstractFloatIterativeSolver {

    /**
     * Left preconditioner
     */
    private FloatPreconditioner M1;

    /**
     * Right preconditioner
     */
    private FloatPreconditioner M2;

    /**
     * Vectors for use in the iterative solution process
     */
    private FloatMatrix1D r, y, z, v, w, p, q, d, s, v_tld, w_tld, y_tld, z_tld, p_tld;

    /**
     * Constructor for QMR. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     */
    public FloatQMR(FloatMatrix1D template) {
        M1 = M;
        M2 = M;
        r = template.copy();
        y = template.copy();
        z = template.copy();
        v = template.copy();
        w = template.copy();
        p = template.copy();
        q = template.copy();
        d = template.copy();
        s = template.copy();
        v_tld = template.copy();
        w_tld = template.copy();
        y_tld = template.copy();
        z_tld = template.copy();
        p_tld = template.copy();
    }

    /**
     * Constructor for QMR. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified. Allows setting different
     * right and left preconditioners
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     * @param M1
     *            Left preconditioner
     * @param M2
     *            Right preconditioner
     */
    public FloatQMR(FloatMatrix1D template, FloatPreconditioner M1, FloatPreconditioner M2) {
        this.M1 = M1;
        this.M2 = M2;
        r = template.copy();
        y = template.copy();
        z = template.copy();
        v = template.copy();
        w = template.copy();
        p = template.copy();
        q = template.copy();
        d = template.copy();
        s = template.copy();
        v_tld = template.copy();
        w_tld = template.copy();
        y_tld = template.copy();
        z_tld = template.copy();
        p_tld = template.copy();
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        checkSizes(A, b, x);

        float rho = 0, rho_1 = 0, xi = 0, gamma = 1.f, gamma_1 = 0, theta = 0, theta_1 = 0, eta = -1.f, delta = 0, ep = 0, beta = 0;

        A.zMult(x, r.assign(b), -1, 1, false);

        v_tld.assign(r);
        M1.apply(v_tld, y);
        rho = DenseFloatAlgebra.DEFAULT.norm(y, Norm.Two);

        w_tld.assign(r);
        M2.transApply(w_tld, z);
        xi = DenseFloatAlgebra.DEFAULT.norm(z, Norm.Two);

        for (iter.setFirst(); !iter.converged(r, x); iter.next()) {

            if (rho == 0)
                throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Breakdown, "rho",
                        iter);

            if (xi == 0)
                throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Breakdown, "xi",
                        iter);

            v.assign(v_tld, FloatFunctions.multSecond(1 / rho));
            y.assign(FloatFunctions.mult(1 / rho));
            w.assign(w_tld, FloatFunctions.multSecond(1 / xi));
            z.assign(FloatFunctions.mult(1 / xi));

            delta = z.zDotProduct(y);

            if (delta == 0)
                throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Breakdown,
                        "delta", iter);

            M2.apply(y, y_tld);
            M1.transApply(z, z_tld);

            if (iter.isFirst()) {
                p.assign(y_tld);
                q.assign(z_tld);
            } else {
                p.assign(y_tld, FloatFunctions.plusMultFirst(-xi * delta / ep));
                q.assign(z_tld, FloatFunctions.plusMultFirst(-rho * delta / ep));
            }

            A.zMult(p, p_tld);

            ep = q.zDotProduct(p_tld);

            if (ep == 0)
                throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Breakdown, "ep",
                        iter);

            beta = ep / delta;

            if (beta == 0)
                throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Breakdown,
                        "beta", iter);

            v_tld.assign(v, FloatFunctions.multSecond(-beta)).assign(p_tld, FloatFunctions.plus);
            M1.apply(v_tld, y);
            rho_1 = rho;
            rho = DenseFloatAlgebra.DEFAULT.norm(y, Norm.Two);

            A.zMult(q, w_tld.assign(w, FloatFunctions.multSecond(-beta)), 1, 1, true);

            M2.transApply(w_tld, z);
            xi = DenseFloatAlgebra.DEFAULT.norm(z, Norm.Two);

            gamma_1 = gamma;
            theta_1 = theta;
            theta = rho / (gamma_1 * beta);
            gamma = 1 / (float) Math.sqrt(1 + theta * theta);

            if (gamma == 0)
                throw new IterativeSolverFloatNotConvergedException(FloatNotConvergedException.Reason.Breakdown,
                        "gamma", iter);

            eta = -eta * rho_1 * gamma * gamma / (beta * gamma_1 * gamma_1);

            if (iter.isFirst()) {
                d.assign(p, FloatFunctions.multSecond(eta));
                s.assign(p_tld, FloatFunctions.multSecond(eta));
            } else {
                float val = theta_1 * theta_1 * gamma * gamma;
                d.assign(FloatFunctions.mult(val)).assign(p, FloatFunctions.plusMultSecond(eta));
                s.assign(FloatFunctions.mult(val)).assign(p_tld, FloatFunctions.plusMultSecond(eta));

            }

            x.assign(d, FloatFunctions.plus);
            r.assign(s, FloatFunctions.minus);
        }

        return x;
    }

    public void setPreconditioner(FloatPreconditioner M) {
        super.setPreconditioner(M);
        M1 = M;
        M2 = M;
    }

}
