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
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import cern.jet.math.tfloat.FloatFunctions;

/**
 * GMRES solver. GMRES solves the unsymmetric linear system <code>Ax = b</code>
 * using the Generalized Minimum Residual method. The GMRES iteration is
 * restarted after a given number of iterations. By default it is restarted
 * after 30 iterations.
 * 
 * @author Templates
 */
public class FloatGMRES extends AbstractFloatIterativeSolver {

    /**
     * After this many iterations, the GMRES will be restarted.
     */
    private int restart;

    /**
     * Vectors for use in the iterative solution process
     */
    private FloatMatrix1D w, u, r;

    /**
     * Vectors spanning the subspace
     */
    private FloatMatrix1D[] v;

    /**
     * Restart vector
     */
    private FloatMatrix1D s;

    /**
     * Hessenberg matrix
     */
    private FloatMatrix2D H;

    /**
     * Givens rotations for the QR factorization
     */
    private FloatGivensRotation[] rotation;

    /**
     * Constructor for GMRES. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified. The iteration is
     * restarted every 30 iterations
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     */
    public FloatGMRES(FloatMatrix1D template) {
        this(template, 30);
    }

    /**
     * Constructor for GMRES. Uses the given vector as template for creating
     * scratch vectors. Typically, the solution or the right hand side vector
     * can be passed, and the template is not modified
     * 
     * @param template
     *            Vector to use as template for the work vectors needed in the
     *            solution process
     * @param restart
     *            GMRES iteration is restarted after this number of iterations
     */
    public FloatGMRES(FloatMatrix1D template, int restart) {
        w = template.copy();
        u = template.copy();
        r = template.copy();
        setRestart(restart);
    }

    /**
     * Sets the restart parameter
     * 
     * @param restart
     *            GMRES iteration is restarted after this number of iterations
     */
    public void setRestart(int restart) {
        this.restart = restart;
        if (restart <= 0)
            throw new IllegalArgumentException("restart must be a positive integer");

        s = new DenseFloatMatrix1D(restart + 1);
        H = new DenseFloatMatrix2D(restart + 1, restart);
        rotation = new FloatGivensRotation[restart + 1];

        v = new FloatMatrix1D[restart + 1];
        for (int i = 0; i < v.length; ++i)
            v[i] = new DenseFloatMatrix1D((int) r.size());
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        checkSizes(A, b, x);

        A.zMult(x, u.assign(b), -1, 1, false);
        M.apply(u, r);
        float normr = DenseFloatAlgebra.DEFAULT.norm(r, Norm.Two);
        M.apply(b, u);

        // Outer iteration
        for (iter.setFirst(); !iter.converged(r, x); iter.next()) {

            v[0].assign(r, FloatFunctions.multSecond(1 / normr));
            s.assign(0).setQuick(0, normr);
            int i = 0;

            // Inner iteration
            for (; i < restart && !iter.converged(Math.abs(s.getQuick(i))); i++, iter.next()) {
                A.zMult(v[i], u);
                M.apply(u, w);

                for (int k = 0; k <= i; k++) {
                    H.setQuick(k, i, w.zDotProduct(v[k]));
                    w.assign(v[k], FloatFunctions.plusMultSecond(-H.getQuick(k, i)));
                }
                H.setQuick(i + 1, i, DenseFloatAlgebra.DEFAULT.norm(w, Norm.Two));
                v[i + 1].assign(w, FloatFunctions.multSecond(1.f / H.getQuick(i + 1, i)));

                // QR factorization of H using Givens rotations
                for (int k = 0; k < i; ++k)
                    rotation[k].apply(H, i, k, k + 1);

                rotation[i] = new FloatGivensRotation(H.getQuick(i, i), H.getQuick(i + 1, i));
                rotation[i].apply(H, i, i, i + 1);
                rotation[i].apply(s, i, i + 1);
            }

            // Update solution in current subspace
            s = DenseFloatAlgebra.DEFAULT.backwardSolve(H.viewPart(0, 0, i, i), s);
            for (int j = 0; j < i; j++)
                x.assign(v[j], FloatFunctions.plusMultSecond(s.getQuick(j)));

            A.zMult(x, u.assign(b), -1, 1, false);
            M.apply(u, r);
            normr = DenseFloatAlgebra.DEFAULT.norm(r, Norm.Two);
        }

        return x;
    }

}
