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

import cern.colt.matrix.Norm;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

/**
 * GMRES solver. GMRES solves the unsymmetric linear system <code>Ax = b</code>
 * using the Generalized Minimum Residual method. The GMRES iteration is
 * restarted after a given number of iterations. By default it is restarted
 * after 30 iterations.
 * 
 * @author Templates
 */
public class DoubleGMRES extends AbstractDoubleIterativeSolver {

    /**
     * After this many iterations, the GMRES will be restarted.
     */
    private int restart;

    /**
     * Vectors for use in the iterative solution process
     */
    private DoubleMatrix1D w, u, r;

    /**
     * Vectors spanning the subspace
     */
    private DoubleMatrix1D[] v;

    /**
     * Restart vector
     */
    private DoubleMatrix1D s;

    /**
     * Hessenberg matrix
     */
    private DoubleMatrix2D H;

    /**
     * Givens rotations for the QR factorization
     */
    private DoubleGivensRotation[] rotation;

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
    public DoubleGMRES(DoubleMatrix1D template) {
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
    public DoubleGMRES(DoubleMatrix1D template, int restart) {
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

        s = new DenseDoubleMatrix1D(restart + 1);
        H = new DenseDoubleMatrix2D(restart + 1, restart);
        rotation = new DoubleGivensRotation[restart + 1];

        v = new DoubleMatrix1D[restart + 1];
        for (int i = 0; i < v.length; ++i)
            v[i] = new DenseDoubleMatrix1D((int) r.size());
    }

    public DoubleMatrix1D solve(DoubleMatrix2D A, DoubleMatrix1D b, DoubleMatrix1D x)
            throws IterativeSolverDoubleNotConvergedException {
        checkSizes(A, b, x);

        A.zMult(x, u.assign(b), -1, 1, false);
        M.apply(u, r);
        double normr = DenseDoubleAlgebra.DEFAULT.norm(r, Norm.Two);
        M.apply(b, u);

        // Outer iteration
        for (iter.setFirst(); !iter.converged(r, x); iter.next()) {

            v[0].assign(r, DoubleFunctions.multSecond(1 / normr));
            s.assign(0).setQuick(0, normr);
            int i = 0;

            // Inner iteration
            for (; i < restart && !iter.converged(Math.abs(s.getQuick(i))); i++, iter.next()) {
                A.zMult(v[i], u);
                M.apply(u, w);

                for (int k = 0; k <= i; k++) {
                    H.setQuick(k, i, w.zDotProduct(v[k]));
                    w.assign(v[k], DoubleFunctions.plusMultSecond(-H.getQuick(k, i)));
                }
                H.setQuick(i + 1, i, DenseDoubleAlgebra.DEFAULT.norm(w, Norm.Two));
                v[i + 1].assign(w, DoubleFunctions.multSecond(1. / H.getQuick(i + 1, i)));

                // QR factorization of H using Givens rotations
                for (int k = 0; k < i; ++k)
                    rotation[k].apply(H, i, k, k + 1);

                rotation[i] = new DoubleGivensRotation(H.getQuick(i, i), H.getQuick(i + 1, i));
                rotation[i].apply(H, i, i, i + 1);
                rotation[i].apply(s, i, i + 1);
            }

            // Update solution in current subspace
            s = DenseDoubleAlgebra.DEFAULT.backwardSolve(H.viewPart(0, 0, i, i), s);
            for (int j = 0; j < i; j++)
                x.assign(v[j], DoubleFunctions.plusMultSecond(s.getQuick(j)));

            A.zMult(x, u.assign(b), -1, 1, false);
            M.apply(u, r);
            normr = DenseDoubleAlgebra.DEFAULT.norm(r, Norm.Two);
        }

        return x;
    }

}
