/*
 * Copyright (C) 2009 Piotr Wendykier
 * 
 * This program is free software; you can redistribute it and/or modify it
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

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatIdentity;
import cern.jet.math.tfloat.FloatFunctions;

/**
 * MRNSD is Modified Residual Norm Steepest Descent method used for solving
 * large-scale, ill-posed inverse problems of the form: b = A*x + noise. This
 * algorithm is nonnegatively constrained.
 * 
 * <p>
 * References:<br>
 * <p>
 * [1] J. Nagy, Z. Strakos,
 * "Enforcing nonnegativity in image reconstruction algorithms" in Mathematical
 * Modeling, Estimation, and Imaging, David C. Wilson, et.al., Eds., 4121
 * (2000), pg. 182--190.
 * </p>
 * <p>
 * [2] L. Kaufman, "Maximum likelihood, least squares and penalized least
 * squares for PET", IEEE Trans. Med. Imag. 12 (1993) pp. 200--214.
 * </p>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class FloatMRNSD extends AbstractFloatIterativeSolver {

    private static final DenseFloatAlgebra alg = DenseFloatAlgebra.DEFAULT;
    public static final float sqrteps = (float) Math.sqrt(Math.pow(2, -52));

    public FloatMRNSD() {
        iter = new MRNSDFloatIterationMonitor();
        ((MRNSDFloatIterationMonitor) iter).setRelativeTolerance(-1);
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        if (!(iter instanceof MRNSDFloatIterationMonitor)) {
            iter = new MRNSDFloatIterationMonitor();
            ((MRNSDFloatIterationMonitor) iter).setRelativeTolerance(-1);
        }
        float alpha;
        float gamma;
        float theta;
        float rnrm;
        FloatMatrix1D r, s, u, w;
        IntArrayList indexList;
        float tau = sqrteps;
        float sigsq = tau;
        //        x.assign(b); // x cannot be 0
        float[] minAndLoc = x.getMinLocation();
        float minX = minAndLoc[0];
        if (minX < 0) {
            x.assign(FloatFunctions.plus(-minX + sigsq));
        }

        if (((MRNSDFloatIterationMonitor) iter).getRelativeTolerance() == -1.0) {
            ((MRNSDFloatIterationMonitor) iter).setRelativeTolerance(sqrteps * alg.norm2(A.zMult(b, null, 1, 0, true)));
        }
        r = A.zMult(x, null);
        r.assign(b, FloatFunctions.plusMultFirst(-1));
        if (!(M instanceof FloatIdentity)) {
            r = M.apply(r, null);
            r = M.transApply(r, null);
            r = A.zMult(r, null, 1, 0, true);
            r.assign(FloatFunctions.neg);
            gamma = x.aggregate(r, FloatFunctions.plus, FloatFunctions.multSquare);
            rnrm = alg.norm2(r);
        } else {
            r = A.zMult(r, null, 1, 0, true);
            r.assign(FloatFunctions.neg);
            gamma = x.aggregate(r, FloatFunctions.plus, FloatFunctions.multSquare);
            rnrm = (float) Math.sqrt(gamma);
        }
        indexList = new IntArrayList((int) b.size());
        for (iter.setFirst(); !iter.converged(rnrm, x); iter.next()) {
            s = x.copy();
            s.assign(r, FloatFunctions.multNeg);
            u = A.zMult(s, null);
            if (!(M instanceof FloatIdentity)) {
                u = M.apply(u, null);
            }
            theta = gamma / u.aggregate(FloatFunctions.plus, FloatFunctions.square);
            s.getNegativeValues(indexList, null);
            w = x.copy();
            w.assign(s, FloatFunctions.divNeg, indexList);
            alpha = Math.min(theta, w.aggregate(FloatFunctions.min, FloatFunctions.identity, indexList));
            x.assign(s, FloatFunctions.plusMultSecond(alpha));
            if (!(M instanceof FloatIdentity)) {
                w = M.transApply(u, null);
                w = A.zMult(w, null, 1, 0, true);
                r.assign(w, FloatFunctions.plusMultSecond(alpha));
                gamma = x.aggregate(r, FloatFunctions.plus, FloatFunctions.multSquare);
                rnrm = alg.norm2(r);
            } else {
                w = A.zMult(u, null, 1, 0, true);
                r.assign(w, FloatFunctions.plusMultSecond(alpha));
                gamma = x.aggregate(r, FloatFunctions.plus, FloatFunctions.multSquare);
                rnrm = (float) Math.sqrt(gamma);
            }
        }
        return x;
    }
}
