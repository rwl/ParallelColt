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

import optimization.FloatFmin;
import optimization.FloatFmin_methods;
import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.matrix.tdouble.algo.solver.HyBRInnerSolver;
import cern.colt.matrix.tdouble.algo.solver.HyBRRegularizationMethod;
import cern.colt.matrix.tfloat.FloatFactory2D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatSingularValueDecomposition;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatIdentity;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatPreconditioner;
import cern.colt.matrix.tfloat.impl.DenseColumnFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import cern.jet.math.tfloat.FloatFunctions;
import cern.jet.stat.tfloat.FloatDescriptive;

/**
 * HyBR is a Hybrid Bidiagonalization Regularization method used for solving
 * large-scale, ill-posed inverse problems of the form: b = A*x + noise The
 * method combines an iterative Lanczos Bidiagonalization (LBD) Method with an
 * SVD-based regularization method to stabilize the semiconvergence behavior
 * that is characteristic of many ill-posed problems. The code is derived from
 * RestoreTools: An Object Oriented Matlab Package for Image Restoration written
 * by James G. Nagy and several of his students, including Julianne Chung,
 * Katrina Palmer, Lisa Perrone, and Ryan Wright.
 * 
 * <p>
 * References:<br>
 * <p>
 * [1] Paige and Saunders, "LSQR an algorithm for sparse linear equations an
 * sparse least squares", ACM Trans. Math Software, 8 (1982), pp. 43-71.
 * </p>
 * <p>
 * [2] Bjorck, Grimme and Van Dooren, "An implicit shift bidiagonalization
 * algorithm for ill-posed systems", BIT 34 (11994), pp. 520-534.
 * </p>
 * <p>
 * [3] Chung, Nagy and O'Leary, "A Weighted GCV Method for Lanczos Hybrid
 * Regularization", Elec. Trans. Numer. Anal., 28 (2008), pp. 149--167.
 * </p>
 * </p>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class FloatHyBR extends AbstractFloatIterativeSolver {

    private HyBRInnerSolver innerSolver;

    private HyBRRegularizationMethod regMethod;

    private float regPar;

    private float omega;

    private boolean reorth;

    private int begReg;

    private float flatTol;

    private boolean computeRnrm;

    private static final DenseFloatAlgebra alg = DenseFloatAlgebra.DEFAULT;

    private static final float FMIN_TOL = 1.0e-4f;

    /**
     * Creates new instance of HyBR solver with default parameters:<br>
     * <br>
     * innerSolver = HyBR.InnerSolver.TIKHONOV<br>
     * regularizationMethod = HyBR.RegularizationMethod.ADAPTWGCV<br>
     * regularizationParameter = 0<br>
     * omega = 0<br>
     * reorthogonalize = false<br>
     * beginRegularization = 2<br>
     * flatTolerance = 1e-6<br>
     * computeRnrm = false;
     */
    public FloatHyBR() {
        this(HyBRInnerSolver.TIKHONOV, HyBRRegularizationMethod.ADAPTWGCV, 0, 0, false, 2, 1e-4f, false);
    }

    /**
     * Creates new instance of HyBR solver.
     * 
     * @param innerSolver
     *            solver for the inner problem
     * @param regularizationMethod
     *            a method for choosing a regularization parameter
     * @param regularizationParameter
     *            if regularizationMethod == HyBR.RegularizationMethod.NONE then
     *            the regularization parameter has to be specified here (value
     *            from the interval (0,1))
     * @param omega
     *            regularizationMethod == HyBR.RegularizationMethod.WGCV then
     *            omega has to be specified here (must be nonnegative)
     * @param reorthogonalize
     *            if thue then Lanczos subspaces are reorthogonalized
     * @param beginRegularization
     *            begin regularization after this iteration (must be at least 2)
     * @param flatTolerance
     *            tolerance for detecting flatness in the GCV curve as a
     *            stopping criteria (must be nonnegative)
     *@param computeRnrm
     *            if true then the norm of relative residual is computed
     */
    public FloatHyBR(HyBRInnerSolver innerSolver, HyBRRegularizationMethod regularizationMethod,
            float regularizationParameter, float omega, boolean reorthogonalize, int beginRegularization,
            float flatTolerance, boolean computeRnrm) {
        this.innerSolver = innerSolver;
        this.regMethod = regularizationMethod;
        if ((regularizationParameter < 0.0) || (regularizationParameter > 1.0)) {
            throw new IllegalArgumentException("regularizationParameter must be a number between 0 and 1.");
        }
        this.regPar = regularizationParameter;
        if (omega < 0.0) {
            throw new IllegalArgumentException("omega must be a nonnegative number.");
        }
        this.omega = omega;
        this.reorth = reorthogonalize;
        if (beginRegularization < 2) {
            throw new IllegalArgumentException("beginRegularization must be greater or equal 2");
        }
        this.begReg = beginRegularization;
        if (flatTolerance < 0.0) {
            throw new IllegalArgumentException("flatTolerance must be a nonnegative number.");
        }
        this.flatTol = flatTolerance;
        this.computeRnrm = computeRnrm;
        this.iter = new HyBRFloatIterationMonitor();
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x)
            throws IterativeSolverFloatNotConvergedException {
        if (!(iter instanceof HyBRFloatIterationMonitor)) {
            this.iter = new HyBRFloatIterationMonitor();
        }
        checkSizes(A, b, x);
        int rows = A.rows();
        int columns = A.columns();
        boolean bump = false;
        boolean warning = false;
        float rnrm = -1.0f;
        int iterationsSave = 0;
        float alpha = 0;
        float alphaSave = 0;
        float beta = 0;
        HyBRInnerSolver inSolver = HyBRInnerSolver.NONE;
        FloatLBD lbd;
        FloatMatrix1D v;
        FloatMatrix1D work;
        FloatMatrix2D Ub, Vb;
        FloatMatrix1D f = null;
        FloatMatrix1D xSave = null;
        float[] sv;
        FloatArrayList omegaList = new FloatArrayList(new float[begReg - 2]);
        FloatArrayList GCV = new FloatArrayList(new float[begReg - 2]);
        FloatMatrix2D U = new DenseFloatMatrix2D(1, (int) b.size());
        FloatMatrix2D C = null;
        FloatMatrix2D V = null;
        DenseFloatSingularValueDecomposition svd;
        if (computeRnrm) {
            work = b.copy();
            A.zMult(x, work, -1, 1, false);
            rnrm = alg.norm2(work);
        }
        if (M instanceof FloatIdentity) {
            beta = alg.norm2(b);
            U.viewRow(0).assign(b, FloatFunctions.multSecond(1.0f / beta));
            lbd = new FloatSimpleLBD(A, U, reorth);
        } else {
            work = new DenseFloatMatrix1D((int) b.size());
            work = M.apply(b, work);
            beta = alg.norm2(work);
            U.viewRow(0).assign(work, FloatFunctions.multSecond(1.0f / beta));
            lbd = new FloatPLBD(M, A, U, reorth);
        }
        for (iter.setFirst(); !iter.converged(rnrm, x); iter.next()) {
            lbd.apply();
            U = lbd.getU();
            C = lbd.getC();
            V = lbd.getV();
            v = new DenseFloatMatrix1D(C.columns() + 1);
            v.setQuick(0, beta);
            int i = iter.iterations();
            if (i >= 1) {
                if (i >= begReg - 1) {
                    inSolver = innerSolver;
                }
                switch (inSolver) {
                case TIKHONOV:
                    svd = alg.svd(C);
                    Ub = svd.getU();
                    sv = svd.getSingularValues();
                    Vb = svd.getV();
                    if (regMethod == HyBRRegularizationMethod.ADAPTWGCV) {
                        work = new DenseFloatMatrix1D(Ub.rows());
                        Ub.zMult(v, work, 1, 0, true);
                        omegaList.add(Math.min(1, findOmega(work, sv)));
                        omega = FloatDescriptive.mean(omegaList);
                    }
                    f = new DenseFloatMatrix1D(Vb.rows());
                    alpha = tikhonovSolver(Ub, sv, Vb, v, f);
                    ((HyBRFloatIterationMonitor) iter).setRegularizationParameter(alpha);
                    GCV.add(GCVstopfun(alpha, Ub.viewRow(0), sv, beta, rows, columns));
                    if (i > 1) {
                        if (Math.abs((GCV.getQuick(i - 1) - GCV.getQuick(i - 2))) / GCV.get(begReg - 2) < flatTol) {
                            V.zMult(f, x);
                            ((HyBRFloatIterationMonitor) iter)
                                    .setStoppingCondition(HyBRFloatIterationMonitor.HyBRStoppingCondition.FLAT_GCV_CURVE);
                            if (computeRnrm) {
                                work = b.copy();
                                A.zMult(x, work, -1, 1, false);
                                ((HyBRFloatIterationMonitor) iter).residual = alg.norm2(work);
                            }
                            return x;
                        } else if ((warning == true) && (GCV.size() > iterationsSave + 3)) {
                            for (int j = iterationsSave; j < GCV.size(); j++) {
                                if (GCV.getQuick(iterationsSave - 1) > GCV.get(j)) {
                                    bump = true;
                                }
                            }
                            if (bump == false) {
                                x.assign(xSave);
                                ((HyBRFloatIterationMonitor) iter)
                                        .setStoppingCondition(HyBRFloatIterationMonitor.HyBRStoppingCondition.MIN_OF_GCV_CURVE_WITHIN_WINDOW_OF_4_ITERATIONS);
                                ((HyBRFloatIterationMonitor) iter).iter = iterationsSave;
                                if (computeRnrm) {
                                    work = b.copy();
                                    A.zMult(x, work, -1, 1, false);
                                    ((HyBRFloatIterationMonitor) iter).residual = alg.norm2(work);
                                }
                                ((HyBRFloatIterationMonitor) iter).setRegularizationParameter(alphaSave);
                                return x;

                            } else {
                                bump = false;
                                warning = false;
                                iterationsSave = iter.getMaxIterations();
                            }
                        } else if (warning == false) {
                            if (GCV.get(i - 2) < GCV.get(i - 1)) {
                                warning = true;
                                xSave = new DenseFloatMatrix1D(V.rows());
                                alphaSave = alpha;
                                V.zMult(f, xSave);
                                iterationsSave = i;
                            }
                        }
                    }
                    break;
                case NONE:
                    f = alg.solve(C, v);
                    break;
                }
                V.zMult(f, x);
                if (computeRnrm) {
                    work = b.copy();
                    A.zMult(x, work, -1, 1, false);
                    rnrm = alg.norm2(work);
                }
            }
        }
        return x;

    }

    protected void checkSizes(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x) {
        if (b.size() != A.rows())
            throw new IllegalArgumentException("b.size() != A.rows()");
        if (x.size() != A.columns())
            throw new IllegalArgumentException("x.size() != A.columns()");
    }

    private float findOmega(FloatMatrix1D bhat, float[] s) {
        int m = (int) bhat.size();
        int n = s.length;
        float alpha = s[n - 1];
        float t0 = bhat.viewPart(n, m - n).aggregate(FloatFunctions.plus, FloatFunctions.square);
        FloatMatrix1D s2 = new DenseFloatMatrix1D(s);
        s2.assign(FloatFunctions.square);
        float alpha2 = alpha * alpha;
        FloatMatrix1D tt = s2.copy();
        tt.assign(FloatFunctions.plus(alpha2));
        tt.assign(FloatFunctions.inv);
        float t1 = s2.aggregate(tt, FloatFunctions.plus, FloatFunctions.mult);
        s2 = new DenseFloatMatrix1D(s);
        s2.assign(FloatFunctions.mult(alpha));
        s2.assign(bhat.viewPart(0, n), FloatFunctions.mult);
        s2.assign(FloatFunctions.square);
        FloatMatrix1D work = tt.copy();
        work.assign(FloatFunctions.pow(3));
        work.assign(FloatFunctions.abs);
        float t3 = work.aggregate(s2, FloatFunctions.plus, FloatFunctions.mult);
        work = new DenseFloatMatrix1D(s);
        work.assign(tt, FloatFunctions.mult);
        float t4 = work.aggregate(FloatFunctions.plus, FloatFunctions.square);
        work = tt.copy();
        work.assign(bhat.viewPart(0, n), FloatFunctions.mult);
        work.assign(FloatFunctions.mult(alpha2));
        float t5 = work.aggregate(FloatFunctions.plus, FloatFunctions.square);
        s2 = new DenseFloatMatrix1D(s);
        s2.assign(bhat.viewPart(0, n), FloatFunctions.mult);
        s2.assign(FloatFunctions.square);
        tt.assign(FloatFunctions.pow(3));
        tt.assign(FloatFunctions.abs);
        float v2 = tt.aggregate(s2, FloatFunctions.plus, FloatFunctions.mult);
        return (m * alpha2 * v2) / (t1 * t3 + t4 * (t5 + t0));
    }

    private float tikhonovSolver(FloatMatrix2D U, float[] s, FloatMatrix2D V, FloatMatrix1D b, FloatMatrix1D x) {
        TikFmin2D fmin;
        FloatMatrix1D bhat = new DenseFloatMatrix1D(U.rows());
        U.zMult(b, bhat, 1, 0, true);
        float alpha = 0;
        switch (regMethod) {
        case GCV:
            fmin = new TikFmin2D(bhat, s, 1);
            alpha = FloatFmin.fmin(0, s[0], fmin, FMIN_TOL);
            break;
        case WGCV:
            fmin = new TikFmin2D(bhat, s, omega);
            alpha = FloatFmin.fmin(0, s[0], fmin, FMIN_TOL);
            break;
        case ADAPTWGCV:
            fmin = new TikFmin2D(bhat, s, omega);
            alpha = FloatFmin.fmin(0, s[0], fmin, FMIN_TOL);
            break;
        case NONE: // regularization parameter is given
            alpha = regPar;
            break;
        }
        FloatMatrix1D d = new DenseFloatMatrix1D(s);
        d.assign(FloatFunctions.square);
        d.assign(FloatFunctions.plus(alpha * alpha));
        bhat = bhat.viewPart(0, s.length);
        FloatMatrix1D S = new DenseFloatMatrix1D(s);
        bhat.assign(S, FloatFunctions.mult);
        bhat.assign(d, FloatFunctions.div);
        V.zMult(bhat, x);
        return alpha;
    }

    private static class TikFmin2D implements FloatFmin_methods {
        FloatMatrix1D bhat;

        float[] s;

        float omega;

        public TikFmin2D(FloatMatrix1D bhat, float[] s, float omega) {
            this.bhat = bhat;
            this.s = s;
            this.omega = omega;
        }

        public float f_to_minimize(float alpha) {
            int m = (int) bhat.size();
            int n = s.length;
            float t0 = bhat.viewPart(n, m - n).aggregate(FloatFunctions.plus, FloatFunctions.square);
            FloatMatrix1D s2 = new DenseFloatMatrix1D(s);
            s2.assign(FloatFunctions.square);
            float alpha2 = alpha * alpha;
            FloatMatrix1D work = s2.copy();
            work.assign(FloatFunctions.plus(alpha2));
            work.assign(FloatFunctions.inv);
            FloatMatrix1D t1 = work.copy();
            t1.assign(FloatFunctions.mult(alpha2));
            FloatMatrix1D t2 = t1.copy();
            t2.assign(bhat.viewPart(0, n), FloatFunctions.mult);
            FloatMatrix1D t3 = work.copy();
            t3.assign(s2, FloatFunctions.mult);
            t3.assign(FloatFunctions.mult(1 - omega));
            float denom = t3.aggregate(t1, FloatFunctions.plus, FloatFunctions.plus) + m - n;
            return n * (t2.aggregate(FloatFunctions.plus, FloatFunctions.square) + t0) / (denom * denom);
        }

    }

    private float GCVstopfun(float alpha, FloatMatrix1D u, float[] s, float beta, int rows, int columns) {
        int k = s.length;
        float beta2 = beta * beta;
        FloatMatrix1D s2 = new DenseFloatMatrix1D(s);
        s2.assign(FloatFunctions.square);
        float alpha2 = alpha * alpha;
        FloatMatrix1D t1 = s2.copy();
        t1.assign(FloatFunctions.plus(alpha2));
        t1.assign(FloatFunctions.inv);
        FloatMatrix1D t2 = t1.copy();
        t2.assign(u.viewPart(0, k), FloatFunctions.mult);
        t2.assign(FloatFunctions.mult(alpha2));
        float num = (float) (beta2
                * (t2.aggregate(FloatFunctions.plus, FloatFunctions.square) + Math.pow(Math.abs(u.getQuick(k)), 2)) / columns);
        float den = (rows - t1.aggregate(s2, FloatFunctions.plus, FloatFunctions.mult)) / columns;
        den = den * den;
        return num / den;
    }

    private interface FloatLBD {
        public void apply();

        public FloatMatrix2D getC();

        public FloatMatrix2D getU();

        public FloatMatrix2D getV();
    }

    private class FloatSimpleLBD implements FloatLBD {
        private final DenseFloatAlgebra alg = DenseFloatAlgebra.DEFAULT;

        private final FloatFactory2D factory = FloatFactory2D.dense;

        private final FloatMatrix2D alphaBeta = new DenseFloatMatrix2D(2, 1);

        private final FloatMatrix2D A;

        private FloatMatrix2D C;

        private FloatMatrix2D U;

        private FloatMatrix2D V;

        private boolean reorth;

        private int counter = 1;

        public FloatSimpleLBD(FloatMatrix2D A, FloatMatrix2D U, boolean reorth) {
            this.A = A;
            this.reorth = reorth;
            this.U = U;
            this.V = null;
            this.C = null;
        }

        public void apply() {
            if (reorth) {
                int k = U.rows();
                FloatMatrix1D u = null;
                FloatMatrix1D v = null;
                FloatMatrix1D column = null;
                if (k == 1) {
                    v = A.zMult(U.viewRow(k - 1), v, 1, 0, true);
                } else {
                    v = A.zMult(U.viewRow(k - 1), v, 1, 0, true);
                    column = V.viewColumn(k - 2);
                    v.assign(column, FloatFunctions.plusMultSecond(-C.getQuick(k - 1, k - 2)));
                    for (int j = 0; j < k - 1; j++) {
                        column = V.viewColumn(j);
                        v.assign(column, FloatFunctions.plusMultSecond(-column.zDotProduct(v)));
                    }
                }
                float alpha = alg.norm2(v);
                v.assign(FloatFunctions.div(alpha));
                u = A.zMult(v, u);
                column = U.viewRow(k - 1);
                u.assign(column, FloatFunctions.plusMultSecond(-alpha));
                for (int j = 0; j < k; j++) {
                    column = U.viewRow(j);
                    u.assign(column, FloatFunctions.plusMultSecond(-column.zDotProduct(u)));
                }
                float beta = alg.norm2(u);
                alphaBeta.setQuick(0, 0, alpha);
                alphaBeta.setQuick(1, 0, beta);
                u.assign(FloatFunctions.div(beta));
                U = factory.appendRow(U, u);
                if (V == null) {
                    V = new DenseColumnFloatMatrix2D((int) v.size(), 1);
                    V.assign((float[]) v.elements());
                } else {
                    V = factory.appendColumn(V, v);
                }
                if (C == null) {
                    C = new DenseFloatMatrix2D(2, 1);
                    C.assign(alphaBeta);
                } else {
                    C = factory.composeBidiagonal(C, alphaBeta);
                }
            } else {
                FloatMatrix1D u = null;
                FloatMatrix1D v = null;
                FloatMatrix1D column = null;
                if (counter == 1) {
                    v = A.zMult(U.viewRow(0), v, 1, 0, true);
                } else {
                    v = A.zMult(U.viewRow(0), v, 1, 0, true);
                    column = V.viewColumn(counter - 2);
                    v.assign(column, FloatFunctions.plusMultSecond(-C.getQuick(counter - 1, counter - 2)));
                }
                float alpha = alg.norm2(v);
                v.assign(FloatFunctions.div(alpha));
                u = A.zMult(v, u);
                column = U.viewRow(0);
                u.assign(column, FloatFunctions.plusMultSecond(-alpha));
                float beta = alg.norm2(u);
                alphaBeta.setQuick(0, 0, alpha);
                alphaBeta.setQuick(1, 0, beta);
                u.assign(FloatFunctions.div(beta));
                U.viewRow(0).assign(u);
                if (V == null) {
                    V = new DenseColumnFloatMatrix2D((int) v.size(), 1);
                    V.assign((float[]) v.elements());
                } else {
                    V = factory.appendColumn(V, v);
                }
                if (C == null) {
                    C = new DenseFloatMatrix2D(2, 1);
                    C.assign(alphaBeta);
                } else {
                    C = factory.composeBidiagonal(C, alphaBeta);
                }
                counter++;
            }
        }

        public FloatMatrix2D getC() {
            return C;
        }

        public FloatMatrix2D getU() {
            return U;
        }

        public FloatMatrix2D getV() {
            return V;
        }
    }

    private class FloatPLBD implements FloatLBD {

        private final DenseFloatAlgebra alg = DenseFloatAlgebra.DEFAULT;

        private final FloatFactory2D factory = FloatFactory2D.dense;

        private final FloatMatrix2D alphaBeta = new DenseFloatMatrix2D(2, 1);

        private final FloatPreconditioner M;

        private final FloatMatrix2D A;

        private FloatMatrix2D C;

        private FloatMatrix2D U;

        private FloatMatrix2D V;

        private boolean reorth;

        private int counter = 1;

        public FloatPLBD(FloatPreconditioner M, FloatMatrix2D A, FloatMatrix2D U, boolean reorth) {
            this.M = M;
            this.A = A;
            this.reorth = reorth;
            this.U = U;
            this.V = null;
            this.C = null;
        }

        public void apply() {
            if (reorth) {
                int k = U.rows();
                FloatMatrix1D u = null;
                FloatMatrix1D v = null;
                FloatMatrix1D row = null;
                if (k == 1) {
                    row = U.viewRow(k - 1).copy();
                    row = M.transApply(row, row);
                    v = A.zMult(row, v, 1, 0, true);
                } else {
                    row = U.viewRow(k - 1).copy();
                    row = M.transApply(row, row);
                    v = A.zMult(row, v, 1, 0, true);
                    row = V.viewColumn(k - 2);
                    v.assign(row, FloatFunctions.plusMultSecond(-C.getQuick(k - 1, k - 2)));
                    for (int j = 0; j < k - 1; j++) {
                        row = V.viewColumn(j);
                        v.assign(row, FloatFunctions.plusMultSecond(-row.zDotProduct(v)));
                    }
                }
                float alpha = alg.norm2(v);
                v.assign(FloatFunctions.div(alpha));
                row = A.zMult(v, row);
                u = M.apply(row, u);
                row = U.viewRow(k - 1);
                u.assign(row, FloatFunctions.plusMultSecond(-alpha));
                for (int j = 0; j < k; j++) {
                    row = U.viewRow(j);
                    u.assign(row, FloatFunctions.plusMultSecond(-row.zDotProduct(u)));
                }
                float beta = alg.norm2(u);
                alphaBeta.setQuick(0, 0, alpha);
                alphaBeta.setQuick(1, 0, beta);
                u.assign(FloatFunctions.div(beta));
                U = factory.appendRow(U, u);
                if (V == null) {
                    V = new DenseColumnFloatMatrix2D((int) v.size(), 1);
                    V.assign((float[]) v.elements());
                } else {
                    V = factory.appendColumn(V, v);
                }
                if (C == null) {
                    C = new DenseFloatMatrix2D(2, 1);
                    C.assign(alphaBeta);
                } else {
                    C = factory.composeBidiagonal(C, alphaBeta);
                }
            } else {
                FloatMatrix1D u = null;
                FloatMatrix1D v = null;
                FloatMatrix1D row = null;
                if (counter == 1) {
                    row = U.viewRow(0).copy();
                    row = M.transApply(row, row);
                    v = A.zMult(row, v, 1, 0, true);
                } else {
                    row = U.viewRow(0).copy();
                    row = M.transApply(row, row);
                    v = A.zMult(row, v, 1, 0, true);
                    row = V.viewColumn(counter - 2);
                    v.assign(row, FloatFunctions.plusMultSecond(-C.getQuick(counter - 1, counter - 2)));
                }
                float alpha = alg.norm2(v);
                v.assign(FloatFunctions.div(alpha));
                row = A.zMult(v, row);
                u = M.apply(row, u);
                row = U.viewRow(0);
                u.assign(row, FloatFunctions.plusMultSecond(-alpha));
                float beta = alg.norm2(u);
                alphaBeta.setQuick(0, 0, alpha);
                alphaBeta.setQuick(1, 0, beta);
                u.assign(FloatFunctions.div(beta));
                U.viewRow(0).assign(u);
                if (V == null) {
                    V = new DenseColumnFloatMatrix2D((int) v.size(), 1);
                    V.assign((float[]) v.elements());
                } else {
                    V = factory.appendColumn(V, v);
                }
                if (C == null) {
                    C = new DenseFloatMatrix2D(2, 1);
                    C.assign(alphaBeta);
                } else {
                    C = factory.composeBidiagonal(C, alphaBeta);
                }
                counter++;
            }
        }

        public FloatMatrix2D getC() {
            return C;
        }

        public FloatMatrix2D getU() {
            return U;
        }

        public FloatMatrix2D getV() {
            return V;
        }
    }
}
