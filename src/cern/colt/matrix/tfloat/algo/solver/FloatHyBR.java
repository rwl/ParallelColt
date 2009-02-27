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
import cern.colt.matrix.tfloat.FloatFactory2D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.FloatAlgebra;
import cern.colt.matrix.tfloat.algo.decomposition.FloatSingularValueDecompositionDC;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatIdentity;
import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatPreconditioner;
import cern.colt.matrix.tfloat.impl.DenseColFloatMatrix2D;
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

    public enum RegularizationMethod {
        GCV, WGCV, ADAPTWGCV, NONE
    }

    public enum InnerSolver {
        TIKHONOV, NONE
    }

    public class HyBROutput {
        private String stoppingCondition;
        private int stoppingIteration;
        private float rnorm;

        public HyBROutput() {
            stoppingCondition = "";
            stoppingIteration = 0;
            rnorm = 0;
        }

        public void setStoppingCondition(String flag) {
            this.stoppingCondition = flag;
        }

        public void setStoppingIteration(int stoppingIteration) {
            this.stoppingIteration = stoppingIteration;
        }

        public void setRnorm(float rnorm) {
            this.rnorm = rnorm;
        }

        public String getStoppingCondition() {
            return stoppingCondition;
        }

        public int getStoppingIteration() {
            return stoppingIteration;
        }

        public float getRnorm() {
            return rnorm;
        }

        public String toString() {

            return "Stopping condition = " + stoppingCondition + "\nStopping iteration = " + stoppingIteration + "\nRelative residual norm = " + rnorm;
        }
    }

    private HyBROutput hybrOutput;

    private InnerSolver innerSolver;

    private RegularizationMethod regMethod;

    private float regPar;

    private float omega;

    private int maxIts;

    private boolean reorth;

    private int begReg;

    private float flatTol;

    private static final FloatAlgebra alg = FloatAlgebra.DEFAULT;

    private static final float FMIN_TOL = 1.0e-4f;

    private boolean computeRnorm = false;

    /**
     * Creates new instance of HyBR solver with default parameters:<br>
     * <br>
     * innerSolver = HyBR.InnerSolver.TIKHONOV<br>
     * regularizationMethod = HyBR.RegularizationMethod.ADAPTWGCV<br>
     * regularizationParameter = -1<br>
     * omega = 0<br>
     * maxIterations = 100<br>
     * reorthogonalize = false<br>
     * beginRegularization = 2<br>
     * flatTolerance = 1e-6<br>
     * computeRnorm = false;
     */
    public FloatHyBR() {
        this(InnerSolver.TIKHONOV, RegularizationMethod.ADAPTWGCV, -1, 0, 100, false, 2, 1e-6f, false);
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
     *            the regularization parameter has to be specified here
     * @param omega
     *            regularizationMethod == HyBR.RegularizationMethod.WGCV then
     *            omega has to be specified here
     * @param maxIterations
     *            maximum number of Lanczos iterations
     * @param reorthogonalize
     *            if thue then Lanczos subspaces are reorthogonalized
     * @param beginRegularization
     *            begin regularization after this iteration
     * @param flatTolerance
     *            tolerance for detecting flatness in the GCV curve as a
     *            stopping criteria
     * @param computeRnorm
     *            if true then the relative residual norm is computed.
     */
    public FloatHyBR(InnerSolver innerSolver, RegularizationMethod regularizationMethod, float regularizationParameter, float omega, int maxIterations, boolean reorthogonalize, int beginRegularization, float flatTolerance, boolean computeRnorm) {
        this.innerSolver = innerSolver;
        this.regMethod = regularizationMethod;
        this.regPar = regularizationParameter;
        this.omega = omega;
        this.maxIts = maxIterations;
        this.reorth = reorthogonalize;
        this.begReg = beginRegularization;
        this.flatTol = flatTolerance;
        this.computeRnorm = computeRnorm;
        this.hybrOutput = new HyBROutput();
        this.iter = null; // HyBR doesn't use an IterationMonitor
    }

    public FloatIterationMonitor getIterationMonitor() {
        throw new IllegalAccessError("HyBR doesn't use an IterationMonitor. Use getHyBROutput() instead.");
    }

    public void setIterationMonitor(FloatIterationMonitor iter) {
        throw new IllegalAccessError("HyBR doesn't use an IterationMonitor. Use getHyBROutput() instead.");
    }

    public HyBROutput getHyBROutput() {
        return hybrOutput;
    }

    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b, FloatMatrix1D x) throws IterativeSolverFloatNotConvergedException {
        checkSizes(A, b, x);
        int columns = A.columns();
        boolean bump = false;
        boolean terminate = true;
        boolean warning = false;
        int iterationsSave = 0;
        float alpha, beta;
        InnerSolver inSolver = InnerSolver.NONE;
        FloatLBD lbd;
        FloatMatrix1D v;
        FloatMatrix1D work;
        FloatMatrix2D Ub, Vb;
        FloatMatrix1D f = null;
        FloatMatrix1D xSave = null;
        float[] sv;
        FloatArrayList omegaList = new FloatArrayList();
        FloatArrayList GCV = new FloatArrayList(new float[begReg - 1]);
        FloatMatrix2D U = new DenseFloatMatrix2D(1, b.size());
        FloatMatrix2D B = null;
        FloatMatrix2D V = null;
        FloatSingularValueDecompositionDC svd;
        work = b.copy();
        //        A.zMult(x, work, -1, 1, false);
        //        hybrOutput.setRnorm(alg.norm2(work));
        if (M instanceof FloatIdentity) {
            beta = alg.norm2(b);
            U.viewRow(0).assign(b, FloatFunctions.multSecond((float) (1.0 / beta)));
            lbd = new FloatSimpleLBD(A, U, reorth);
        } else {
            work = new DenseFloatMatrix1D(b.size());
            work = M.apply(b, work);
            beta = alg.norm2(work);
            U.viewRow(0).assign(work, FloatFunctions.multSecond((float) (1.0 / beta)));
            lbd = new FloatPLBD(M, A, U, reorth);
        }
        for (int i = 0; i <= maxIts; i++) {
            lbd.apply();
            U = lbd.getU();
            B = lbd.getB();
            V = lbd.getV();
            v = new DenseFloatMatrix1D(U.rows());
            v.setQuick(0, beta);
            if (i >= 1) {
                if (i >= begReg - 1) {
                    inSolver = innerSolver;
                }
                switch (inSolver) {
                case TIKHONOV:
                    svd = alg.svdDC(B);
                    Ub = svd.getU();
                    sv = svd.getSingularValues();
                    Vb = svd.getV();
                    if (regMethod == RegularizationMethod.ADAPTWGCV) {
                        work = new DenseFloatMatrix1D(Ub.rows());
                        Ub.zMult(v, work, 1, 0, true);
                        omegaList.add(Math.min(1, findOmega(work, sv)));
                        omega = FloatDescriptive.mean(omegaList);
                    }
                    f = new DenseFloatMatrix1D(Vb.rows());
                    alpha = tikhonovSolver(Ub, sv, Vb, v, f);
                    GCV.add(GCVstopfun(alpha, Ub.viewRow(0), sv, beta, columns));
                    if ((i > 1) && (terminate == true)) {
                        if (Math.abs((GCV.get(i) - GCV.get(i - 1))) / GCV.get(begReg) < flatTol) {
                            V.zMult(f, x);
                            hybrOutput.setStoppingCondition("flat GCV curve");
                            hybrOutput.setStoppingIteration(i);
                            if (computeRnorm) {
                                work = b.copy();
                                A.zMult(x, work, -1, 1, false);
                                hybrOutput.setRnorm(alg.norm2(work));
                            }
                            return x;
                        } else if ((warning == true) && (GCV.size() > iterationsSave + 3)) {
                            for (int j = iterationsSave; j < GCV.size() - 1; j++) {
                                if (GCV.get(iterationsSave) > GCV.get(j + 1)) {
                                    bump = true;
                                }
                            }
                            if (bump == false) {
                                x.assign(xSave);
                                hybrOutput.setStoppingCondition("min of GCV curve (within window of 4 iterations)");
                                hybrOutput.setStoppingIteration(iterationsSave);
                                if (computeRnorm) {
                                    work = b.copy();
                                    A.zMult(x, work, -1, 1, false);
                                    hybrOutput.setRnorm(alg.norm2(work));
                                }
                                return x;

                            } else {
                                bump = false;
                                warning = false;
                                iterationsSave = maxIts;
                            }
                        } else if (warning == false) {
                            if (GCV.get(i - 1) < GCV.get(i)) {
                                warning = true;
                                xSave = new DenseFloatMatrix1D(V.rows());
                                V.zMult(f, xSave);
                                iterationsSave = i;
                            }
                        }
                    }
                    break;
                case NONE:
                    f = alg.solve(B, v);
                    break;
                }
                V.zMult(f, x);
            }
        }
        if (computeRnorm) {
            work = b.copy();
            A.zMult(x, work, -1, 1, false);
            hybrOutput.setRnorm(alg.norm2(work));
        }
        hybrOutput.setStoppingCondition("performed max number of iterations");
        hybrOutput.setStoppingIteration(maxIts);
        return x;

    }

    private float findOmega(FloatMatrix1D bhat, float[] s) {
        int m = bhat.size();
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
        FloatMatrix1D work_vec = tt.copy();
        work_vec.assign(FloatFunctions.pow(3));
        work_vec.assign(FloatFunctions.abs);
        float t3 = work_vec.aggregate(s2, FloatFunctions.plus, FloatFunctions.mult);
        work_vec = new DenseFloatMatrix1D(s);
        work_vec.assign(tt, FloatFunctions.mult);
        float t4 = work_vec.aggregate(FloatFunctions.plus, FloatFunctions.square);
        work_vec = tt.copy();
        work_vec.assign(bhat.viewPart(0, n), FloatFunctions.mult);
        work_vec.assign(FloatFunctions.mult(alpha2));
        float t5 = work_vec.aggregate(FloatFunctions.plus, FloatFunctions.square);
        s2 = new DenseFloatMatrix1D(s);
        s2.assign(bhat.viewPart(0, n), FloatFunctions.mult);
        s2.assign(FloatFunctions.square);
        tt.assign(FloatFunctions.pow(3));
        tt.assign(FloatFunctions.abs);
        float v2 = tt.aggregate(s2, FloatFunctions.plus, FloatFunctions.mult);
        return (m * alpha2 * v2) / (t1 * t3 + t4 * (t5 + t0));
    }

    private float tikhonovSolver(FloatMatrix2D U, float[] s, FloatMatrix2D V, FloatMatrix1D b, FloatMatrix1D x) {
        TikFmin_2D fmin;
        FloatMatrix1D bhat = new DenseFloatMatrix1D(U.rows());
        U.zMult(b, bhat, 1, 0, true);
        float alpha = 0;
        switch (regMethod) {
        case GCV:
            fmin = new TikFmin_2D(bhat, s, 1);
            alpha = FloatFmin.fmin(0, 1, fmin, FMIN_TOL);
            break;
        case WGCV:
            fmin = new TikFmin_2D(bhat, s, omega);
            alpha = FloatFmin.fmin(0, 1, fmin, FMIN_TOL);
            break;
        case ADAPTWGCV:
            fmin = new TikFmin_2D(bhat, s, omega);
            alpha = FloatFmin.fmin(0, 1, fmin, FMIN_TOL);
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

    private static class TikFmin_2D implements FloatFmin_methods {
        FloatMatrix1D bhat;

        float[] s;

        float omega;

        public TikFmin_2D(FloatMatrix1D bhat, float[] s, float omega) {
            this.bhat = bhat;
            this.s = s;
            this.omega = omega;
        }

        public float f_to_minimize(float alpha) {
            int m = bhat.size();
            int n = s.length;
            float t0 = bhat.viewPart(n, m - n).aggregate(FloatFunctions.plus, FloatFunctions.square);
            FloatMatrix1D s2 = new DenseFloatMatrix1D(s);
            s2.assign(FloatFunctions.square);
            float alpha2 = alpha * alpha;
            FloatMatrix1D work_vec = s2.copy();
            work_vec.assign(FloatFunctions.plus(alpha2));
            work_vec.assign(FloatFunctions.inv);
            FloatMatrix1D t1 = work_vec.copy();
            t1.assign(FloatFunctions.mult(alpha2));
            FloatMatrix1D t2 = t1.copy();
            t2.assign(bhat.viewPart(0, n), FloatFunctions.mult);
            FloatMatrix1D t3 = work_vec.copy();
            t3.assign(s2, FloatFunctions.mult);
            t3.assign(FloatFunctions.mult(1 - omega));
            float denom = t3.aggregate(t1, FloatFunctions.plus, FloatFunctions.plus) + m - n;
            return n * (t2.aggregate(FloatFunctions.plus, FloatFunctions.square) + t0) / (denom * denom);
        }

    }

    private float GCVstopfun(float alpha, FloatMatrix1D u, float[] s, float beta, int n) {
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
        float num = (float) (beta2 * (t2.aggregate(FloatFunctions.plus, FloatFunctions.square) + Math.pow(Math.abs(u.getQuick(k)), 2)) / (float) n);
        float den = (n - t1.aggregate(s2, FloatFunctions.plus, FloatFunctions.mult)) / (float) n;
        den = den * den;
        return num / den;
    }

    private interface FloatLBD {
        public void apply();

        public FloatMatrix2D getB();

        public FloatMatrix2D getU();

        public FloatMatrix2D getV();
    }

    private class FloatSimpleLBD implements FloatLBD {
        private final FloatAlgebra alg = FloatAlgebra.DEFAULT;

        private final FloatFactory2D factory = FloatFactory2D.dense;

        private final FloatMatrix2D alphaBeta = new DenseFloatMatrix2D(2, 1);

        private final FloatMatrix2D A;

        private FloatMatrix2D B;

        private FloatMatrix2D U;

        private FloatMatrix2D V;

        private boolean reorth;

        public FloatSimpleLBD(FloatMatrix2D A, FloatMatrix2D U, boolean reorth) {
            this.A = A;
            this.reorth = reorth;
            this.U = U;
            this.V = null;
            this.B = null;
        }

        public void apply() {
            int k = U.rows();
            FloatMatrix1D u = null;
            FloatMatrix1D v = null;
            FloatMatrix1D column = null;
            if (k == 1) {
                v = A.zMult(U.viewRow(k - 1), v, 1, 0, true);
            } else {
                v = A.zMult(U.viewRow(k - 1), v, 1, 0, true);
                column = V.viewColumn(k - 2).copy();
                v.assign(column.assign(FloatFunctions.mult(B.getQuick(k - 1, k - 2))), FloatFunctions.minus);
                if (reorth) {
                    for (int j = 0; j < k - 1; j++) {
                        column = V.viewColumn(j).copy();
                        v.assign(column.assign(FloatFunctions.mult(column.zDotProduct(v))), FloatFunctions.minus);
                    }
                }
            }
            float alpha = alg.norm2(v);
            v.assign(FloatFunctions.div(alpha));
            u = A.zMult(v, u);
            column = U.viewRow(k - 1).copy();
            u.assign(column.assign(FloatFunctions.mult(alpha)), FloatFunctions.minus);
            if (reorth) {
                for (int j = 0; j < k; j++) {
                    column = U.viewRow(j).copy();
                    u.assign(column.assign(FloatFunctions.mult(column.zDotProduct(u))), FloatFunctions.minus);
                }
            }
            float beta = alg.norm2(u);
            alphaBeta.setQuick(0, 0, alpha);
            alphaBeta.setQuick(1, 0, beta);
            u.assign(FloatFunctions.div(beta));
            U = factory.appendRow(U, u);
            if (V == null) {
                V = new DenseColFloatMatrix2D(v.size(), 1);
                V.assign((float[]) v.elements());
            } else {
                V = factory.appendColumn(V, v);
            }
            if (B == null) {
                B = new DenseFloatMatrix2D(2, 1);
                B.assign(alphaBeta);
            } else {
                B = factory.composeBidiagonal(B, alphaBeta);
            }
        }

        public FloatMatrix2D getB() {
            return B;
        }

        public FloatMatrix2D getU() {
            return U;
        }

        public FloatMatrix2D getV() {
            return V;
        }
    }

    private class FloatPLBD implements FloatLBD {

        private final FloatAlgebra alg = FloatAlgebra.DEFAULT;

        private final FloatFactory2D factory = FloatFactory2D.dense;

        private final FloatMatrix2D alphaBeta = new DenseFloatMatrix2D(2, 1);

        private final FloatPreconditioner M;

        private final FloatMatrix2D A;

        private FloatMatrix2D B;

        private FloatMatrix2D U;

        private FloatMatrix2D V;

        private boolean reorth;

        public FloatPLBD(FloatPreconditioner M, FloatMatrix2D A, FloatMatrix2D U, boolean reorth) {
            this.M = M;
            this.A = A;
            this.reorth = reorth;
            this.U = U;
            this.V = null;
            this.B = null;
        }

        public void apply() {
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
                row = V.viewColumn(k - 2).copy();
                v.assign(row.assign(FloatFunctions.mult(B.getQuick(k - 1, k - 2))), FloatFunctions.minus);
                if (reorth) {
                    for (int j = 0; j < k - 1; j++) {
                        row = V.viewColumn(j).copy();
                        v.assign(row.assign(FloatFunctions.mult(row.zDotProduct(v))), FloatFunctions.minus);
                    }
                }
            }
            float alpha = alg.norm2(v);
            v.assign(FloatFunctions.div(alpha));
            row = A.zMult(v, row);
            u = M.apply(row, u);
            row = U.viewRow(k - 1).copy();
            u.assign(row.assign(FloatFunctions.mult(alpha)), FloatFunctions.minus);
            if (reorth) {
                for (int j = 0; j < k; j++) {
                    row = U.viewRow(j).copy();
                    u.assign(row.assign(FloatFunctions.mult(row.zDotProduct(u))), FloatFunctions.minus);
                }
            }
            float beta = alg.norm2(u);
            alphaBeta.setQuick(0, 0, alpha);
            alphaBeta.setQuick(1, 0, beta);
            u.assign(FloatFunctions.div(beta));
            U = factory.appendRow(U, u);
            if (V == null) {
                V = new DenseColFloatMatrix2D(v.size(), 1);
                V.assign((float[]) v.elements());
            } else {
                V = factory.appendColumn(V, v);
            }
            if (B == null) {
                B = new DenseFloatMatrix2D(2, 1);
                B.assign(alphaBeta);
            } else {
                B = factory.composeBidiagonal(B, alphaBeta);
            }
        }

        public FloatMatrix2D getB() {
            return B;
        }

        public FloatMatrix2D getU() {
            return U;
        }

        public FloatMatrix2D getV() {
            return V;
        }
    }
}
