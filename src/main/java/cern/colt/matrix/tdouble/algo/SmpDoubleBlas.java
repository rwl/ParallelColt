/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.algo;

import java.util.concurrent.Future;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Parallel implementation of the Basic Linear Algebra System for symmetric
 * multi processing boxes. In all cases, no or only marginal speedup is seen for
 * small problem sizes; they are detected and the sequential algorithm is used.
 * 
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 0.9, 16/04/2000
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SmpDoubleBlas implements DoubleBlas {

    public SmpDoubleBlas() {
    }

    public void assign(DoubleMatrix2D A, final cern.colt.function.tdouble.DoubleFunction function) {
        A.assign(function);
    }

    public void assign(DoubleMatrix2D A, DoubleMatrix2D B,
            final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        A.assign(B, function);
    }

    public double dasum(DoubleMatrix1D x) {
        return x.aggregate(DoubleFunctions.plus, DoubleFunctions.abs);
    }

    public void daxpy(double alpha, DoubleMatrix1D x, DoubleMatrix1D y) {
        y.assign(x, DoubleFunctions.plusMultSecond(alpha));
    }

    public void daxpy(double alpha, DoubleMatrix2D A, DoubleMatrix2D B) {
        B.assign(A, DoubleFunctions.plusMultSecond(alpha));
    }

    public void dcopy(DoubleMatrix1D x, DoubleMatrix1D y) {
        y.assign(x);
    }

    public void dcopy(DoubleMatrix2D A, DoubleMatrix2D B) {
        B.assign(A);
    }

    public double ddot(DoubleMatrix1D x, DoubleMatrix1D y) {
        return x.zDotProduct(y);
    }

    public void dgemm(final boolean transposeA, final boolean transposeB, final double alpha, final DoubleMatrix2D A,
            final DoubleMatrix2D B, final double beta, final DoubleMatrix2D C) {
        A.zMult(B, C, alpha, beta, transposeA, transposeB);
    }

    public void dgemv(final boolean transposeA, final double alpha, DoubleMatrix2D A, final DoubleMatrix1D x,
            final double beta, DoubleMatrix1D y) {
        A.zMult(x, y, alpha, beta, transposeA);
    }

    public void dger(double alpha, DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix2D A) {
        cern.jet.math.tdouble.DoublePlusMultSecond fun = cern.jet.math.tdouble.DoublePlusMultSecond.plusMult(0);
        int rows = A.rows();
        for (int i = 0; i < rows; i++) {
            fun.multiplicator = alpha * x.getQuick(i);
            A.viewRow(i).assign(y, fun);
        }
    }

    public double dnrm2(DoubleMatrix1D x) {
        return DenseDoubleAlgebra.DEFAULT.norm2(x);
    }

    public void drot(DoubleMatrix1D x, DoubleMatrix1D y, double c, double s) {
        x.checkSize(y);
        DoubleMatrix1D tmp = x.copy();

        x.assign(DoubleFunctions.mult(c));
        x.assign(y, DoubleFunctions.plusMultSecond(s));

        y.assign(DoubleFunctions.mult(c));
        y.assign(tmp, DoubleFunctions.minusMult(s));
    }

    public void drotg(double a, double b, double rotvec[]) {
        double c, s, roe, scale, r, z, ra, rb;

        roe = b;

        if (Math.abs(a) > Math.abs(b))
            roe = a;

        scale = Math.abs(a) + Math.abs(b);

        if (scale != 0.0) {

            ra = a / scale;
            rb = b / scale;
            r = scale * Math.sqrt(ra * ra + rb * rb);
            r = sign(1.0, roe) * r;
            c = a / r;
            s = b / r;
            z = 1.0;
            if (Math.abs(a) > Math.abs(b))
                z = s;
            if ((Math.abs(b) >= Math.abs(a)) && (c != 0.0))
                z = 1.0 / c;

        } else {

            c = 1.0;
            s = 0.0;
            r = 0.0;
            z = 0.0;

        }

        a = r;
        b = z;

        rotvec[0] = a;
        rotvec[1] = b;
        rotvec[2] = c;
        rotvec[3] = s;
    }

    public void dscal(double alpha, DoubleMatrix1D x) {
        x.assign(DoubleFunctions.mult(alpha));
    }

    public void dscal(double alpha, DoubleMatrix2D A) {
        A.assign(DoubleFunctions.mult(alpha));
    }

    public void dswap(DoubleMatrix1D x, DoubleMatrix1D y) {
        y.swap(x);
    }

    public void dswap(DoubleMatrix2D A, DoubleMatrix2D B) {
        A.checkShape(B);
        int rows = A.rows();
        for (int i = 0; i < rows; i++)
            A.viewRow(i).swap(B.viewRow(i));
    }

    public void dsymv(boolean isUpperTriangular, final double alpha, DoubleMatrix2D A, final DoubleMatrix1D x,
            final double beta, final DoubleMatrix1D y) {
        final DoubleMatrix2D A_loc;
        if (isUpperTriangular) {
            A_loc = A.viewDice();
        } else {
            A_loc = A;
        }
        DoubleProperty.DEFAULT.checkSquare(A_loc);
        int size = A_loc.rows();
        if (size != x.size() || size != y.size()) {
            throw new IllegalArgumentException(A_loc.toStringShort() + ", " + x.toStringShort() + ", "
                    + y.toStringShort());
        }
        final DoubleMatrix1D tmp = x.like();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            double sum = 0;
                            for (int j = 0; j <= i; j++) {
                                sum += A_loc.getQuick(i, j) * x.getQuick(j);
                            }
                            for (int j = i + 1; j < lastIdx; j++) {
                                sum += A_loc.getQuick(j, i) * x.getQuick(j);
                            }
                            tmp.setQuick(i, alpha * sum + beta * y.getQuick(i));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                double sum = 0;
                for (int j = 0; j <= i; j++) {
                    sum += A_loc.getQuick(i, j) * x.getQuick(j);
                }
                for (int j = i + 1; j < size; j++) {
                    sum += A_loc.getQuick(j, i) * x.getQuick(j);
                }
                tmp.setQuick(i, alpha * sum + beta * y.getQuick(i));
            }
        }
        y.assign(tmp);
    }

    public void dtrmv(boolean isUpperTriangular, final boolean transposeA, final boolean isUnitTriangular,
            DoubleMatrix2D A, final DoubleMatrix1D x) {
        final DoubleMatrix2D A_loc;
        final boolean isUpperTriangular_loc;
        if (transposeA) {
            A_loc = A.viewDice();
            isUpperTriangular_loc = !isUpperTriangular;
        } else {
            A_loc = A;
            isUpperTriangular_loc = isUpperTriangular;
        }

        DoubleProperty.DEFAULT.checkSquare(A_loc);
        int size = A_loc.rows();
        if (size != x.size()) {
            throw new IllegalArgumentException(A_loc.toStringShort() + ", " + x.toStringShort());
        }

        final DoubleMatrix1D b = x.like();
        final DoubleMatrix1D y = x.like();
        if (isUnitTriangular) {
            y.assign(1);
        } else {
            for (int i = 0; i < size; i++) {
                y.setQuick(i, A_loc.getQuick(i, i));
            }
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            double sum = 0;
                            if (!isUpperTriangular_loc) {
                                for (int j = 0; j < i; j++) {
                                    sum += A_loc.getQuick(i, j) * x.getQuick(j);
                                }
                                sum += y.getQuick(i) * x.getQuick(i);
                            } else {
                                sum += y.getQuick(i) * x.getQuick(i);
                                for (int j = i + 1; j < lastIdx; j++) {
                                    sum += A_loc.getQuick(i, j) * x.getQuick(j);
                                }
                            }
                            b.setQuick(i, sum);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                double sum = 0;
                if (!isUpperTriangular_loc) {
                    for (int j = 0; j < i; j++) {
                        sum += A_loc.getQuick(i, j) * x.getQuick(j);
                    }
                    sum += y.getQuick(i) * x.getQuick(i);
                } else {
                    sum += y.getQuick(i) * x.getQuick(i);
                    for (int j = i + 1; j < size; j++) {
                        sum += A_loc.getQuick(i, j) * x.getQuick(j);
                    }
                }
                b.setQuick(i, sum);
            }
        }
        x.assign(b);
    }

    public int idamax(DoubleMatrix1D x) {
        DoubleMatrix1D x_abs = x.copy();
        x_abs.assign(DoubleFunctions.abs);
        double[] maxAndLoc = x_abs.getMaxLocation();
        return (int) maxAndLoc[1];
    }

    /**
     * Implements the FORTRAN sign (not sin) function. See the code for details.
     * 
     * @param a
     *            a
     * @param b
     *            b
     */
    private double sign(double a, double b) {
        if (b < 0.0) {
            return -Math.abs(a);
        } else {
            return Math.abs(a);
        }
    }
}
