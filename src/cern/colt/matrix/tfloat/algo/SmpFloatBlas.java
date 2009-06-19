/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.algo;

import java.util.concurrent.Future;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.jet.math.tfloat.FloatFunctions;
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
public class SmpFloatBlas implements FloatBlas {

    public SmpFloatBlas() {
    }

    public void assign(FloatMatrix2D A, final cern.colt.function.tfloat.FloatFunction function) {
        A.assign(function);
    }

    public void assign(FloatMatrix2D A, FloatMatrix2D B, final cern.colt.function.tfloat.FloatFloatFunction function) {
        A.assign(B, function);
    }

    public float dasum(FloatMatrix1D x) {
        return x.aggregate(FloatFunctions.plus, FloatFunctions.abs);
    }

    public void daxpy(float alpha, FloatMatrix1D x, FloatMatrix1D y) {
        y.assign(x, FloatFunctions.plusMultSecond(alpha));
    }

    public void daxpy(float alpha, FloatMatrix2D A, FloatMatrix2D B) {
        B.assign(A, FloatFunctions.plusMultSecond(alpha));
    }

    public void dcopy(FloatMatrix1D x, FloatMatrix1D y) {
        y.assign(x);
    }

    public void dcopy(FloatMatrix2D A, FloatMatrix2D B) {
        B.assign(A);
    }

    public float ddot(FloatMatrix1D x, FloatMatrix1D y) {
        return x.zDotProduct(y);
    }

    public void dgemm(final boolean transposeA, final boolean transposeB, final float alpha, final FloatMatrix2D A,
            final FloatMatrix2D B, final float beta, final FloatMatrix2D C) {
        A.zMult(B, C, alpha, beta, transposeA, transposeB);
    }

    public void dgemv(final boolean transposeA, final float alpha, FloatMatrix2D A, final FloatMatrix1D x,
            final float beta, FloatMatrix1D y) {
        A.zMult(x, y, alpha, beta, transposeA);
    }

    public void dger(float alpha, FloatMatrix1D x, FloatMatrix1D y, FloatMatrix2D A) {
        cern.jet.math.tfloat.FloatPlusMultSecond fun = cern.jet.math.tfloat.FloatPlusMultSecond.plusMult(0);
        int rows = A.rows();
        for (int i = 0; i < rows; i++) {
            fun.multiplicator = alpha * x.getQuick(i);
            A.viewRow(i).assign(y, fun);
        }
    }

    public float dnrm2(FloatMatrix1D x) {
        return DenseFloatAlgebra.DEFAULT.norm2(x);
    }

    public void drot(FloatMatrix1D x, FloatMatrix1D y, float c, float s) {
        x.checkSize(y);
        FloatMatrix1D tmp = x.copy();

        x.assign(FloatFunctions.mult(c));
        x.assign(y, FloatFunctions.plusMultSecond(s));

        y.assign(FloatFunctions.mult(c));
        y.assign(tmp, FloatFunctions.minusMult(s));
    }

    public void drotg(float a, float b, float rotvec[]) {
        float c, s, roe, scale, r, z, ra, rb;

        roe = b;

        if (Math.abs(a) > Math.abs(b))
            roe = a;

        scale = Math.abs(a) + Math.abs(b);

        if (scale != 0.0) {

            ra = a / scale;
            rb = b / scale;
            r = scale * (float) Math.sqrt(ra * ra + rb * rb);
            r = sign(1.0f, roe) * r;
            c = a / r;
            s = b / r;
            z = 1.0f;
            if (Math.abs(a) > Math.abs(b))
                z = s;
            if ((Math.abs(b) >= Math.abs(a)) && (c != 0.0))
                z = 1.0f / c;

        } else {

            c = 1.0f;
            s = 0.0f;
            r = 0.0f;
            z = 0.0f;

        }

        a = r;
        b = z;

        rotvec[0] = a;
        rotvec[1] = b;
        rotvec[2] = c;
        rotvec[3] = s;
    }

    public void dscal(float alpha, FloatMatrix1D x) {
        x.assign(FloatFunctions.mult(alpha));
    }

    public void dscal(float alpha, FloatMatrix2D A) {
        A.assign(FloatFunctions.mult(alpha));
    }

    public void dswap(FloatMatrix1D x, FloatMatrix1D y) {
        y.swap(x);
    }

    public void dswap(FloatMatrix2D A, FloatMatrix2D B) {
        A.checkShape(B);
        int rows = A.rows();
        for (int i = 0; i < rows; i++)
            A.viewRow(i).swap(B.viewRow(i));
    }

    public void dsymv(boolean isUpperTriangular, final float alpha, FloatMatrix2D A, final FloatMatrix1D x,
            final float beta, final FloatMatrix1D y) {
        final FloatMatrix2D A_loc;
        if (isUpperTriangular) {
            A_loc = A.viewDice();
        } else {
            A_loc = A;
        }
        FloatProperty.DEFAULT.checkSquare(A_loc);
        int size = A_loc.rows();
        if (size != x.size() || size != y.size()) {
            throw new IllegalArgumentException(A_loc.toStringShort() + ", " + x.toStringShort() + ", "
                    + y.toStringShort());
        }
        final FloatMatrix1D tmp = x.like();
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
                            float sum = 0;
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
                float sum = 0;
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
            FloatMatrix2D A, final FloatMatrix1D x) {
        final FloatMatrix2D A_loc;
        final boolean isUpperTriangular_loc;
        if (transposeA) {
            A_loc = A.viewDice();
            isUpperTriangular_loc = !isUpperTriangular;
        } else {
            A_loc = A;
            isUpperTriangular_loc = isUpperTriangular;
        }

        FloatProperty.DEFAULT.checkSquare(A_loc);
        int size = A_loc.rows();
        if (size != x.size()) {
            throw new IllegalArgumentException(A_loc.toStringShort() + ", " + x.toStringShort());
        }

        final FloatMatrix1D b = x.like();
        final FloatMatrix1D y = x.like();
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
                            float sum = 0;
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
                float sum = 0;
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

    public int idamax(FloatMatrix1D x) {
        FloatMatrix1D x_abs = x.copy();
        x_abs.assign(FloatFunctions.abs);
        float[] maxAndLoc = x_abs.getMaxLocation();
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
    private float sign(float a, float b) {
        if (b < 0.0) {
            return -Math.abs(a);
        } else {
            return Math.abs(a);
        }
    }
}
