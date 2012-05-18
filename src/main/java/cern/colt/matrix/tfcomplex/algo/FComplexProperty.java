/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.algo;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.jet.math.tfcomplex.FComplex;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Tests matrices for equality.
 * <p>
 * Except where explicitly indicated, all methods involving equality tests (
 * <tt>==</tt>) allow for numerical instability, to a degree specified upon
 * instance construction and returned by method {@link #tolerance()}. The public
 * static final variable <tt>DEFAULT</tt> represents a default Property object
 * with a tolerance of <tt>1.0E-5</tt>. The public static final variable
 * <tt>ZERO</tt> represents a Property object with a tolerance of <tt>0.0</tt>.
 * The public static final variable <tt>SEVEN</tt> represents a Property object
 * with a tolerance of <tt>1.0E-7</tt>. As long as you are happy with these
 * tolerances, there is no need to construct Property objects. Simply use idioms
 * like <tt>Property.DEFAULT.equals(A,B)</tt>,
 * <tt>Property.ZERO.equals(A,B)</tt>, <tt>Property.TWELVE.equals(A,B)</tt>.
 * <p>
 * To work with a different tolerance (e.g. <tt>1.0E-2</tt>) use the constructor
 * and/or method {@link #setTolerance(float)}. Note that the public static final
 * Property objects are immutable: Is is not possible to alter their tolerance.
 * Any attempt to do so will throw an Exception.
 * <p>
 * Note that this implementation is not synchronized.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.1, 28/May/2000 (fixed strange bugs involving NaN, -inf, inf)
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FComplexProperty {

    /**
     * The default Property object; currently has <tt>tolerance()==1.0E-5</tt>.
     */
    public static final FComplexProperty DEFAULT = new FComplexProperty(1.0E-5f);

    /**
     * A Property object with <tt>tolerance()==0.0</tt>.
     */
    public static final FComplexProperty ZERO = new FComplexProperty(0.0f);

    /**
     * A Property object with <tt>tolerance()==1.0E-7</tt>.
     */
    public static final FComplexProperty SEVEN = new FComplexProperty(1.0E-7f);

    protected float tolerance;

    /**
     * Not instantiable by no-arg constructor.
     */
    private FComplexProperty() {
        this(1.0E-5f); // just to be on the safe side
    }

    /**
     * Constructs an instance with a tolerance of
     * <tt>Math.abs(newTolerance)</tt>.
     */
    public FComplexProperty(float newTolerance) {
        tolerance = Math.abs(newTolerance);
    }

    /**
     * Sets the tolerance to <tt>Math.abs(newTolerance)</tt>.
     * 
     * @throws UnsupportedOperationException
     *             if <tt>this==DEFAULT || this==ZERO || this==TWELVE</tt>.
     */
    public void setTolerance(float newTolerance) {
        if (this == DEFAULT || this == ZERO || this == SEVEN) {
            throw new IllegalArgumentException("Attempted to modify immutable object.");
        }
        tolerance = Math.abs(newTolerance);
    }

    /**
     * Returns the current tolerance.
     */
    public float tolerance() {
        return tolerance;
    }

    /**
     * Returns whether all cells of the given matrix <tt>A</tt> are equal to the
     * given value.
     * 
     * @param A
     *            the first matrix to compare.
     * @param value
     *            the value to compare against.
     * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final FComplexMatrix1D A, final float[] value) {
        if (A == null)
            return false;
        final float epsilon = tolerance();
        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        int size = (int) A.size();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        float[] diff = new float[2];
                        for (int i = firstIdx; i < lastIdx; i++) {
                            float[] x = A.getQuick(i);
                            diff[0] = Math.abs(value[0] - x[0]);
                            diff[1] = Math.abs(value[1] - x[1]);
                            if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                    && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                    || (FComplex.isEqual(value, x, epsilon))) {
                                diff[0] = 0;
                                diff[1] = 0;
                            }
                            if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                                return false;
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            float[] diff = new float[2];
            for (int i = 0; i < A.size(); i++) {
                float[] x = A.getQuick(i);
                diff[0] = Math.abs(value[0] - x[0]);
                diff[1] = Math.abs(value[1] - x[1]);
                if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                        && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                        || (FComplex.isEqual(value, x, epsilon))) {
                    diff[0] = 0;
                    diff[1] = 0;
                }
                if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal.
     * 
     * @param A
     *            the first matrix to compare.
     * @param B
     *            the second matrix to compare.
     * @return <tt>true</tt> if both matrices are equal; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final FComplexMatrix1D A, final FComplexMatrix1D B) {
        if (A == B)
            return true;
        if (!(A != null && B != null))
            return false;
        int size = (int) A.size();
        if (size != B.size())
            return false;

        final float epsilon = tolerance();
        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        float[] diff = new float[2];
                        for (int i = firstIdx; i < lastIdx; i++) {
                            float[] x = A.getQuick(i);
                            float[] value = B.getQuick(i);
                            diff[0] = Math.abs(value[0] - x[0]);
                            diff[1] = Math.abs(value[1] - x[1]);
                            if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                    && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                    || (FComplex.isEqual(value, x, epsilon))) {
                                diff[0] = 0;
                                diff[1] = 0;
                            }
                            if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                                return false;
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            float[] diff = new float[2];
            for (int i = 0; i < size; i++) {
                float[] x = A.getQuick(i);
                float[] value = B.getQuick(i);
                diff[0] = Math.abs(value[0] - x[0]);
                diff[1] = Math.abs(value[1] - x[1]);
                if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                        && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                        || (FComplex.isEqual(value, x, epsilon))) {
                    diff[0] = 0;
                    diff[1] = 0;
                }
                if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * Returns whether all cells of the given matrix <tt>A</tt> are equal to the
     * given value.
     * 
     * @param A
     *            the first matrix to compare.
     * @param value
     *            the value to compare against.
     * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final FComplexMatrix2D A, final float[] value) {
        if (A == null)
            return false;
        int rows = A.rows();
        int columns = A.columns();
        boolean result = false;
        final float epsilon = tolerance();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, A.rows());
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = A.rows() / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? A.rows() : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        float[] diff = new float[2];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < A.columns(); c++) {
                                float[] x = A.getQuick(r, c);
                                diff[0] = Math.abs(value[0] - x[0]);
                                diff[1] = Math.abs(value[1] - x[1]);
                                if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                        && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                        || (FComplex.isEqual(value, x, epsilon))) {
                                    diff[0] = 0;
                                    diff[1] = 0;
                                }
                                if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                                    return false;
                                }
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            float[] diff = new float[2];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    float[] x = A.getQuick(r, c);
                    diff[0] = Math.abs(value[0] - x[0]);
                    diff[1] = Math.abs(value[1] - x[1]);
                    if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                            && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                            || (FComplex.isEqual(value, x, epsilon))) {
                        diff[0] = 0;
                        diff[1] = 0;
                    }
                    if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    /**
     * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal.
     * 
     * @param A
     *            the first matrix to compare.
     * @param B
     *            the second matrix to compare.
     * @return <tt>true</tt> if both matrices are equal; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final FComplexMatrix2D A, final FComplexMatrix2D B) {
        if (A == B)
            return true;
        if (!(A != null && B != null))
            return false;
        int rows = A.rows();
        int columns = A.columns();
        if (columns != B.columns() || rows != B.rows())
            return false;
        boolean result = false;
        final float epsilon = tolerance();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, A.rows());
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = A.rows() / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? A.rows() : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        float[] diff = new float[2];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < A.columns(); c++) {
                                float[] x = A.getQuick(r, c);
                                float[] value = B.getQuick(r, c);
                                diff[0] = Math.abs(value[0] - x[0]);
                                diff[1] = Math.abs(value[1] - x[1]);
                                if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                        && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                        || (FComplex.isEqual(value, x, epsilon))) {
                                    diff[0] = 0;
                                    diff[1] = 0;
                                }
                                if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                                    return false;
                                }
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {

            float[] diff = new float[2];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    float[] x = A.getQuick(r, c);
                    float[] value = B.getQuick(r, c);
                    diff[0] = Math.abs(value[0] - x[0]);
                    diff[1] = Math.abs(value[1] - x[1]);
                    if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                            && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                            || (FComplex.isEqual(value, x, epsilon))) {
                        diff[0] = 0;
                        diff[1] = 0;
                    }
                    if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    /**
     * Returns whether all cells of the given matrix <tt>A</tt> are equal to the
     * given value.
     * 
     * @param A
     *            the first matrix to compare.
     * @param value
     *            the value to compare against.
     * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final FComplexMatrix3D A, final float[] value) {
        if (A == null)
            return false;
        final int slices = A.slices();
        final int rows = A.rows();
        final int columns = A.columns();
        boolean result = false;
        final float epsilon = tolerance();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        float[] diff = new float[2];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    float[] x = A.getQuick(s, r, c);
                                    diff[0] = Math.abs(value[0] - x[0]);
                                    diff[1] = Math.abs(value[1] - x[1]);
                                    if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                            && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                            || (FComplex.isEqual(value, x, epsilon))) {
                                        diff[0] = 0;
                                        diff[1] = 0;
                                    }
                                    if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                                        return false;
                                    }
                                }
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            float[] diff = new float[2];
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        float[] x = A.getQuick(s, r, c);
                        diff[0] = Math.abs(value[0] - x[0]);
                        diff[1] = Math.abs(value[1] - x[1]);
                        if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                || (FComplex.isEqual(value, x, epsilon))) {
                            diff[0] = 0;
                            diff[1] = 0;
                        }
                        if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
    }

    /**
     * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal.
     * 
     * @param A
     *            the first matrix to compare.
     * @param B
     *            the second matrix to compare.
     * @return <tt>true</tt> if both matrices are equal; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final FComplexMatrix3D A, final FComplexMatrix3D B) {
        if (A == B)
            return true;
        if (!(A != null && B != null))
            return false;
        boolean result = false;
        final int slices = A.slices();
        final int rows = A.rows();
        final int columns = A.columns();
        if (columns != B.columns() || rows != B.rows() || slices != B.slices())
            return false;
        final float epsilon = tolerance();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int startslice = j * k;
                final int stopslice;
                if (j == nthreads - 1) {
                    stopslice = slices;
                } else {
                    stopslice = startslice + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        float[] diff = new float[2];
                        for (int s = startslice; s < stopslice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    float[] x = A.getQuick(s, r, c);
                                    float[] value = B.getQuick(s, r, c);
                                    diff[0] = Math.abs(value[0] - x[0]);
                                    diff[1] = Math.abs(value[1] - x[1]);
                                    if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                            && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                            || (FComplex.isEqual(value, x, epsilon))) {
                                        diff[0] = 0;
                                        diff[1] = 0;
                                    }
                                    if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                                        return false;
                                    }
                                }
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            float[] diff = new float[2];
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        float[] x = A.getQuick(s, r, c);
                        float[] value = B.getQuick(s, r, c);
                        diff[0] = Math.abs(value[0] - x[0]);
                        diff[1] = Math.abs(value[1] - x[1]);
                        if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                                && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                                || (FComplex.isEqual(value, x, epsilon))) {
                            diff[0] = 0;
                            diff[1] = 0;
                        }
                        if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
    }

}
