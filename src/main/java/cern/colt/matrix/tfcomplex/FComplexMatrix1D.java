/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.AbstractMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.jet.math.tfcomplex.FComplex;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Abstract base class for 1-d matrices (aka <i>vectors</i>) holding
 * <tt>complex</tt> elements. A matrix has a number of cells (its <i>size</i>),
 * which are assigned upon instance construction. Elements are accessed via zero
 * based indexes. Legal indexes are of the form <tt>[0..size()-1]</tt>. Any
 * attempt to access an element at a coordinate
 * <tt>index&lt;0 || index&gt;=size()</tt> will throw an
 * <tt>IndexOutOfBoundsException</tt>.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public abstract class FComplexMatrix1D extends AbstractMatrix1D {

    private static final long serialVersionUID = 1L;

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected FComplexMatrix1D() {
    }

    /**
     * Applies a function to each cell and aggregates the results.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell value.
     * @param f
     *            a function transforming the current cell value.
     * @return the aggregated measure.
     * @see cern.jet.math.tfcomplex.FComplexFunctions
     */
    public float[] aggregate(final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr,
            final cern.colt.function.tfcomplex.FComplexFComplexFunction f) {
        float[] b = new float[2];
        int size = (int) size();
        if (size == 0) {
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        float[] a = f.apply(getQuick(0));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float[] a = f.apply(getQuick(firstIdx));
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            a = aggr.apply(a, f.apply(getQuick(i)));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            for (int i = 1; i < size; i++) {
                a = aggr.apply(a, f.apply(getQuick(i)));
            }
        }
        return a;
    }

    /**
     * Applies a function to each corresponding cell of two matrices and
     * aggregates the results.
     * 
     * @param other
     *            the secondary matrix to operate on.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell values.
     * @param f
     *            a function transforming the current cell values.
     * @return the aggregated measure.
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     * @see cern.jet.math.tfcomplex.FComplexFunctions
     */
    public float[] aggregate(final FComplexMatrix1D other,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction f) {
        checkSize(other);
        int size = (int) size();
        if (size == 0) {
            float[] b = new float[2];
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        float[] a = f.apply(getQuick(0), other.getQuick(0));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float[] a = f.apply(getQuick(firstIdx), other.getQuick(firstIdx));
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            a = aggr.apply(a, f.apply(getQuick(i), other.getQuick(i)));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            for (int i = 1; i < size; i++) {
                a = aggr.apply(a, f.apply(getQuick(i), other.getQuick(i)));
            }
        }
        return a;
    }

    /**
     * Assigns the result of a function to each cell;
     * 
     * @param f
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tfcomplex.FComplexFunctions
     */
    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexFComplexFunction f) {
        int size = (int) size();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, f.apply(getQuick(i)));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, f.apply(getQuick(i)));
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to all cells that satisfy a condition.
     * 
     * @param cond
     *            a condition.
     * 
     * @param f
     *            a function object.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tfcomplex.FComplexFunctions
     */
    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond,
            final cern.colt.function.tfcomplex.FComplexFComplexFunction f) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    float[] elem;

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elem = getQuick(i);
                            if (cond.apply(elem) == true) {
                                setQuick(i, f.apply(elem));
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] elem;
            for (int i = 0; i < size; i++) {
                elem = getQuick(i);
                if (cond.apply(elem) == true) {
                    setQuick(i, f.apply(elem));
                }
            }
        }
        return this;
    }

    /**
     * Assigns a value to all cells that satisfy a condition.
     * 
     * @param cond
     *            a condition.
     * 
     * @param value
     *            a value (re=value[0], im=value[1]).
     * @return <tt>this</tt> (for convenience only).
     * 
     */
    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond, final float[] value) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    float[] elem;

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elem = getQuick(i);
                            if (cond.apply(elem) == true) {
                                setQuick(i, value);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] elem;
            for (int i = 0; i < size; i++) {
                elem = getQuick(i);
                if (cond.apply(elem) == true) {
                    setQuick(i, value);
                }
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to the real part of the receiver. The
     * imaginary part of the receiver is reset to zero.
     * 
     * @param f
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tfcomplex.FComplexFunctions
     */
    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexRealFunction f) {
        int size = (int) size();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, f.apply(getQuick(i)), 0);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, f.apply(getQuick(i)), 0);
            }
        }
        return this;
    }

    /**
     * Replaces all cell values of the receiver with the values of another
     * matrix. Both matrices must have the same size. If both matrices share the
     * same cells (as is the case if they are views derived from the same
     * matrix) and intersect in an ambiguous way, then replaces <i>as if</i>
     * using an intermediate auxiliary deep copy of <tt>other</tt>.
     * 
     * @param other
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public FComplexMatrix1D assign(FComplexMatrix1D other) {
        if (other == this)
            return this;
        checkSize(other);
        final FComplexMatrix1D otherLoc;
        if (haveSharedCells(other)) {
            otherLoc = other.copy();
        } else {
            otherLoc = other;
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, otherLoc.getQuick(i));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, otherLoc.getQuick(i));
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to each cell;
     * 
     * @param y
     *            the secondary matrix to operate on.
     * @param f
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != y.size()</tt>.
     * @see cern.jet.math.tfcomplex.FComplexFunctions
     */
    public FComplexMatrix1D assign(final FComplexMatrix1D y,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction f) {
        int size = (int) size();
        checkSize(y);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, f.apply(getQuick(i), y.getQuick(i)));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, f.apply(getQuick(i), y.getQuick(i)));
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>re</tt> and <tt>im</tt>.
     * 
     * @param re
     *            the real part of the value to be filled into the cells.
     * @param im
     *            the imaginary part of the value to be filled into the cells.
     * 
     * @return <tt>this</tt> (for convenience only).
     */
    public FComplexMatrix1D assign(final float re, final float im) {
        int size = (int) size();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, re, im);
                        }

                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, re, im);
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the same number of cells as the receiver. Complex
     * data is represented by 2 float values in sequence: the real and imaginary
     * parts, i.e. input array must be of size 2*size().
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != 2*size()</tt>.
     */
    public FComplexMatrix1D assign(final float[] values) {
        int size = (int) size();
        if (values.length != 2 * size)
            throw new IllegalArgumentException("The length of values[] must be equal to 2*size()=" + size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, values[2 * i], values[2 * i + 1]);
                        }

                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, values[2 * i], values[2 * i + 1]);
            }
        }
        return this;
    }

    /**
     * Replaces imaginary part of the receiver with the values of another real
     * matrix. The real part remains unchanged. Both matrices must have the same
     * size.
     * 
     * @param other
     *            the source matrix to copy from
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public FComplexMatrix1D assignImaginary(final FloatMatrix1D other) {
        checkSize(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            float re = getQuick(i)[0];
                            float im = other.getQuick(i);
                            setQuick(i, re, im);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                float re = getQuick(i)[0];
                float im = other.getQuick(i);
                setQuick(i, re, im);
            }
        }
        return this;
    }

    /**
     * Replaces real part of the receiver with the values of another real
     * matrix. The imaginary part remains unchanged. Both matrices must have the
     * same size.
     * 
     * @param other
     *            the source matrix to copy from
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public FComplexMatrix1D assignReal(final FloatMatrix1D other) {
        checkSize(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            float re = other.getQuick(i);
                            float im = getQuick(i)[1];
                            setQuick(i, re, im);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                float re = other.getQuick(i);
                float im = getQuick(i)[1];
                setQuick(i, re, im);
            }
        }
        return this;
    }

    /**
     * Returns the number of cells having non-zero values; ignores tolerance.
     * 
     * @return the number of cells having non-zero values.
     */
    public int cardinality() {
        int size = (int) size();
        int cardinality = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        float[] tmp = new float[2];
                        for (int i = firstIdx; i < lastIdx; i++) {
                            tmp = getQuick(i);
                            if ((tmp[0] != 0.0) || (tmp[1] != 0.0))
                                cardinality++;
                        }
                        return cardinality;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Integer) futures[j].get();
                }
                cardinality = results[0];
                for (int j = 1; j < nthreads; j++) {
                    cardinality += results[j];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            float[] tmp = new float[2];
            for (int i = 0; i < size; i++) {
                tmp = getQuick(i);
                if ((tmp[0] != 0.0) || (tmp[1] != 0.0))
                    cardinality++;
            }
        }
        return cardinality;
    }

    /**
     * Constructs and returns a deep copy of the receiver.
     * <p>
     * <b>Note that the returned matrix is an independent deep copy.</b> The
     * returned matrix is not backed by this matrix, so changes in the returned
     * matrix are not reflected in this matrix, and vice-versa.
     * 
     * @return a deep copy of the receiver.
     */
    public FComplexMatrix1D copy() {
        FComplexMatrix1D copy = like();
        copy.assign(this);
        return copy;
    }

    /**
     * Returns whether all cells are equal to the given value.
     * 
     * @param value
     *            the value to test against (re=value[0], im=value[1]).
     * 
     * @return <tt>true</tt> if all cells are equal to the given value,
     *         <tt>false</tt> otherwise.
     */
    public boolean equals(float[] value) {
        return cern.colt.matrix.tfcomplex.algo.FComplexProperty.DEFAULT.equals(this, value);
    }

    /**
     * Compares this object against the specified object. The result is
     * <code>true</code> if and only if the argument is not <code>null</code>
     * and is at least a <code>ComplexMatrix1D</code> object that has the same
     * sizes as the receiver and has exactly the same values at the same
     * indexes.
     * 
     * @param obj
     *            the object to compare with.
     * @return <code>true</code> if the objects are the same; <code>false</code>
     *         otherwise.
     */

    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (!(obj instanceof FComplexMatrix1D))
            return false;

        return cern.colt.matrix.tfcomplex.algo.FComplexProperty.DEFAULT.equals(this, (FComplexMatrix1D) obj);
    }

    /**
     * Returns the matrix cell value at coordinate <tt>index</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @return the value of the specified cell.
     * @throws IndexOutOfBoundsException
     *             if <tt>index&lt;0 || index&gt;=size()</tt>.
     */
    public float[] get(int index) {
        int size = (int) size();
        if (index < 0 || index >= size)
            checkIndex(index);
        return getQuick(index);
    }

    /**
     * Returns the elements of this matrix.
     * 
     * @return the elements
     */
    public abstract Object elements();

    /**
     * Returns the imaginary part of this matrix
     * 
     * @return the imaginary part
     */
    public abstract FloatMatrix1D getImaginaryPart();

    /**
     * Fills the coordinates and values of cells having non-zero values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * <p>
     * In general, fill order is <i>unspecified</i>. This implementation fills
     * like: <tt>for (index = 0..size()-1)  do ... </tt>. However, subclasses
     * are free to us any other order, even an order that may change over time
     * as cell values are changed. (Of course, result lists indexes are
     * guaranteed to correspond to the same cell).
     * 
     * @param indexList
     *            the list to be filled with indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getNonZeros(final IntArrayList indexList, final ArrayList<float[]> valueList) {
        indexList.clear();
        valueList.clear();
        int s = (int) size();
        for (int i = 0; i < s; i++) {
            float[] value = getQuick(i);
            if (value[0] != 0 || value[1] != 0) {
                indexList.add(i);
                valueList.add(value);
            }
        }

    }

    /**
     * Returns the matrix cell value at coordinate <tt>index</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @return the value of the specified cell.
     */
    public abstract float[] getQuick(int index);

    /**
     * Returns the real part of this matrix
     * 
     * @return the real part
     */
    public abstract FloatMatrix1D getRealPart();

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the same size. For example, if the receiver is an
     * instance of type <tt>DenseComplexMatrix1D</tt> the new matrix must also
     * be of type <tt>DenseComplexMatrix1D</tt>. In general, the new matrix
     * should have internal parametrization as similar as possible.
     * 
     * @return a new empty matrix of the same dynamic type.
     */
    public FComplexMatrix1D like() {
        int size = (int) size();
        return like(size);
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified size. For example, if the receiver
     * is an instance of type <tt>DenseFComplexMatrix1D</tt> the new matrix must
     * also be of type <tt>DenseFComplexMatrix1D</tt>. In general, the new
     * matrix should have internal parametrization as similar as possible.
     * 
     * @param size
     *            the number of cell the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public abstract FComplexMatrix1D like(int size);

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, entirely independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseFComplexMatrix1D</tt> the new
     * matrix must be of type <tt>DenseFComplexMatrix2D</tt>.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public abstract FComplexMatrix2D like2D(int rows, int columns);

    /**
     * Returns new FloatMatrix2D of size rows x columns whose elements are taken
     * column-wise from this matrix.
     * 
     * @param rows
     *            number of rows
     * @param columns
     *            number of columns
     * @return new 2D matrix with columns being the elements of this matrix.
     */
    public abstract FComplexMatrix2D reshape(int rows, int columns);

    /**
     * Returns new FloatMatrix3D of size slices x rows x columns, whose elements
     * are taken column-wise from this matrix.
     * 
     * @param rows
     *            number of rows
     * @param columns
     *            number of columns
     * @return new 2D matrix with columns being the elements of this matrix.
     */
    public abstract FComplexMatrix3D reshape(int slices, int rows, int columns);

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * @param index
     *            the index of the cell.
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     * 
     * @throws IndexOutOfBoundsException
     *             if <tt>index&lt;0 || index&gt;=size()</tt>.
     */
    public void set(int index, float re, float im) {
        int size = (int) size();
        if (index < 0 || index >= size)
            checkIndex(index);
        setQuick(index, re, im);
    }

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * @param index
     *            the index of the cell.
     * @param value
     *            the value to be filled into the specified cell (re=value[0],
     *            im=value[1]).
     * 
     * @throws IndexOutOfBoundsException
     *             if <tt>index&lt;0 || index&gt;=size()</tt>.
     */
    public void set(int index, float[] value) {
        int size = (int) size();
        if (index < 0 || index >= size)
            checkIndex(index);
        setQuick(index, value);
    }

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     */
    public abstract void setQuick(int index, float re, float im);

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @param value
     *            the value to be filled into the specified cell (re=value[0],
     *            im=value[1]).
     */
    public abstract void setQuick(int index, float[] value);

    /**
     * Swaps each element <tt>this[i]</tt> with <tt>other[i]</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public void swap(final FComplexMatrix1D other) {
        int size = (int) size();
        checkSize(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        float[] tmp;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            tmp = getQuick(i);
                            setQuick(i, other.getQuick(i));
                            other.setQuick(i, tmp);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp;
            for (int i = 0; i < size; i++) {
                tmp = getQuick(i);
                setQuick(i, other.getQuick(i));
                other.setQuick(i, tmp);
            }
        }
    }

    /**
     * Constructs and returns a 1-dimensional array containing the cell values.
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa. The returned array
     * <tt>values</tt> has the form <br>
     * <tt>for (int i = 0; i < size; i++) {
     * 		  tmp = getQuick(i);
     * 		  values[2 * i] = tmp[0]; //real part
     * 		  values[2 * i + 1] = tmp[1]; //imaginary part
     * 	   }</tt>
     * 
     * @return an array filled with the values of the cells.
     */
    public float[] toArray() {
        int size = (int) size();
        float[] values = new float[2 * size];
        toArray(values);
        return values;
    }

    /**
     * Fills the cell values into the specified 1-dimensional array. The values
     * are copied. So subsequent changes in <tt>values</tt> are not reflected in
     * the matrix, and vice-versa. After this call returns the array
     * <tt>values</tt> has the form <br>
     * <tt>for (int i = 0; i < size; i++) {
     * 		  tmp = getQuick(i);
     * 		  values[2 * i] = tmp[0]; //real part
     * 		  values[2 * i + 1] = tmp[1]; //imaginary part
     * 	   }</tt>
     * 
     * @throws IllegalArgumentException
     *             if <tt>values.length < 2*size()</tt>.
     */
    public void toArray(final float[] values) {
        int size = (int) size();
        if (values.length < 2 * size)
            throw new IllegalArgumentException("values too small");
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        float[] tmp;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            tmp = getQuick(i);
                            values[2 * i] = tmp[0];
                            values[2 * i + 1] = tmp[1];
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp;
            for (int i = 0; i < size; i++) {
                tmp = getQuick(i);
                values[2 * i] = tmp[0];
                values[2 * i + 1] = tmp[1];
            }
        }
    }

    /**
     * Returns a string representation using default formatting ("%.4f").
     * 
     * @return a string representation of the matrix.
     */

    public String toString() {
        return toString("%.4f");
    }

    /**
     * Returns a string representation using given <tt>format</tt>
     * 
     * @param format
     *            a format for java.lang.String.format().
     * @return a string representation of the matrix.
     */
    public String toString(String format) {
        StringBuffer s = new StringBuffer(String.format("ComplexMatrix1D: %d elements\n\n", size()));
        float[] elem = new float[2];
        for (int i = 0; i < size(); i++) {
            elem = getQuick(i);
            if (elem[1] == 0) {
                s.append(String.format(format + "\n", elem[0]));
                continue;
            }
            if (elem[0] == 0) {
                s.append(String.format(format + "i\n", elem[1]));
                continue;
            }
            if (elem[1] < 0) {
                s.append(String.format(format + " - " + format + "i\n", elem[0], -elem[1]));
                continue;
            }
            s.append(String.format(format + " + " + format + "i\n", elem[0], elem[1]));
        }
        return s.toString();
    }

    /**
     * Constructs and returns a new <i>flip view</i>. What used to be index
     * <tt>0</tt> is now index <tt>size()-1</tt>, ..., what used to be index
     * <tt>size()-1</tt> is now index <tt>0</tt>. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * 
     * @return a new flip view.
     */
    public FComplexMatrix1D viewFlip() {
        return (FComplexMatrix1D) (view().vFlip());
    }

    /**
     * Constructs and returns a new <i>sub-range view</i> that is a
     * <tt>width</tt> sub matrix starting at <tt>index</tt>.
     * 
     * Operations on the returned view can only be applied to the restricted
     * range. Any attempt to access coordinates not contained in the view will
     * throw an <tt>IndexOutOfBoundsException</tt>.
     * <p>
     * <b>Note that the view is really just a range restriction:</b> The
     * returned matrix is backed by this matrix, so changes in the returned
     * matrix are reflected in this matrix, and vice-versa.
     * <p>
     * The view contains the cells from <tt>index..index+width-1</tt>. and has
     * <tt>view.size() == width</tt>. A view's legal coordinates are again zero
     * based, as usual. In other words, legal coordinates of the view are
     * <tt>0 .. view.size()-1==width-1</tt>. As usual, any attempt to access a
     * cell at other coordinates will throw an
     * <tt>IndexOutOfBoundsException</tt>.
     * 
     * @param index
     *            The index of the first cell.
     * @param width
     *            The width of the range.
     * @throws IndexOutOfBoundsException
     *             if <tt>index<0 || width<0 || index+width>size()</tt>.
     * @return the new view.
     * 
     */
    public FComplexMatrix1D viewPart(int index, int width) {
        return (FComplexMatrix1D) (view().vPart(index, width));
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding the cells matching the given condition. Applies the condition to
     * each cell and takes only those cells where
     * <tt>condition.apply(get(i))</tt> yields <tt>true</tt>.
     * 
     * The returned view is backed by this matrix, so changes in the returned
     * view are reflected in this matrix, and vice-versa.
     * 
     * @param condition
     *            The condition to be matched.
     * @return the new view.
     */
    public FComplexMatrix1D viewSelection(cern.colt.function.tfcomplex.FComplexProcedure condition) {
        int size = (int) size();
        IntArrayList matches = new IntArrayList();
        for (int i = 0; i < size; i++) {
            if (condition.apply(getQuick(i)))
                matches.add(i);
        }
        matches.trimToSize();
        return viewSelection(matches.elements());
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding the indicated cells. There holds
     * <tt>view.size() == indexes.length</tt> and
     * <tt>view.get(i) == this.get(indexes[i])</tt>. Indexes can occur multiple
     * times and can be in arbitrary order.
     * 
     * Note that modifying <tt>indexes</tt> after this call has returned has no
     * effect on the view. The returned view is backed by this matrix, so
     * changes in the returned view are reflected in this matrix, and
     * vice-versa.
     * 
     * @param indexes
     *            The indexes of the cells that shall be visible in the new
     *            view. To indicate that <i>all</i> cells shall be visible,
     *            simply set this parameter to <tt>null</tt>.
     * @return the new view.
     * @throws IndexOutOfBoundsException
     *             if <tt>!(0 <= indexes[i] < size())</tt> for any
     *             <tt>i=0..indexes.length()-1</tt>.
     */
    public FComplexMatrix1D viewSelection(int[] indexes) {
        // check for "all"
        int size = (int) size();
        if (indexes == null) {
            indexes = new int[size];
            for (int i = size - 1; --i >= 0;)
                indexes[i] = i;
        }

        checkIndexes(indexes);
        int[] offsets = new int[indexes.length];
        for (int i = 0; i < indexes.length; i++) {
            offsets[i] = (int) index(indexes[i]);
        }
        return viewSelectionLike(offsets);
    }

    /**
     * Constructs and returns a new <i>stride view</i> which is a sub matrix
     * consisting of every i-th cell. More specifically, the view has size
     * <tt>this.size()/stride</tt> holding cells <tt>this.get(i*stride)</tt> for
     * all <tt>i = 0..size()/stride - 1</tt>.
     * 
     * @param stride
     *            the step factor.
     * @throws IndexOutOfBoundsException
     *             if <tt>stride <= 0</tt>.
     * @return the new view.
     * 
     */
    public FComplexMatrix1D viewStrides(int stride) {
        return (FComplexMatrix1D) (view().vStrides(stride));
    }

    /**
     * Returns the dot product of two vectors x and y. Operates on cells at
     * indexes <tt>0 .. Math.min(size(),y.size())</tt>.
     * 
     * @param y
     *            the second vector.
     * @return the sum of products.
     */
    public float[] zDotProduct(FComplexMatrix1D y) {
        int size = (int) size();
        return zDotProduct(y, 0, size);
    }

    /**
     * Returns the dot product of two vectors x and y. Operates on cells at
     * indexes <tt>from .. Min(size(),y.size(),from+length)-1</tt>.
     * 
     * @param y
     *            the second vector.
     * @param from
     *            the first index to be considered.
     * @param length
     *            the number of cells to be considered.
     * @return the sum of products; zero if <tt>from<0 || length<0</tt>.
     */
    public float[] zDotProduct(final FComplexMatrix1D y, final int from, int length) {
        int size = (int) size();
        if (from < 0 || length <= 0)
            return new float[] { 0, 0 };

        int tail = from + length;
        if (size < tail)
            tail = size;
        if (y.size < tail)
            tail = y.size;
        length = tail - from;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        float[] sum = new float[2];

        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, length);
            Future<?>[] futures = new Future[nthreads];
            float[][] results = new float[nthreads][2];
            int k = length / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? length : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float[] sum = new float[2];
                        float[] tmp;
                        int idx;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            idx = k + from;
                            tmp = y.getQuick(idx);
                            tmp[1] = -tmp[1]; // complex conjugate
                            sum = FComplex.plus(sum, FComplex.mult(tmp, getQuick(idx)));
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (float[]) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum = FComplex.plus(sum, results[j]);
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            float[] tmp;
            int idx;
            for (int k = 0; k < length; k++) {
                idx = k + from;
                tmp = y.getQuick(idx);
                tmp[1] = -tmp[1]; // complex conjugate
                sum = FComplex.plus(sum, FComplex.mult(tmp, getQuick(idx)));
            }
        }
        return sum;
    }

    /**
     * Returns the dot product of two vectors x and y.
     * 
     * @param y
     *            the second vector.
     * @param nonZeroIndexes
     *            the indexes of cells in <tt>y</tt>having a non-zero value.
     * @return the sum of products.
     */
    public float[] zDotProduct(FComplexMatrix1D y, int from, int length, IntArrayList nonZeroIndexes) {
        int size = (int) size();
        if (from < 0 || length <= 0)
            return new float[] { 0, 0 };

        int tail = from + length;
        if (size < tail)
            tail = size;
        if (y.size < tail)
            tail = y.size;
        length = tail - from;
        if (length <= 0)
            return new float[] { 0, 0 };

        // setup
        IntArrayList indexesCopy = nonZeroIndexes.copy();
        indexesCopy.trimToSize();
        indexesCopy.quickSort();
        int[] nonZeroIndexElements = indexesCopy.elements();
        int index = 0;
        int s = indexesCopy.size();

        // skip to start
        while ((index < s) && nonZeroIndexElements[index] < from)
            index++;

        // now the sparse dot product
        int i;
        float[] sum = new float[2];
        float[] tmp;
        while ((--length >= 0) && (index < s) && ((i = nonZeroIndexElements[index]) < tail)) {
            tmp = y.getQuick(i);
            tmp[1] = -tmp[1]; // complex conjugate
            sum = FComplex.plus(sum, FComplex.mult(tmp, getQuick(i)));
            index++;
        }

        return sum;
    }

    /**
     * Returns the sum of all cells.
     * 
     * @return the sum.
     */
    public float[] zSum() {
        float[] sum = new float[2];
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            float[][] results = new float[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float[] sum = new float[2];
                        for (int k = firstIdx; k < lastIdx; k++) {
                            sum = FComplex.plus(sum, getQuick(k));
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (float[]) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum[0] += results[j][0];
                    sum[1] += results[j][1];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            for (int k = 0; k < size; k++) {
                sum = FComplex.plus(sum, getQuick(k));
            }
        }
        return sum;
    }

    /**
     * Returns the number of cells having non-zero values, but at most
     * maxCardinality; ignores tolerance.
     * 
     * @param maxCardinality
     *            maximal cardinality
     * @return number of cells having non-zero values, but at most
     *         maxCardinality.
     */
    protected int cardinality(int maxCardinality) {
        int size = (int) size();
        int cardinality = 0;
        int i = 0;
        float[] tmp = new float[2];
        while (i++ < size && cardinality < maxCardinality) {
            tmp = getQuick(i);
            if ((tmp[0] != 0.0) || (tmp[1] != 0.0))
                cardinality++;
        }
        return cardinality;
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     */
    protected FComplexMatrix1D getContent() {
        return this;
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     * 
     * @param other
     *            matrix
     * @return <tt>true</tt> if both matrices share at least one identical cell
     */
    protected boolean haveSharedCells(FComplexMatrix1D other) {
        if (other == null)
            return false;
        if (this == other)
            return true;
        return getContent().haveSharedCellsRaw(other.getContent());
    }

    /**
     * Always returns false
     * 
     * @param other
     *            matrix
     * @return false
     */
    protected boolean haveSharedCellsRaw(FComplexMatrix1D other) {
        return false;
    }

    /**
     * Constructs and returns a new view equal to the receiver. The view is a
     * shallow clone. Calls <code>clone()</code> and casts the result.
     * <p>
     * <b>Note that the view is not a deep copy.</b> The returned matrix is
     * backed by this matrix, so changes in the returned matrix are reflected in
     * this matrix, and vice-versa.
     * <p>
     * Use {@link #copy()} to construct an independent deep copy rather than a
     * new view.
     * 
     * @return a new view of the receiver.
     */
    protected FComplexMatrix1D view() {
        return (FComplexMatrix1D) clone();
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param offsets
     *            the offsets of the visible elements.
     * @return a new view.
     */
    protected abstract FComplexMatrix1D viewSelectionLike(int[] offsets);

    /**
     * Returns the dot product of two vectors x and y.
     * 
     * @param y
     *            the second vector.
     * @param nonZeroIndexes
     *            the indexes of cells in <tt>y</tt>having a non-zero value.
     * @return the sum of products.
     */
    protected float[] zDotProduct(FComplexMatrix1D y, IntArrayList nonZeroIndexes) {
        return zDotProduct(y, 0, (int) size(), nonZeroIndexes);
    }

}
