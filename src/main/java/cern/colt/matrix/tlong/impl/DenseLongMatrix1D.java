/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.matrix.tlong.LongMatrix1D;
import cern.colt.matrix.tlong.LongMatrix2D;
import cern.colt.matrix.tlong.LongMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 1-d matrix (aka <i>vector</i>) holding <tt>int</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Longernally holds one single contigous one-dimensional array. Note that this
 * implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*size()</tt>. Thus, a 1000000 matrix uses 8 MB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 * <p>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseLongMatrix1D extends LongMatrix1D {
    private static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix.
     */
    protected long[] elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public DenseLongMatrix1D(long[] values) {
        this(values.length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of cells. All entries are
     * initially <tt>0</tt>.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public DenseLongMatrix1D(int size) {
        setUp(size);
        this.elements = new long[size];
    }

    /**
     * Constructs a matrix with the given parameters.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param elements
     *            the cells.
     * @param zero
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @param isView
     *            if true then a matrix view is constructed
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public DenseLongMatrix1D(int size, long[] elements, int zero, int stride, boolean isView) {
        setUp(size, zero, stride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f) {
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        long a = f.apply(elements[idx]);
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            a = aggr.apply(a, f.apply(elements[idx]));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero]);
            int idx = zero;
            for (int i = 1; i < size; i++) {
                idx += stride;
                a = aggr.apply(a, f.apply(elements[idx]));
            }
        }
        return a;
    }

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f, final IntArrayList indexList) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int size = indexList.size();
        final int[] indexElements = indexList.elements();
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        int idx = zero + indexElements[firstIdx] * stride;
                        long a = f.apply(elements[idx]);
                        long elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx = zero + indexElements[i] * stride;
                            elem = elements[idx];
                            a = aggr.apply(a, f.apply(elem));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            long elem;
            int idx = zero + indexElements[0] * stride;
            a = f.apply(elements[idx]);
            for (int i = 1; i < size; i++) {
                idx = zero + indexElements[i] * stride;
                elem = elements[idx];
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    public long aggregate(final LongMatrix1D other, final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongLongFunction f) {
        if (!(other instanceof DenseLongMatrix1D)) {
            return super.aggregate(other, aggr, f);
        }
        checkSize(other);
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        final long[] elemsOther = (long[]) other.elements();
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        long a = f.apply(elements[idx], elemsOther[idxOther]);
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            idxOther += strideOther;
                            a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero], elemsOther[zeroOther]);
            int idx = zero;
            int idxOther = zeroOther;
            for (int i = 1; i < size; i++) {
                idx += stride;
                idxOther += strideOther;
                a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
            }
        }
        return a;
    }

    public LongMatrix1D assign(final cern.colt.function.tlong.LongFunction function) {
        final long multiplicator;
        if (function instanceof cern.jet.math.tlong.LongMult) {
            // x[i] = mult*x[i]
            multiplicator = ((cern.jet.math.tlong.LongMult) function).multiplicator;
            if (multiplicator == 1) {
                return this;
            }
        } else {
            multiplicator = 0;
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
                        int idx = zero + firstIdx * stride;
                        // specialization for speed
                        if (function instanceof cern.jet.math.tlong.LongMult) {
                            // x[i] = mult*x[i]
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] *= multiplicator;
                                idx += stride;
                            }
                        } else {
                            // the general case x[i] = f(x[i])
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] = function.apply(elements[idx]);
                                idx += stride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero - stride;
            // specialization for speed
            if (function instanceof cern.jet.math.tlong.LongMult) {
                // x[i] = mult*x[i]
                for (int k = size; --k >= 0;) {
                    elements[idx += stride] *= multiplicator;
                }
            } else {
                // the general case x[i] = f(x[i])
                for (int k = size; --k >= 0;) {
                    elements[idx += stride] = function.apply(elements[idx]);
                }
            }
        }
        return this;
    }

    public LongMatrix1D assign(final cern.colt.function.tlong.LongProcedure cond,
            final cern.colt.function.tlong.LongFunction function) {
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
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            if (cond.apply(elements[idx]) == true) {
                                elements[idx] = function.apply(elements[idx]);
                            }
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int i = 0; i < size; i++) {
                if (cond.apply(elements[idx]) == true) {
                    elements[idx] = function.apply(elements[idx]);
                }
                idx += stride;
            }
        }
        return this;
    }

    public LongMatrix1D assign(final cern.colt.function.tlong.LongProcedure cond, final long value) {
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
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            if (cond.apply(elements[idx]) == true) {
                                elements[idx] = value;
                            }
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int i = 0; i < size; i++) {
                if (cond.apply(elements[idx]) == true) {
                    elements[idx] = value;
                }
                idx += stride;
            }
        }
        return this;
    }

    public LongMatrix1D assign(final long value) {
        final long[] elems = this.elements;
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
                        int idx = zero + firstIdx * stride;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            elems[idx] = value;
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int i = 0; i < size; i++) {
                elems[idx] = value;
                idx += stride;
            }
        }
        return this;
    }

    public LongMatrix1D assign(final long[] values) {
        if (values.length != size)
            throw new IllegalArgumentException("Must have same number of cells: length=" + values.length + "size()="
                    + size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isNoView) {
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
                nthreads = Math.min(nthreads, size);
                Future<?>[] futures = new Future[nthreads];
                int k = size / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx;
                    if (j == nthreads - 1) {
                        lastIdx = size;
                    } else {
                        lastIdx = firstIdx + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            int idx = zero + firstIdx * stride;
                            for (int i = firstIdx; i < lastIdx; i++) {
                                elements[idx] = values[i];
                                idx += stride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = zero;
                for (int i = 0; i < size; i++) {
                    elements[idx] = values[i];
                    idx += stride;
                }
            }
        }
        return this;
    }

    public LongMatrix1D assign(final int[] values) {
        if (values.length != size)
            throw new IllegalArgumentException("Must have same number of cells: length=" + values.length + "size()="
                    + size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx;
                if (j == nthreads - 1) {
                    lastIdx = size;
                } else {
                    lastIdx = firstIdx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elements[idx] = values[i];
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int i = 0; i < size; i++) {
                elements[idx] = values[i];
                idx += stride;
            }
        }
        return this;
    }

    public LongMatrix1D assign(LongMatrix1D source) {
        // overriden for performance only
        if (!(source instanceof DenseLongMatrix1D)) {
            super.assign(source);
            return this;
        }
        DenseLongMatrix1D other = (DenseLongMatrix1D) source;
        if (other == this)
            return this;
        checkSize(other);
        if (isNoView && other.isNoView) {
            // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            LongMatrix1D c = other.copy();
            if (!(c instanceof DenseLongMatrix1D)) {
                // should not happen
                super.assign(source);
                return this;
            }
            other = (DenseLongMatrix1D) c;
        }

        final long[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride;
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
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            elements[idx] = elemsOther[idxOther];
                            idx += stride;
                            idxOther += strideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int k = 0; k < size; k++) {
                elements[idx] = elemsOther[idxOther];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return this;
    }

    public LongMatrix1D assign(final LongMatrix1D y, final cern.colt.function.tlong.LongLongFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseLongMatrix1D)) {
            super.assign(y, function);
            return this;
        }
        checkSize(y);
        final int zeroOther = (int) y.index(0);
        final int strideOther = y.stride();
        final long[] elemsOther = (long[]) y.elements();
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
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        // specialized for speed
                        if (function == cern.jet.math.tlong.LongFunctions.mult) {
                            // x[i] = x[i] * y[i]
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] *= elemsOther[idxOther];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tlong.LongFunctions.div) {
                            // x[i] = x[i] / y[i]
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] /= elemsOther[idxOther];
                                idx += stride;
                                idxOther += strideOther;

                            }
                        } else if (function instanceof cern.jet.math.tlong.LongPlusMultFirst) {
                            long multiplicator = ((cern.jet.math.tlong.LongPlusMultFirst) function).multiplicator;
                            if (multiplicator == 0) {
                                // x[i] = 0*x[i] + y[i]
                                for (int k = firstIdx; k < lastIdx; k++) {
                                    elements[idx] = elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                for (int k = firstIdx; k < lastIdx; k++) {
                                    elements[idx] += elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = -x[i] + y[i]
                                for (int k = firstIdx; k < lastIdx; k++) {
                                    elements[idx] = elemsOther[idxOther] - elements[idx];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else {
                                // the general case x[i] = mult*x[i] + y[i]
                                for (int k = firstIdx; k < lastIdx; k++) {
                                    elements[idx] = multiplicator * elements[idx] + elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            }
                        } else if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
                            long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
                            if (multiplicator == 0) {
                                // x[i] = x[i] + 0*y[i]
                                return;
                            } else if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                for (int k = firstIdx; k < lastIdx; k++) {
                                    elements[idx] += elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = x[i] - y[i]
                                for (int k = firstIdx; k < lastIdx; k++) {
                                    elements[idx] -= elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else {
                                // the general case x[i] = x[i] + mult*y[i]
                                for (int k = firstIdx; k < lastIdx; k++) {
                                    elements[idx] += multiplicator * elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }

                            }
                        } else {
                            // the general case x[i] = f(x[i],y[i])
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
                                idx += stride;
                                idxOther += strideOther;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            // specialized for speed
            int idx = zero;
            int idxOther = zeroOther;
            if (function == cern.jet.math.tlong.LongFunctions.mult) {
                // x[i] = x[i] * y[i]
                for (int k = 0; k < size; k++) {
                    elements[idx] *= elemsOther[idxOther];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tlong.LongFunctions.div) {
                // x[i] = x[i] / y[i]
                for (int k = 0; k < size; k++) {
                    elements[idx] /= elemsOther[idxOther];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
                long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
                if (multiplicator == 0) {
                    // x[i] = x[i] + 0*y[i]
                    return this;
                } else if (multiplicator == 1) {
                    // x[i] = x[i] + y[i]
                    for (int k = 0; k < size; k++) {
                        elements[idx] += elemsOther[idxOther];
                        idx += stride;
                        idxOther += strideOther;
                    }
                } else if (multiplicator == -1) {
                    // x[i] = x[i] - y[i]
                    for (int k = 0; k < size; k++) {
                        elements[idx] -= elemsOther[idxOther];
                        idx += stride;
                        idxOther += strideOther;
                    }
                } else {
                    // the general case x[i] = x[i] + mult*y[i]
                    for (int k = 0; k < size; k++) {
                        elements[idx] += multiplicator * elemsOther[idxOther];
                        idx += stride;
                        idxOther += strideOther;
                    }
                }
            } else {
                // the general case x[i] = f(x[i],y[i])
                for (int k = 0; k < size; k++) {
                    elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
                    idx += stride;
                    idxOther += strideOther;
                }
            }
        }
        return this;
    }

    public int cardinality() {
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
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            if (elements[idx] != 0)
                                cardinality++;
                            idx += stride;
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
            int idx = zero;
            for (int i = 0; i < size; i++) {
                if (elements[idx] != 0)
                    cardinality++;
                idx += stride;
            }
        }
        return cardinality;
    }

    public long[] elements() {
        return elements;
    }

    public void getNonZeros(final LongArrayList indexList, final LongArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int idx = zero;
        int rem = size % 2;
        if (rem == 1) {
            long value = elements[idx];
            if (value != 0) {
                indexList.add(0);
                valueList.add(value);
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            long value = elements[idx];
            if (value != 0) {
                indexList.add(i);
                valueList.add(value);
            }
            idx += stride;
            value = elements[idx];
            if (value != 0) {
                indexList.add(i + 1);
                valueList.add(value);
            }
            idx += stride;
        }
    }

    public void getPositiveValues(final LongArrayList indexList, final LongArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int idx = zero;
        int rem = size % 2;
        if (rem == 1) {
            long value = elements[idx];
            if (value > 0) {
                indexList.add(0);
                valueList.add(value);
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            long value = elements[idx];
            if (value > 0) {
                indexList.add(i);
                valueList.add(value);
            }
            idx += stride;
            value = elements[idx];
            if (value > 0) {
                indexList.add(i + 1);
                valueList.add(value);
            }
            idx += stride;
        }
    }

    public void getNegativeValues(final LongArrayList indexList, final LongArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int idx = zero;
        int rem = size % 2;
        if (rem == 1) {
            long value = elements[idx];
            if (value < 0) {
                indexList.add(0);
                valueList.add(value);
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            long value = elements[idx];
            if (value < 0) {
                indexList.add(i);
                valueList.add(value);
            }
            idx += stride;
            value = elements[idx];
            if (value < 0) {
                indexList.add(i + 1);
                valueList.add(value);
            }
            idx += stride;
        }
    }

    public long[] getMaxLocation() {
        int location = 0;
        long maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        long maxValue = elements[idx];
                        int location = (idx - zero) / stride;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            if (maxValue < elements[idx]) {
                                maxValue = elements[idx];
                                location = (idx - zero) / stride;
                            }
                        }
                        return new long[] { maxValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                maxValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < nthreads; j++) {
                    if (maxValue < results[j][0]) {
                        maxValue = results[j][0];
                        location = (int) results[j][1];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            maxValue = elements[zero];
            location = 0;
            int idx = zero;
            for (int i = 1; i < size; i++) {
                idx += stride;
                if (maxValue < elements[idx]) {
                    maxValue = elements[idx];
                    location = (idx - zero) / stride;
                }
            }
        }
        return new long[] { maxValue, location };
    }

    public long[] getMinLocation() {
        int location = 0;
        long minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        long minValue = elements[idx];
                        int location = (idx - zero) / stride;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            if (minValue > elements[idx]) {
                                minValue = elements[idx];
                                location = (idx - zero) / stride;
                            }
                        }
                        return new long[] { minValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                minValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < nthreads; j++) {
                    if (minValue > results[j][0]) {
                        minValue = results[j][0];
                        location = (int) results[j][1];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            minValue = elements[zero];
            location = 0;
            int idx = zero;
            for (int i = 1; i < size; i++) {
                idx += stride;
                if (minValue > elements[idx]) {
                    minValue = elements[idx];
                    location = (idx - zero) / stride;
                }
            }
        }
        return new long[] { minValue, location };
    }

    public long getQuick(int index) {
        return elements[zero + index * stride];
    }

    public LongMatrix1D like(int size) {
        return new DenseLongMatrix1D(size);
    }

    public LongMatrix2D like2D(int rows, int columns) {
        return new DenseLongMatrix2D(rows, columns);
    }

    public LongMatrix2D reshape(final int rows, final int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        LongMatrix2D M = new DenseLongMatrix2D(rows, columns);
        final long[] elemsOther = (long[]) M.elements();
        final int zeroOther = (int) M.index(0, 0);
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        int idxOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            idxOther = zeroOther + c * colStrideOther;
                            idx = zero + (c * rows) * stride;
                            for (int r = 0; r < rows; r++) {
                                elemsOther[idxOther] = elements[idx];
                                idxOther += rowStrideOther;
                                idx += stride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxOther;
            int idx = zero;
            for (int c = 0; c < columns; c++) {
                idxOther = zeroOther + c * colStrideOther;
                for (int r = 0; r < rows; r++) {
                    elemsOther[idxOther] = elements[idx];
                    idxOther += rowStrideOther;
                    idx += stride;
                }
            }
        }
        return M;
    }

    public LongMatrix3D reshape(final int slices, final int rows, final int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        LongMatrix3D M = new DenseLongMatrix3D(slices, rows, columns);
        final long[] elemsOther = (long[]) M.elements();
        final int zeroOther = (int) M.index(0, 0, 0);
        final int sliceStrideOther = M.sliceStride();
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        int idxOther;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
                                idx = zero + (s * rows * columns + c * rows) * stride;
                                for (int r = 0; r < rows; r++) {
                                    elemsOther[idxOther] = elements[idx];
                                    idxOther += rowStrideOther;
                                    idx += stride;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxOther;
            int idx = zero;
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
                    for (int r = 0; r < rows; r++) {
                        elemsOther[idxOther] = elements[idx];
                        idxOther += rowStrideOther;
                        idx += stride;
                    }
                }
            }
        }
        return M;
    }

    public void setQuick(int index, long value) {
        elements[zero + index * stride] = value;
    }

    public void swap(final LongMatrix1D other) {
        // overriden for performance only
        if (!(other instanceof DenseLongMatrix1D)) {
            super.swap(other);
        }
        DenseLongMatrix1D y = (DenseLongMatrix1D) other;
        if (y == this)
            return;
        checkSize(y);
        final long[] elemsOther = y.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
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
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            long tmp = elements[idx];
                            elements[idx] = elemsOther[idxOther];
                            elemsOther[idxOther] = tmp;
                            idx += stride;
                            idxOther += strideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int k = 0; k < size; k++) {
                long tmp = elements[idx];
                elements[idx] = elemsOther[idxOther];
                elemsOther[idxOther] = tmp;
                idx += stride;
                idxOther += strideOther;
            }
        }
    }

    public void toArray(long[] values) {
        if (values.length < size)
            throw new IllegalArgumentException("values too small");
        if (this.isNoView)
            System.arraycopy(this.elements, 0, values, 0, this.elements.length);
        else
            super.toArray(values);
    }

    public long zDotProduct(LongMatrix1D y) {
        if (!(y instanceof DenseLongMatrix1D)) {
            return super.zDotProduct(y);
        }
        DenseLongMatrix1D yy = (DenseLongMatrix1D) y;
        final long[] elemsOther = yy.elements;
        int zeroThis = (int) index(0);
        int zeroOther = (int) yy.index(0);
        int strideOther = yy.stride;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        long sum = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            final int zeroThisF = zeroThis;
            final int zeroOtherF = zeroOther;
            final int strideOtherF = strideOther;
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Long[] results = new Long[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        int idx = zeroThisF + firstIdx * stride;
                        int idxOther = zeroOtherF + firstIdx * strideOtherF;
                        idx -= stride;
                        idxOther -= strideOtherF;
                        long sum = 0;
                        int min = lastIdx - firstIdx;
                        for (int k = min / 4; --k >= 0;) {
                            sum += elements[idx += stride] * elemsOther[idxOther += strideOtherF]
                                    + elements[idx += stride] * elemsOther[idxOther += strideOtherF]
                                    + elements[idx += stride] * elemsOther[idxOther += strideOtherF]
                                    + elements[idx += stride] * elemsOther[idxOther += strideOtherF];
                        }
                        for (int k = min % 4; --k >= 0;) {
                            sum += elements[idx += stride] * elemsOther[idxOther += strideOtherF];
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Long) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum += results[j];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            zeroThis -= stride;
            zeroOther -= strideOther;
            for (int k = size / 4; --k >= 0;) {
                sum += elements[zeroThis += stride] * elemsOther[zeroOther += strideOther]
                        + elements[zeroThis += stride] * elemsOther[zeroOther += strideOther]
                        + elements[zeroThis += stride] * elemsOther[zeroOther += strideOther]
                        + elements[zeroThis += stride] * elemsOther[zeroOther += strideOther];
            }
            for (int k = size % 4; --k >= 0;) {
                sum += elements[zeroThis += stride] * elemsOther[zeroOther += strideOther];
            }
        }
        return sum;
    }

    public long zDotProduct(LongMatrix1D y, int from, int length) {
        if (!(y instanceof DenseLongMatrix1D)) {
            return super.zDotProduct(y, from, length);
        }
        DenseLongMatrix1D yy = (DenseLongMatrix1D) y;

        int tail = from + length;
        if (from < 0 || length < 0)
            return 0;
        if (size < tail)
            tail = size;
        if (y.size() < tail)
            tail = (int) y.size();
        final long[] elementsOther = yy.elements;
        int zeroThis = (int) index(from);
        int zeroOther = (int) yy.index(from);
        int strideOther = yy.stride;
        if (elements == null || elementsOther == null)
            throw new InternalError();
        long sum = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (length >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            final int zeroThisF = zeroThis;
            final int zeroOtherF = zeroOther;
            final int strideOtherF = strideOther;
            nthreads = Math.min(nthreads, length);
            Future<?>[] futures = new Future[nthreads];
            Long[] results = new Long[nthreads];
            int k = length / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? length : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        int idx = zeroThisF + firstIdx * stride;
                        int idxOther = zeroOtherF + firstIdx * strideOtherF;
                        idx -= stride;
                        idxOther -= strideOtherF;
                        long sum = 0;
                        int min = lastIdx - firstIdx;
                        for (int k = min / 4; --k >= 0;) {
                            sum += elements[idx += stride] * elementsOther[idxOther += strideOtherF]
                                    + elements[idx += stride] * elementsOther[idxOther += strideOtherF]
                                    + elements[idx += stride] * elementsOther[idxOther += strideOtherF]
                                    + elements[idx += stride] * elementsOther[idxOther += strideOtherF];
                        }
                        for (int k = min % 4; --k >= 0;) {
                            sum += elements[idx += stride] * elementsOther[idxOther += strideOtherF];
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Long) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum += results[j];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            zeroThis -= stride;
            zeroOther -= strideOther;
            int min = tail - from;
            for (int k = min / 4; --k >= 0;) {
                sum += elements[zeroThis += stride] * elementsOther[zeroOther += strideOther]
                        + elements[zeroThis += stride] * elementsOther[zeroOther += strideOther]
                        + elements[zeroThis += stride] * elementsOther[zeroOther += strideOther]
                        + elements[zeroThis += stride] * elementsOther[zeroOther += strideOther];
            }
            for (int k = min % 4; --k >= 0;) {
                sum += elements[zeroThis += stride] * elementsOther[zeroOther += strideOther];
            }
        }
        return sum;
    }

    public long zSum() {
        long sum = 0;
        final long[] elems = this.elements;
        if (elems == null)
            throw new InternalError();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Long[] results = new Long[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        int sum = 0;
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            sum += elems[idx];
                            idx += stride;
                        }
                        return Long.valueOf(sum);
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Long) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum += results[j];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int idx = zero;
            for (int k = 0; k < size; k++) {
                sum += elems[idx];
                idx += stride;
            }
        }
        return sum;
    }

    protected int cardinality(int maxCardinality) {
        int cardinality = 0;
        int index = zero;
        long[] elems = this.elements;
        int i = size;
        while (--i >= 0 && cardinality < maxCardinality) {
            if (elems[index] != 0)
                cardinality++;
            index += stride;
        }
        return cardinality;
    }

    protected boolean haveSharedCellsRaw(LongMatrix1D other) {
        if (other instanceof SelectedDenseLongMatrix1D) {
            SelectedDenseLongMatrix1D otherMatrix = (SelectedDenseLongMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseLongMatrix1D) {
            DenseLongMatrix1D otherMatrix = (DenseLongMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        return zero + rank * stride;
    }

    protected LongMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseLongMatrix1D(this.elements, offsets);
    }
}
