/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import cern.colt.matrix.tint.IntMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 1-d matrix (aka <i>vector</i>) holding <tt>int</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array. Note that this
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
public class DenseIntMatrix1D extends IntMatrix1D {
    private static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix.
     */
    protected int[] elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public DenseIntMatrix1D(int[] values) {
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
    public DenseIntMatrix1D(int size) {
        setUp(size);
        this.elements = new int[size];
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
    public DenseIntMatrix1D(int size, int[] elements, int zero, int stride, boolean isView) {
        setUp(size, zero, stride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    public int aggregate(final cern.colt.function.tint.IntIntFunction aggr, final cern.colt.function.tint.IntFunction f) {
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        int a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        int a = f.apply(elements[idx]);
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

    public int aggregate(final cern.colt.function.tint.IntIntFunction aggr,
            final cern.colt.function.tint.IntFunction f, final IntArrayList indexList) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int size = indexList.size();
        final int[] indexElements = indexList.elements();
        int a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {

                    public Integer call() throws Exception {
                        int idx = zero + indexElements[firstIdx] * stride;
                        int a = f.apply(elements[idx]);
                        int elem;
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
            int elem;
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

    public int aggregate(final IntMatrix1D other, final cern.colt.function.tint.IntIntFunction aggr,
            final cern.colt.function.tint.IntIntFunction f) {
        if (!(other instanceof DenseIntMatrix1D)) {
            return super.aggregate(other, aggr, f);
        }
        checkSize(other);
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        final int[] elemsOther = (int[]) other.elements();
        int a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        int a = f.apply(elements[idx], elemsOther[idxOther]);
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

    public IntMatrix1D assign(final cern.colt.function.tint.IntFunction function) {
        final int multiplicator;
        if (function instanceof cern.jet.math.tint.IntMult) {
            // x[i] = mult*x[i]
            multiplicator = ((cern.jet.math.tint.IntMult) function).multiplicator;
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
                        if (function instanceof cern.jet.math.tint.IntMult) {
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
            if (function instanceof cern.jet.math.tint.IntMult) {
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

    public IntMatrix1D assign(final cern.colt.function.tint.IntProcedure cond,
            final cern.colt.function.tint.IntFunction function) {
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

    public IntMatrix1D assign(final cern.colt.function.tint.IntProcedure cond, final int value) {
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

    public IntMatrix1D assign(final int value) {
        final int[] elems = this.elements;
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

    public IntMatrix1D assign(final int[] values) {
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

    public IntMatrix1D assign(IntMatrix1D source) {
        // overriden for performance only
        if (!(source instanceof DenseIntMatrix1D)) {
            super.assign(source);
            return this;
        }
        DenseIntMatrix1D other = (DenseIntMatrix1D) source;
        if (other == this)
            return this;
        checkSize(other);
        if (isNoView && other.isNoView) {
            // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            IntMatrix1D c = other.copy();
            if (!(c instanceof DenseIntMatrix1D)) {
                // should not happen
                super.assign(source);
                return this;
            }
            other = (DenseIntMatrix1D) c;
        }

        final int[] elemsOther = other.elements;
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

    public IntMatrix1D assign(final IntMatrix1D y, final cern.colt.function.tint.IntIntFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseIntMatrix1D)) {
            super.assign(y, function);
            return this;
        }
        checkSize(y);
        final int zeroOther = (int) y.index(0);
        final int strideOther = y.stride();
        final int[] elemsOther = (int[]) y.elements();
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
                        if (function == cern.jet.math.tint.IntFunctions.mult) {
                            // x[i] = x[i] * y[i]
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] *= elemsOther[idxOther];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tint.IntFunctions.div) {
                            // x[i] = x[i] / y[i]
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] /= elemsOther[idxOther];
                                idx += stride;
                                idxOther += strideOther;

                            }
                        } else if (function instanceof cern.jet.math.tint.IntPlusMultFirst) {
                            int multiplicator = ((cern.jet.math.tint.IntPlusMultFirst) function).multiplicator;
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
                        } else if (function instanceof cern.jet.math.tint.IntPlusMultSecond) {
                            int multiplicator = ((cern.jet.math.tint.IntPlusMultSecond) function).multiplicator;
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
            if (function == cern.jet.math.tint.IntFunctions.mult) {
                // x[i] = x[i] * y[i]
                for (int k = 0; k < size; k++) {
                    elements[idx] *= elemsOther[idxOther];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tint.IntFunctions.div) {
                // x[i] = x[i] / y[i]
                for (int k = 0; k < size; k++) {
                    elements[idx] /= elemsOther[idxOther];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function instanceof cern.jet.math.tint.IntPlusMultSecond) {
                int multiplicator = ((cern.jet.math.tint.IntPlusMultSecond) function).multiplicator;
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

    public int[] elements() {
        return elements;
    }

    public void getNonZeros(final IntArrayList indexList, final IntArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int idx = zero;
        int rem = size % 2;
        if (rem == 1) {
            int value = elements[idx];
            if (value != 0) {
                indexList.add(0);
                valueList.add(value);
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            int value = elements[idx];
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

    public void getPositiveValues(final IntArrayList indexList, final IntArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int idx = zero;
        int rem = size % 2;
        if (rem == 1) {
            int value = elements[idx];
            if (value > 0) {
                indexList.add(0);
                valueList.add(value);
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            int value = elements[idx];
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

    public void getNegativeValues(final IntArrayList indexList, final IntArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int idx = zero;
        int rem = size % 2;
        if (rem == 1) {
            int value = elements[idx];
            if (value < 0) {
                indexList.add(0);
                valueList.add(value);
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            int value = elements[idx];
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

    public int[] getMaxLocation() {
        int location = 0;
        int maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int[][] results = new int[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<int[]>() {
                    public int[] call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        int maxValue = elements[idx];
                        int location = (idx - zero) / stride;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            if (maxValue < elements[idx]) {
                                maxValue = elements[idx];
                                location = (idx - zero) / stride;
                            }
                        }
                        return new int[] { maxValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (int[]) futures[j].get();
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
        return new int[] { maxValue, location };
    }

    public int[] getMinLocation() {
        int location = 0;
        int minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int[][] results = new int[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<int[]>() {
                    public int[] call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        int minValue = elements[idx];
                        int location = (idx - zero) / stride;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            if (minValue > elements[idx]) {
                                minValue = elements[idx];
                                location = (idx - zero) / stride;
                            }
                        }
                        return new int[] { minValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (int[]) futures[j].get();
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
        return new int[] { minValue, location };
    }

    public int getQuick(int index) {
        return elements[zero + index * stride];
    }

    public IntMatrix1D like(int size) {
        return new DenseIntMatrix1D(size);
    }

    public IntMatrix2D like2D(int rows, int columns) {
        return new DenseIntMatrix2D(rows, columns);
    }

    public IntMatrix2D reshape(final int rows, final int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        IntMatrix2D M = new DenseIntMatrix2D(rows, columns);
        final int[] elemsOther = (int[]) M.elements();
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

    public IntMatrix3D reshape(final int slices, final int rows, final int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        IntMatrix3D M = new DenseIntMatrix3D(slices, rows, columns);
        final int[] elemsOther = (int[]) M.elements();
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

    public void setQuick(int index, int value) {
        elements[zero + index * stride] = value;
    }

    public void swap(final IntMatrix1D other) {
        // overriden for performance only
        if (!(other instanceof DenseIntMatrix1D)) {
            super.swap(other);
        }
        DenseIntMatrix1D y = (DenseIntMatrix1D) other;
        if (y == this)
            return;
        checkSize(y);
        final int[] elemsOther = y.elements;
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
                            int tmp = elements[idx];
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
                int tmp = elements[idx];
                elements[idx] = elemsOther[idxOther];
                elemsOther[idxOther] = tmp;
                idx += stride;
                idxOther += strideOther;
            }
        }
    }

    public void toArray(int[] values) {
        if (values.length < size)
            throw new IllegalArgumentException("values too small");
        if (this.isNoView)
            System.arraycopy(this.elements, 0, values, 0, this.elements.length);
        else
            super.toArray(values);
    }

    public int zDotProduct(IntMatrix1D y) {
        if (!(y instanceof DenseIntMatrix1D)) {
            return super.zDotProduct(y);
        }
        DenseIntMatrix1D yy = (DenseIntMatrix1D) y;
        final int[] elemsOther = yy.elements;
        int zeroThis = (int) index(0);
        int zeroOther = (int) yy.index(0);
        int strideOther = yy.stride;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        int sum = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            final int zeroThisF = zeroThis;
            final int zeroOtherF = zeroOther;
            final int strideOtherF = strideOther;
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int idx = zeroThisF + firstIdx * stride;
                        int idxOther = zeroOtherF + firstIdx * strideOtherF;
                        idx -= stride;
                        idxOther -= strideOtherF;
                        int sum = 0;
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
                    results[j] = (Integer) futures[j].get();
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

    public int zDotProduct(IntMatrix1D y, int from, int length) {
        if (!(y instanceof DenseIntMatrix1D)) {
            return super.zDotProduct(y, from, length);
        }
        DenseIntMatrix1D yy = (DenseIntMatrix1D) y;

        int tail = from + length;
        if (from < 0 || length < 0)
            return 0;
        if (size < tail)
            tail = size;
        if (y.size() < tail)
            tail = (int) y.size();
        final int[] elementsOther = yy.elements;
        int zeroThis = (int) index(from);
        int zeroOther = (int) yy.index(from);
        int strideOther = yy.stride;
        if (elements == null || elementsOther == null)
            throw new InternalError();
        int sum = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (length >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            final int zeroThisF = zeroThis;
            final int zeroOtherF = zeroOther;
            final int strideOtherF = strideOther;
            nthreads = Math.min(nthreads, length);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = length / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? length : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int idx = zeroThisF + firstIdx * stride;
                        int idxOther = zeroOtherF + firstIdx * strideOtherF;
                        idx -= stride;
                        idxOther -= strideOtherF;
                        int sum = 0;
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
                    results[j] = (Integer) futures[j].get();
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

    public int zSum() {
        int sum = 0;
        final int[] elems = this.elements;
        if (elems == null)
            throw new InternalError();
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
                        int sum = 0;
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            sum += elems[idx];
                            idx += stride;
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Integer) futures[j].get();
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
        int[] elems = this.elements;
        int i = size;
        while (--i >= 0 && cardinality < maxCardinality) {
            if (elems[index] != 0)
                cardinality++;
            index += stride;
        }
        return cardinality;
    }

    protected boolean haveSharedCellsRaw(IntMatrix1D other) {
        if (other instanceof SelectedDenseIntMatrix1D) {
            SelectedDenseIntMatrix1D otherMatrix = (SelectedDenseIntMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseIntMatrix1D) {
            DenseIntMatrix1D otherMatrix = (DenseIntMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        return zero + rank * stride;
    }

    protected IntMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseIntMatrix1D(this.elements, offsets);
    }
}
