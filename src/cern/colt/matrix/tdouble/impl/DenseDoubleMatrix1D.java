/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import edu.emory.mathcs.jtransforms.dct.DoubleDCT_1D;
import edu.emory.mathcs.jtransforms.dht.DoubleDHT_1D;
import edu.emory.mathcs.jtransforms.dst.DoubleDST_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 1-d matrix (aka <i>vector</i>) holding <tt>double</tt> elements. First
 * see the <a href="package-summary.html">package summary</a> and javadoc <a
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
public class DenseDoubleMatrix1D extends DoubleMatrix1D {
    private static final long serialVersionUID = -706456704651139684L;

    private DoubleFFT_1D fft;

    private DoubleDCT_1D dct;

    private DoubleDST_1D dst;

    private DoubleDHT_1D dht;

    /**
     * The elements of this matrix.
     */
    protected double[] elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public DenseDoubleMatrix1D(double[] values) {
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
    public DenseDoubleMatrix1D(int size) {
        setUp(size);
        this.elements = new double[size];
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
    public DenseDoubleMatrix1D(int size, double[] elements, int zero, int stride, boolean isView) {
        setUp(size, zero, stride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    public double aggregate(final cern.colt.function.tdouble.DoubleDoubleFunction aggr, final cern.colt.function.tdouble.DoubleFunction f) {
        if (size == 0)
            return Double.NaN;
        double a = 0;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int firstIdx = size - j * k;
                final int lastIdx = (j == (np - 1)) ? 0 : firstIdx - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {
                    public Double call() throws Exception {
                        int idx = zero + (firstIdx - 1) * stride;
                        double a = f.apply(elements[idx]);
                        for (int i = firstIdx - 1; --i >= lastIdx;) {
                            a = aggr.apply(a, f.apply(elements[idx -= stride]));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            int idx = zero + (size - 1) * stride;
            a = f.apply(elements[idx]);
            for (int i = size - 1; --i >= 0;) {
                a = aggr.apply(a, f.apply(elements[idx -= stride]));
            }
        }
        return a;
    }
    
    public double aggregate(final cern.colt.function.tdouble.DoubleDoubleFunction aggr, final cern.colt.function.tdouble.DoubleFunction f, final IntArrayList indexList) {
        if (size() == 0)
            return Double.NaN;
        final int size = indexList.size();
        final int[] indexElements = indexList.elements();
        double a = 0;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {

                    public Double call() throws Exception {
                        int idx = zero + indexElements[startidx] * stride;
                        double a = f.apply(elements[idx]);
                        double elem;
                        for (int i = startidx + 1; i < stopidx; i++) {
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
            double elem;
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

    public double aggregate(final DoubleMatrix1D other, final cern.colt.function.tdouble.DoubleDoubleFunction aggr, final cern.colt.function.tdouble.DoubleDoubleFunction f) {
        if (!(other instanceof DenseDoubleMatrix1D)) {
            return super.aggregate(other, aggr, f);
        }
        checkSize(other);
        if (size == 0)
            return Double.NaN;
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        final double[] elemsOther = (double[]) other.elements();
        double a = 0;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            Double[] results = new Double[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {
                    public Double call() throws Exception {
                        int idx = zero + startidx * stride;
                        int idxOther = zeroOther + startidx * strideOther;
                        double a = f.apply(elements[idx], elemsOther[idxOther]);
                        for (int i = startidx + 1; i < stopidx; i++) {
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

    public DoubleMatrix1D assign(final cern.colt.function.tdouble.DoubleFunction function) {
        final double multiplicator;
        if (function instanceof cern.jet.math.tdouble.DoubleMult) {
            // x[i] = mult*x[i]
            multiplicator = ((cern.jet.math.tdouble.DoubleMult) function).multiplicator;
            if (multiplicator == 1) {
                return this;
            }
        } else {
            multiplicator = 0;
        }
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + startidx * stride;
                        // specialization for speed
                        if (function instanceof cern.jet.math.tdouble.DoubleMult) {
                            // x[i] = mult*x[i]
                            for (int k = startidx; k < stopidx; k++) {
                                elements[idx] *= multiplicator;
                                idx += stride;
                            }
                        } else {
                            // the general case x[i] = f(x[i])
                            for (int k = startidx; k < stopidx; k++) {
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
            if (function instanceof cern.jet.math.tdouble.DoubleMult) {
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

    public DoubleMatrix1D assign(final cern.colt.function.tdouble.DoubleProcedure cond, final cern.colt.function.tdouble.DoubleFunction function) {
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + startidx * stride;
                        for (int i = startidx; i < stopidx; i++) {
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

    public DoubleMatrix1D assign(final cern.colt.function.tdouble.DoubleProcedure cond, final double value) {
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + startidx * stride;
                        for (int i = startidx; i < stopidx; i++) {
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

    public DoubleMatrix1D assign(final double value) {
        final double[] elems = this.elements;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + startidx * stride;
                        for (int k = startidx; k < stopidx; k++) {
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

    public DoubleMatrix1D assign(final double[] values) {
        if (values.length != size)
            throw new IllegalArgumentException("Must have same number of cells: length=" + values.length + "size()=" + size());
        int np = ConcurrencyUtils.getNumberOfThreads();
        if (isNoView) {
            if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
                Future<?>[] futures = new Future[np];
                int k = size / np;
                for (int j = 0; j < np; j++) {
                    final int startidx = j * k;
                    final int length;
                    if (j == np - 1) {
                        length = size - startidx;
                    } else {
                        length = k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            System.arraycopy(values, startidx, elements, startidx, length);
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                System.arraycopy(values, 0, this.elements, 0, values.length);
            }
        } else {
            if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
                Future<?>[] futures = new Future[np];
                int k = size / np;
                for (int j = 0; j < np; j++) {
                    final int startidx = j * k;
                    final int stopidx;
                    if (j == np - 1) {
                        stopidx = size;
                    } else {
                        stopidx = startidx + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            int idx = zero + startidx * stride;
                            for (int i = startidx; i < stopidx; i++) {
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

    public DoubleMatrix1D assign(DoubleMatrix1D source) {
        // overriden for performance only
        if (!(source instanceof DenseDoubleMatrix1D)) {
            super.assign(source);
            return this;
        }
        DenseDoubleMatrix1D other = (DenseDoubleMatrix1D) source;
        if (other == this)
            return this;
        checkSize(other);
        if (isNoView && other.isNoView) {
            // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            DoubleMatrix1D c = other.copy();
            if (!(c instanceof DenseDoubleMatrix1D)) {
                // should not happen
                super.assign(source);
                return this;
            }
            other = (DenseDoubleMatrix1D) c;
        }

        final double[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + startidx * stride;
                        int idxOther = zeroOther + startidx * strideOther;
                        for (int k = startidx; k < stopidx; k++) {
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

    public DoubleMatrix1D assign(final DoubleMatrix1D y, final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseDoubleMatrix1D)) {
            super.assign(y, function);
            return this;
        }
        checkSize(y);
        final int zeroOther = (int) y.index(0);
        final int strideOther = y.stride();
        final double[] elemsOther = (double[]) y.elements();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + startidx * stride;
                        int idxOther = zeroOther + startidx * strideOther;
                        // specialized for speed
                        if (function == cern.jet.math.tdouble.DoubleFunctions.mult) {
                            // x[i] = x[i] * y[i]
                            for (int k = startidx; k < stopidx; k++) {
                                elements[idx] *= elemsOther[idxOther];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tdouble.DoubleFunctions.div) {
                            // x[i] = x[i] / y[i]
                            for (int k = startidx; k < stopidx; k++) {
                                elements[idx] /= elemsOther[idxOther];
                                idx += stride;
                                idxOther += strideOther;

                            }
                        } else if (function instanceof cern.jet.math.tdouble.DoublePlusMultFirst) {
                            double multiplicator = ((cern.jet.math.tdouble.DoublePlusMultFirst) function).multiplicator;
                            if (multiplicator == 0) {
                                // x[i] = 0*x[i] + y[i]
                                for (int k = startidx; k < stopidx; k++) {
                                    elements[idx] = elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                for (int k = startidx; k < stopidx; k++) {
                                    elements[idx] += elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = -x[i] + y[i]
                                for (int k = startidx; k < stopidx; k++) {
                                    elements[idx] = elemsOther[idxOther] - elements[idx];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else {
                                // the general case x[i] = mult*x[i] + y[i]
                                for (int k = startidx; k < stopidx; k++) {
                                    elements[idx] = multiplicator * elements[idx] + elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            }
                        } else if (function instanceof cern.jet.math.tdouble.DoublePlusMultSecond) {
                            double multiplicator = ((cern.jet.math.tdouble.DoublePlusMultSecond) function).multiplicator;
                            if (multiplicator == 0) {
                                // x[i] = x[i] + 0*y[i]
                                return;
                            } else if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                for (int k = startidx; k < stopidx; k++) {
                                    elements[idx] += elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = x[i] - y[i]
                                for (int k = startidx; k < stopidx; k++) {
                                    elements[idx] -= elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }
                            } else {
                                // the general case x[i] = x[i] + mult*y[i]
                                for (int k = startidx; k < stopidx; k++) {
                                    elements[idx] += multiplicator * elemsOther[idxOther];
                                    idx += stride;
                                    idxOther += strideOther;
                                }

                            }
                        } else {
                            // the general case x[i] = f(x[i],y[i])
                            for (int k = startidx; k < stopidx; k++) {
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
            if (function == cern.jet.math.tdouble.DoubleFunctions.mult) {
                // x[i] = x[i] * y[i]
                for (int k = 0; k < size; k++) {
                    elements[idx] *= elemsOther[idxOther];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tdouble.DoubleFunctions.div) {
                // x[i] = x[i] / y[i]
                for (int k = 0; k < size; k++) {
                    elements[idx] /= elemsOther[idxOther];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function instanceof cern.jet.math.tdouble.DoublePlusMultSecond) {
                double multiplicator = ((cern.jet.math.tdouble.DoublePlusMultSecond) function).multiplicator;
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
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            Integer[] results = new Integer[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopsize;
                if (j == np - 1) {
                    stopsize = size;
                } else {
                    stopsize = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        int idx = zero + startidx * stride;
                        for (int i = startidx; i < stopsize; i++) {
                            if (elements[idx] != 0)
                                cardinality++;
                            idx += stride;
                        }
                        return cardinality;
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (Integer) futures[j].get();
                }
                cardinality = results[0];
                for (int j = 1; j < np; j++) {
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

    /**
     * Computes the discrete cosine transform (DCT-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     * 
     */
    public void dct(boolean scale) {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (dct == null) {
            dct = new DoubleDCT_1D(size);
        }
        if (isNoView) {
            dct.forward(elements, scale);
        } else {
            DoubleMatrix1D copy = this.copy();
            dct.forward((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    /**
     * Computes the discrete Hartley transform (DHT) of this matrix.
     * 
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     * 
     */
    public void dht() {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (dht == null) {
            dht = new DoubleDHT_1D(size);
        }
        if (isNoView) {
            dht.forward(elements);
        } else {
            DoubleMatrix1D copy = this.copy();
            dht.forward((double[]) copy.elements());
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    /**
     * Computes the discrete sine transform (DST-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     */
    public void dst(boolean scale) {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (dst == null) {
            dst = new DoubleDST_1D(size);
        }
        if (isNoView) {
            dst.forward(elements, scale);
        } else {
            DoubleMatrix1D copy = this.copy();
            dst.forward((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    public double[] elements() {
        return elements;
    }

    /**
     * Computes the discrete Fourier transform (DFT) of this matrix. The
     * physical layout of the output data is as follows:
     * 
     * <pre>
     * this[2*k] = Re[k], 0&lt;=k&lt;size/2
     * this[2*k+1] = Im[k], 0&lt;k&lt;size/2
     * this[1] = Re[size/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>getFft</code>. To get back the original
     * data, use <code>ifft</code>.
     * 
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     */
    public void fft() {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (fft == null) {
            fft = new DoubleFFT_1D(size);
        }
        if (isNoView) {
            fft.realForward(elements);
        } else {
            DoubleMatrix1D copy = this.copy();
            fft.realForward((double[]) copy.elements());
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    /**
     * Returns new complex matrix which is the discrete Fourier transform (DFT)
     * of this matrix.
     * 
     * @return the discrete Fourier transform (DFT) of this matrix.
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     */
    public DenseDComplexMatrix1D getFft() {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        final double[] elems;
        if (isNoView == true) {
            elems = elements;
        } else {
            elems = (double[]) this.copy().elements();
        }
        DenseDComplexMatrix1D c = new DenseDComplexMatrix1D(size);
        final double[] cElems = (double[]) ((DenseDComplexMatrix1D) c).elements();
        System.arraycopy(elems, 0, cElems, 0, size);
        if (fft == null) {
            fft = new DoubleFFT_1D(size);
        }
        fft.realForwardFull(cElems);
        ConcurrencyUtils.setNumberOfThreads(oldNp);
        return c;
    }

    /**
     * Returns new complex matrix which is the inverse of the discrete Fourier
     * (IDFT) transform of this matrix.
     * 
     * @return the inverse of the discrete Fourier transform (IDFT) of this
     *         matrix.
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     */
    public DenseDComplexMatrix1D getIfft(boolean scale) {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        final double[] elems;
        if (isNoView == true) {
            elems = elements;
        } else {
            elems = (double[]) this.copy().elements();
        }
        DenseDComplexMatrix1D c = new DenseDComplexMatrix1D(size);
        final double[] cElems = (double[]) ((DenseDComplexMatrix1D) c).elements();
        System.arraycopy(elems, 0, cElems, 0, size);
        if (fft == null) {
            fft = new DoubleFFT_1D(size);
        }
        fft.realInverseFull(cElems, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNp);
        return c;
    }

    public void getNonZeros(final IntArrayList indexList, final DoubleArrayList valueList) {
        boolean fillIndexList = indexList != null;
        boolean fillValueList = valueList != null;
        if (fillIndexList)
            indexList.clear();
        if (fillValueList)
            valueList.clear();
        int rem = size % 2;
        int idx = zero;
        if (rem == 1) {
            double value = elements[idx];
            if (value != 0) {
                if(fillIndexList) {
                    indexList.add(0);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            double value = elements[idx];
            if (value != 0) {
                if(fillIndexList) {
                    indexList.add(i);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;
            value = elements[idx];
            if (value != 0) {
                if(fillIndexList) {
                    indexList.add(i + 1);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;
        }
    }

    public void getPositiveValues(final IntArrayList indexList, final DoubleArrayList valueList) {
        boolean fillIndexList = indexList != null;
        boolean fillValueList = valueList != null;
        if (fillIndexList)
            indexList.clear();
        if (fillValueList)
            valueList.clear();
        int rem = size % 2;
        int idx = zero;
        if (rem == 1) {
            double value = elements[idx];
            if (value > 0) {
                if(fillIndexList) {
                    indexList.add(0);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            double value = elements[idx];
            if (value > 0) {
                if(fillIndexList) {
                    indexList.add(i);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;
            value = elements[idx];
            if (value > 0) {
                if(fillIndexList) {
                    indexList.add(i + 1);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;
        }
    }

    public void getNegativeValues(final IntArrayList indexList, final DoubleArrayList valueList) {
        boolean fillIndexList = indexList != null;
        boolean fillValueList = valueList != null;
        if (fillIndexList)
            indexList.clear();
        if (fillValueList)
            valueList.clear();
        int rem = size % 2;
        int idx = zero;
        if (rem == 1) {
            double value = elements[idx];
            if (value < 0) {
                if(fillIndexList) {
                    indexList.add(0);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;

        }
        for (int i = rem; i < size; i += 2) {
            double value = elements[idx];
            if (value < 0) {
                if(fillIndexList) {
                    indexList.add(i);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;
            value = elements[idx];
            if (value < 0) {
                if(fillIndexList) {
                    indexList.add(i + 1);
                }
                if(fillValueList) {
                    valueList.add(value);
                }
            }
            idx += stride;
        }
    }

    public double[] getMaxLocation() {
        int location = 0;
        double maxValue = 0;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            double[][] results = new double[np][2];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int idx = zero + startidx * stride;
                        double maxValue = elements[idx];
                        int location = (idx - zero) / stride;
                        for (int i = startidx + 1; i < stopidx; i++) {
                            idx += stride;
                            if (maxValue < elements[idx]) {
                                maxValue = elements[idx];
                                location = (idx - zero) / stride;
                            }
                        }
                        return new double[] { maxValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (double[]) futures[j].get();
                }
                maxValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < np; j++) {
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
        return new double[] { maxValue, location };
    }

    public double[] getMinLocation() {
        int location = 0;
        double minValue = 0;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            double[][] results = new double[np][2];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int idx = zero + startidx * stride;
                        double minValue = elements[idx];
                        int location = (idx - zero) / stride;
                        for (int i = startidx + 1; i < stopidx; i++) {
                            idx += stride;
                            if (minValue > elements[idx]) {
                                minValue = elements[idx];
                                location = (idx - zero) / stride;
                            }
                        }
                        return new double[] { minValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (double[]) futures[j].get();
                }
                minValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < np; j++) {
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
        return new double[] { minValue, location };
    }

    public double getQuick(int index) {
        return elements[zero + index * stride];
    }

    /**
     * Computes the inverse of the discrete cosine transform (DCT-III) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     * 
     */
    public void idct(boolean scale) {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (dct == null) {
            dct = new DoubleDCT_1D(size);
        }
        if (isNoView) {
            dct.inverse(elements, scale);
        } else {
            DoubleMatrix1D copy = this.copy();
            dct.inverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    /**
     * Computes the inverse of the discrete Hartley transform (IDHT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     * 
     */
    public void idht(boolean scale) {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (dht == null) {
            dht = new DoubleDHT_1D(size);
        }
        if (isNoView) {
            dht.inverse(elements, scale);
        } else {
            DoubleMatrix1D copy = this.copy();
            dht.inverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    /**
     * Computes the inverse of discrete sine transform (DST-III) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     */
    public void idst(boolean scale) {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (dst == null) {
            dst = new DoubleDST_1D(size);
        }
        if (isNoView) {
            dst.inverse(elements, scale);
        } else {
            DoubleMatrix1D copy = this.copy();
            dst.inverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    /**
     * Computes the inverse of the discrete Fourier transform (DFT) of this
     * matrix. The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * this[2*k] = Re[k], 0&lt;=k&lt;size/2
     * this[2*k+1] = Im[k], 0&lt;k&lt;size/2
     * this[1] = Re[size/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>getIfft</code>.
     * 
     * @throws IllegalArgumentException
     *             if the size of this matrix is not a power of 2 number.
     */
    public void ifft(boolean scale) {
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (fft == null) {
            fft = new DoubleFFT_1D(size);
        }
        if (isNoView) {
            fft.realInverse(elements, scale);
        } else {
            DoubleMatrix1D copy = this.copy();
            fft.realInverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    public DoubleMatrix1D like(int size) {
        return new DenseDoubleMatrix1D(size);
    }

    public DoubleMatrix2D like2D(int rows, int columns) {
        return new DenseDoubleMatrix2D(rows, columns);
    }

    public DoubleMatrix2D reshape(final int rows, final int cols) {
        if (rows * cols != size) {
            throw new IllegalArgumentException("rows*cols != size");
        }
        DoubleMatrix2D M = new DenseDoubleMatrix2D(rows, cols);
        final double[] elemsOther = (double[]) M.elements();
        final int zeroOther = (int) M.index(0, 0);
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = cols / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = cols;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        int idxOther;
                        for (int c = startcol; c < stopcol; c++) {
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
            for (int c = 0; c < cols; c++) {
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

    public DoubleMatrix3D reshape(final int slices, final int rows, final int cols) {
        if (slices * rows * cols != size) {
            throw new IllegalArgumentException("slices*rows*cols != size");
        }
        DoubleMatrix3D M = new DenseDoubleMatrix3D(slices, rows, cols);
        final double[] elemsOther = (double[]) M.elements();
        final int zeroOther = (int) M.index(0, 0, 0);
        final int sliceStrideOther = M.sliceStride();
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = slices / np;
            for (int j = 0; j < np; j++) {
                final int startslice = j * k;
                final int stopslice;
                if (j == np - 1) {
                    stopslice = slices;
                } else {
                    stopslice = startslice + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        int idxOther;
                        for (int s = startslice; s < stopslice; s++) {
                            for (int c = 0; c < cols; c++) {
                                idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
                                idx = zero + (s * rows * cols + c * rows) * stride;
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
                for (int c = 0; c < cols; c++) {
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

    public void setQuick(int index, double value) {
        elements[zero + index * stride] = value;
    }

    public void swap(final DoubleMatrix1D other) {
        // overriden for performance only
        if (!(other instanceof DenseDoubleMatrix1D)) {
            super.swap(other);
        }
        DenseDoubleMatrix1D y = (DenseDoubleMatrix1D) other;
        if (y == this)
            return;
        checkSize(y);
        final double[] elemsOther = y.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + startidx * stride;
                        int idxOther = zeroOther + startidx * strideOther;
                        for (int k = startidx; k < stopidx; k++) {
                            double tmp = elements[idx];
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
                double tmp = elements[idx];
                elements[idx] = elemsOther[idxOther];
                elemsOther[idxOther] = tmp;
                idx += stride;
                idxOther += strideOther;
            }
        }
    }

    public void toArray(double[] values) {
        if (values.length < size)
            throw new IllegalArgumentException("values too small");
        if (this.isNoView)
            System.arraycopy(this.elements, 0, values, 0, this.elements.length);
        else
            super.toArray(values);
    }

    public double zDotProduct(DoubleMatrix1D y, int from, int length) {
        if (!(y instanceof DenseDoubleMatrix1D)) {
            return super.zDotProduct(y, from, length);
        }
        DenseDoubleMatrix1D yy = (DenseDoubleMatrix1D) y;

        int tail = from + length;
        if (from < 0 || length < 0)
            return 0;
        if (size < tail)
            tail = size;
        if (y.size() < tail)
            tail = y.size();
        final double[] elemsOther = yy.elements;
        int zeroThis = (int) index(from);
        int zeroOther = (int) yy.index(from);
        int strideOther = yy.stride;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        double sum = 0;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (length >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            final int zeroThisF = zeroThis;
            final int zeroOtherF = zeroOther;
            final int strideOtherF = strideOther;
            Future<?>[] futures = new Future[np];
            Double[] results = new Double[np];
            int k = length / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = length;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {
                    public Double call() throws Exception {
                        int idx = zeroThisF + startidx * stride;
                        int idxOther = zeroOtherF + startidx * strideOtherF;
                        idx -= stride;
                        idxOther -= strideOtherF;
                        double sum = 0;
                        int min = stopidx - startidx;
                        for (int k = min / 4; --k >= 0;) {
                            sum += elements[idx += stride] * elemsOther[idxOther += strideOtherF] + elements[idx += stride] * elemsOther[idxOther += strideOtherF] + elements[idx += stride] * elemsOther[idxOther += strideOtherF] + elements[idx += stride] * elemsOther[idxOther += strideOtherF];
                        }
                        for (int k = min % 4; --k >= 0;) {
                            sum += elements[idx += stride] * elemsOther[idxOther += strideOtherF];
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (Double) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < np; j++) {
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
                sum += elements[zeroThis += stride] * elemsOther[zeroOther += strideOther] + elements[zeroThis += stride] * elemsOther[zeroOther += strideOther] + elements[zeroThis += stride] * elemsOther[zeroOther += strideOther] + elements[zeroThis += stride]
                        * elemsOther[zeroOther += strideOther];
            }
            for (int k = min % 4; --k >= 0;) {
                sum += elements[zeroThis += stride] * elemsOther[zeroOther += strideOther];
            }
        }
        return sum;
    }

    public double zSum() {
        double sum = 0;
        final double[] elems = this.elements;
        if (elems == null)
            throw new InternalError();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            Double[] results = new Double[np];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {
                    public Double call() throws Exception {
                        double sum = 0;
                        int idx = zero + startidx * stride;
                        for (int i = startidx; i < stopidx; i++) {
                            sum += elems[idx];
                            idx += stride;
                        }
                        return Double.valueOf(sum);
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (Double) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < np; j++) {
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
        double[] elems = this.elements;
        int i = size;
        while (--i >= 0 && cardinality < maxCardinality) {
            if (elems[index] != 0)
                cardinality++;
            index += stride;
        }
        return cardinality;
    }

    protected boolean haveSharedCellsRaw(DoubleMatrix1D other) {
        if (other instanceof SelectedDenseDoubleMatrix1D) {
            SelectedDenseDoubleMatrix1D otherMatrix = (SelectedDenseDoubleMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseDoubleMatrix1D) {
            DenseDoubleMatrix1D otherMatrix = (DenseDoubleMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        return zero + rank * stride;
    }

    protected DoubleMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseDoubleMatrix1D(this.elements, offsets);
    }
}
