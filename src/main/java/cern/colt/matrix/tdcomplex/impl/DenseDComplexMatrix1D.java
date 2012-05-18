/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 1-d matrix (aka <i>vector</i>) holding <tt>complex</tt> elements.
 * <p>
 * Internally holds one single contiguous one-dimensional array. Complex data is
 * represented by 2 double values in sequence, i.e. elements[zero + 2 * k *
 * stride] constitute real part and elements[zero + 2 * k * stride + 1]
 * constitute imaginary part (k=0,...,size()-1).
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseDComplexMatrix1D extends DComplexMatrix1D {

    private static final long serialVersionUID = 1L;

    private DoubleFFT_1D fft;

    /**
     * The elements of this matrix. Complex data is represented by 2 double
     * values in sequence, i.e. elements[zero + 2 * k * stride] constitute real
     * part and elements[zero + 2 * k * stride] constitute imaginary part
     * (k=0,...,size()-1).
     */
    protected double[] elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa. Due to the fact that complex data is represented
     * by 2 double values in sequence: the real and imaginary parts, the size of
     * new matrix will be equal to values.length / 2.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public DenseDComplexMatrix1D(double[] values) {
        this(values.length / 2);
        assign(values);
    }

    /**
     * Constructs a complex matrix with the same size as <tt>realPart</tt>
     * matrix and fills the real part of this matrix with elements of
     * <tt>realPart</tt>.
     * 
     * @param realPart
     *            a real matrix whose elements become a real part of this matrix
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public DenseDComplexMatrix1D(DoubleMatrix1D realPart) {
        this((int) realPart.size());
        assignReal(realPart);
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
    public DenseDComplexMatrix1D(int size) {
        setUp(size, 0, 2);
        this.isNoView = true;
        this.elements = new double[2 * size];
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
     * @param isNoView
     *            if false then the view is constructed
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public DenseDComplexMatrix1D(int size, double[] elements, int zero, int stride, boolean isNoView) {
        setUp(size, zero, stride);
        this.elements = elements;
        this.isNoView = isNoView;
    }

    public double[] aggregate(final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction aggr,
            final cern.colt.function.tdcomplex.DComplexDComplexFunction f) {
        double[] b = new double[2];
        if (size == 0) {
            b[0] = Double.NaN;
            b[1] = Double.NaN;
            return b;
        }
        double[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {

                    public double[] call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        double[] a = f.apply(new double[] { elements[idx], elements[idx + 1] });
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            a = aggr.apply(a, f.apply(new double[] { elements[idx], elements[idx + 1] }));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(new double[] { elements[zero], elements[zero + 1] });
            int idx = zero;
            for (int i = 1; i < size; i++) {
                idx += stride;
                a = aggr.apply(a, f.apply(new double[] { elements[idx], elements[idx + 1] }));
            }
        }
        return a;
    }

    public double[] aggregate(final DComplexMatrix1D other,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction aggr,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction f) {
        if (!(other instanceof DenseDComplexMatrix1D)) {
            return super.aggregate(other, aggr, f);
        }
        checkSize(other);
        if (size == 0) {
            double[] b = new double[2];
            b[0] = Double.NaN;
            b[1] = Double.NaN;
            return b;
        }
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        final double[] elemsOther = (double[]) other.elements();
        double[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        double[] a = f.apply(new double[] { elements[idx], elements[idx + 1] }, new double[] {
                                elemsOther[idxOther], elemsOther[idxOther + 1] });
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            idxOther += strideOther;
                            a = aggr.apply(a, f.apply(new double[] { elements[idx], elements[idx + 1] }, new double[] {
                                    elemsOther[idxOther], elemsOther[idxOther + 1] }));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            a = f.apply(new double[] { elements[zero], elements[zero + 1] }, new double[] { elemsOther[zeroOther],
                    elemsOther[zeroOther + 1] });
            for (int i = 1; i < size; i++) {
                idx += stride;
                idxOther += strideOther;
                a = aggr.apply(a, f.apply(new double[] { elements[idx], elements[idx + 1] }, new double[] {
                        elemsOther[idxOther], elemsOther[idxOther + 1] }));
            }
        }
        return a;
    }

    public DComplexMatrix1D assign(final cern.colt.function.tdcomplex.DComplexDComplexFunction function) {
        if (this.elements == null)
            throw new InternalError();
        if (function instanceof cern.jet.math.tdcomplex.DComplexMult) {
            double[] multiplicator = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
            if (multiplicator[0] == 1 && multiplicator[1] == 0)
                return this;
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
                        double[] tmp = new double[2];
                        int idx = zero + firstIdx * stride;
                        if (function instanceof cern.jet.math.tdcomplex.DComplexMult) {
                            double[] multiplicator = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] = elements[idx] * multiplicator[0] - elements[idx + 1] * multiplicator[1];
                                elements[idx + 1] = elements[idx + 1] * multiplicator[0] + elements[idx]
                                        * multiplicator[1];
                                idx += stride;
                            }
                        } else {
                            for (int k = firstIdx; k < lastIdx; k++) {
                                tmp[0] = elements[idx];
                                tmp[1] = elements[idx + 1];
                                tmp = function.apply(tmp);
                                elements[idx] = tmp[0];
                                elements[idx + 1] = tmp[1];
                                idx += stride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] tmp = new double[2];
            int idx = zero;
            if (function instanceof cern.jet.math.tdcomplex.DComplexMult) {
                double[] multiplicator = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
                for (int k = 0; k < size; k++) {
                    elements[idx] = elements[idx] * multiplicator[0] - elements[idx + 1] * multiplicator[1];
                    elements[idx + 1] = elements[idx + 1] * multiplicator[0] + elements[idx] * multiplicator[1];
                    idx += stride;
                }
            } else {
                for (int k = 0; k < size; k++) {
                    tmp[0] = elements[idx];
                    tmp[1] = elements[idx + 1];
                    tmp = function.apply(tmp);
                    elements[idx] = tmp[0];
                    elements[idx + 1] = tmp[1];
                    idx += stride;
                }
            }
        }
        return this;
    }

    public DComplexMatrix1D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond,
            final cern.colt.function.tdcomplex.DComplexDComplexFunction function) {
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
                        double[] elem = new double[2];
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elem[0] = elements[idx];
                            elem[1] = elements[idx + 1];
                            if (cond.apply(elem) == true) {
                                elem = function.apply(elem);
                                elements[idx] = elem[0];
                                elements[idx + 1] = elem[1];
                            }
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] elem = new double[2];
            int idx = zero;
            for (int i = 0; i < size; i++) {
                elem[0] = elements[idx];
                elem[1] = elements[idx + 1];
                if (cond.apply(elem) == true) {
                    elem = function.apply(elem);
                    elements[idx] = elem[0];
                    elements[idx + 1] = elem[1];
                }
                idx += stride;
            }
        }
        return this;
    }

    public DComplexMatrix1D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond, final double[] value) {
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
                        double[] elem = new double[2];
                        int idx = zero + firstIdx * stride;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elem[0] = elements[idx];
                            elem[1] = elements[idx + 1];
                            if (cond.apply(elem) == true) {
                                elements[idx] = value[0];
                                elements[idx + 1] = value[1];
                            }
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] elem = new double[2];
            int idx = zero;
            for (int i = 0; i < size; i++) {
                elem[0] = elements[idx];
                elem[1] = elements[idx + 1];
                if (cond.apply(elem) == true) {
                    elements[idx] = value[0];
                    elements[idx + 1] = value[1];
                }
                idx += stride;
            }
        }
        return this;
    }

    public DComplexMatrix1D assign(final cern.colt.function.tdcomplex.DComplexRealFunction function) {
        if (this.elements == null)
            throw new InternalError();
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
                        if (function == DComplexFunctions.abs) {
                            for (int k = firstIdx; k < lastIdx; k++) {
                                double absX = Math.abs(elements[idx]);
                                double absY = Math.abs(elements[idx + 1]);
                                if (absX == 0.0 && absY == 0.0) {
                                    elements[idx] = 0;
                                } else if (absX >= absY) {
                                    double d = elements[idx + 1] / elements[idx];
                                    elements[idx] = absX * Math.sqrt(1.0 + d * d);
                                } else {
                                    double d = elements[idx] / elements[idx + 1];
                                    elements[idx] = absY * Math.sqrt(1.0 + d * d);
                                }
                                elements[idx + 1] = 0;
                                idx += stride;
                            }
                        } else {
                            double[] tmp = new double[2];
                            for (int k = firstIdx; k < lastIdx; k++) {
                                tmp[0] = elements[idx];
                                tmp[1] = elements[idx + 1];
                                tmp[0] = function.apply(tmp);
                                elements[idx] = tmp[0];
                                elements[idx + 1] = 0;
                                idx += stride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            if (function == DComplexFunctions.abs) {
                for (int k = 0; k < size; k++) {
                    double absX = Math.abs(elements[idx]);
                    double absY = Math.abs(elements[idx + 1]);
                    if (absX == 0.0 && absY == 0.0) {
                        elements[idx] = 0;
                    } else if (absX >= absY) {
                        double d = elements[idx + 1] / elements[idx];
                        elements[idx] = absX * Math.sqrt(1.0 + d * d);
                    } else {
                        double d = elements[idx] / elements[idx + 1];
                        elements[idx] = absY * Math.sqrt(1.0 + d * d);
                    }
                    elements[idx + 1] = 0;
                    idx += stride;
                }
            } else {
                double[] tmp = new double[2];
                for (int k = 0; k < size; k++) {
                    tmp[0] = elements[idx];
                    tmp[1] = elements[idx + 1];
                    tmp[0] = function.apply(tmp);
                    elements[idx] = tmp[0];
                    elements[idx + 1] = 0;
                    idx += stride;
                }
            }
        }
        return this;
    }

    public DComplexMatrix1D assign(DComplexMatrix1D source) {
        if (!(source instanceof DenseDComplexMatrix1D)) {
            return super.assign(source);
        }
        DenseDComplexMatrix1D other = (DenseDComplexMatrix1D) source;
        if (other == this)
            return this;
        checkSize(other);
        if (isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            DComplexMatrix1D c = other.copy();
            if (!(c instanceof DenseDComplexMatrix1D)) { // should not happen
                return super.assign(source);
            }
            other = (DenseDComplexMatrix1D) c;
        }

        final double[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int strideOther = other.stride;
        final int zeroOther = (int) other.index(0);

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
                            elements[idx + 1] = elemsOther[idxOther + 1];
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
                elements[idx + 1] = elemsOther[idxOther + 1];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return this;
    }

    public DComplexMatrix1D assign(DComplexMatrix1D y,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction function) {
        if (!(y instanceof DenseDComplexMatrix1D)) {
            return super.assign(y, function);
        }
        checkSize(y);
        final double[] elemsOther = (double[]) y.elements();
        final int zeroOther = (int) y.index(0);
        final int strideOther = y.stride();

        if (elements == null || elemsOther == null)
            throw new InternalError();
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
                        if (function == cern.jet.math.tdcomplex.DComplexFunctions.plus) {
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] += elemsOther[idxOther];
                                elements[idx + 1] += elemsOther[idxOther + 1];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.minus) {
                            for (int k = firstIdx; k < lastIdx; k++) {
                                elements[idx] -= elemsOther[idxOther];
                                elements[idx + 1] -= elemsOther[idxOther + 1];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.div) {
                            double[] tmp = new double[2];
                            for (int k = firstIdx; k < lastIdx; k++) {
                                double re = elemsOther[idxOther];
                                double im = elemsOther[idxOther + 1];
                                double scalar;
                                if (Math.abs(re) >= Math.abs(im)) {
                                    scalar = (1.0 / (re + im * (im / re)));
                                    tmp[0] = scalar * (elements[idx] + elements[idx + 1] * (im / re));
                                    tmp[1] = scalar * (elements[idx + 1] - elements[idx] * (im / re));
                                } else {
                                    scalar = (1.0 / (re * (re / im) + im));
                                    tmp[0] = scalar * (elements[idx] * (re / im) + elements[idx + 1]);
                                    tmp[1] = scalar * (elements[idx + 1] * (re / im) - elements[idx]);
                                }
                                elements[idx] = tmp[0];
                                elements[idx + 1] = tmp[1];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.mult) {
                            double[] tmp = new double[2];
                            for (int k = firstIdx; k < lastIdx; k++) {
                                tmp[0] = elements[idx] * elemsOther[idxOther] - elements[idx + 1]
                                        * elemsOther[idxOther + 1];
                                tmp[1] = elements[idx + 1] * elemsOther[idxOther] + elements[idx]
                                        * elemsOther[idxOther + 1];
                                elements[idx] = tmp[0];
                                elements[idx + 1] = tmp[1];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjFirst) {
                            double[] tmp = new double[2];
                            for (int k = firstIdx; k < lastIdx; k++) {
                                tmp[0] = elements[idx] * elemsOther[idxOther] + elements[idx + 1]
                                        * elemsOther[idxOther + 1];
                                tmp[1] = -elements[idx + 1] * elemsOther[idxOther] + elements[idx]
                                        * elemsOther[idxOther + 1];
                                elements[idx] = tmp[0];
                                elements[idx + 1] = tmp[1];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjSecond) {
                            double[] tmp = new double[2];
                            for (int k = firstIdx; k < lastIdx; k++) {
                                tmp[0] = elements[idx] * elemsOther[idxOther] + elements[idx + 1]
                                        * elemsOther[idxOther + 1];
                                tmp[1] = elements[idx + 1] * elemsOther[idxOther] - elements[idx]
                                        * elemsOther[idxOther + 1];
                                elements[idx] = tmp[0];
                                elements[idx + 1] = tmp[1];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        } else {
                            double[] tmp1 = new double[2];
                            double[] tmp2 = new double[2];
                            for (int k = firstIdx; k < lastIdx; k++) {
                                tmp1[0] = elements[idx];
                                tmp1[1] = elements[idx + 1];
                                tmp2[0] = elemsOther[idxOther];
                                tmp2[1] = elemsOther[idxOther + 1];
                                tmp1 = function.apply(tmp1, tmp2);
                                elements[idx] = tmp1[0];
                                elements[idx + 1] = tmp1[1];
                                idx += stride;
                                idxOther += strideOther;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            if (function == cern.jet.math.tdcomplex.DComplexFunctions.plus) {
                for (int k = 0; k < size; k++) {
                    elements[idx] += elemsOther[idxOther];
                    elements[idx + 1] += elemsOther[idxOther + 1];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.minus) {
                for (int k = 0; k < size; k++) {
                    elements[idx] -= elemsOther[idxOther];
                    elements[idx + 1] -= elemsOther[idxOther + 1];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.div) {
                double[] tmp = new double[2];
                for (int k = 0; k < size; k++) {
                    double re = elemsOther[idxOther];
                    double im = elemsOther[idxOther + 1];
                    double scalar;
                    if (Math.abs(re) >= Math.abs(im)) {
                        scalar = (1.0 / (re + im * (im / re)));
                        tmp[0] = scalar * (elements[idx] + elements[idx + 1] * (im / re));
                        tmp[1] = scalar * (elements[idx + 1] - elements[idx] * (im / re));
                    } else {
                        scalar = (1.0 / (re * (re / im) + im));
                        tmp[0] = scalar * (elements[idx] * (re / im) + elements[idx + 1]);
                        tmp[1] = scalar * (elements[idx + 1] * (re / im) - elements[idx]);
                    }
                    elements[idx] = tmp[0];
                    elements[idx + 1] = tmp[1];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.mult) {
                double[] tmp = new double[2];
                for (int k = 0; k < size; k++) {
                    tmp[0] = elements[idx] * elemsOther[idxOther] - elements[idx + 1] * elemsOther[idxOther + 1];
                    tmp[1] = elements[idx + 1] * elemsOther[idxOther] + elements[idx] * elemsOther[idxOther + 1];
                    elements[idx] = tmp[0];
                    elements[idx + 1] = tmp[1];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjFirst) {
                double[] tmp = new double[2];
                for (int k = 0; k < size; k++) {
                    tmp[0] = elements[idx] * elemsOther[idxOther] + elements[idx + 1] * elemsOther[idxOther + 1];
                    tmp[1] = -elements[idx + 1] * elemsOther[idxOther] + elements[idx] * elemsOther[idxOther + 1];
                    elements[idx] = tmp[0];
                    elements[idx + 1] = tmp[1];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjSecond) {
                double[] tmp = new double[2];
                for (int k = 0; k < size; k++) {
                    tmp[0] = elements[idx] * elemsOther[idxOther] + elements[idx + 1] * elemsOther[idxOther + 1];
                    tmp[1] = elements[idx + 1] * elemsOther[idxOther] - elements[idx] * elemsOther[idxOther + 1];
                    elements[idx] = tmp[0];
                    elements[idx + 1] = tmp[1];
                    idx += stride;
                    idxOther += strideOther;
                }
            } else {
                double[] tmp1 = new double[2];
                double[] tmp2 = new double[2];
                for (int k = 0; k < size; k++) {
                    tmp1[0] = elements[idx];
                    tmp1[1] = elements[idx + 1];
                    tmp2[0] = elemsOther[idxOther];
                    tmp2[1] = elemsOther[idxOther + 1];
                    tmp1 = function.apply(tmp1, tmp2);
                    elements[idx] = tmp1[0];
                    elements[idx + 1] = tmp1[1];
                    idx += stride;
                    idxOther += strideOther;
                }
            }
        }
        return this;
    }

    public DComplexMatrix1D assign(final double re, final double im) {
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
                            elements[idx] = re;
                            elements[idx + 1] = im;
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int i = 0; i < size; i++) {
                this.elements[idx] = re;
                this.elements[idx + 1] = im;
                idx += stride;
            }
        }
        return this;
    }

    public DComplexMatrix1D assign(double[] values) {
        if (isNoView) {
            if (values.length != 2 * size)
                throw new IllegalArgumentException("The length of values[] must be equal to 2*size()=" + 2 * size());
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            super.assign(values);
        }
        return this;
    }

    public DComplexMatrix1D assignImaginary(final DoubleMatrix1D other) {
        if (!(other instanceof DenseDoubleMatrix1D)) {
            return super.assignImaginary(other);
        }
        checkSize(other);
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        final double[] elemsOther = (double[]) other.elements();
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
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elements[idx + 1] = elemsOther[idxOther];
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
            for (int i = 0; i < size; i++) {
                elements[idx + 1] = elemsOther[idxOther];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return this;
    }

    public DComplexMatrix1D assignReal(final DoubleMatrix1D other) {
        if (!(other instanceof DenseDoubleMatrix1D)) {
            return super.assignReal(other);
        }
        checkSize(other);
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        final double[] elemsOther = (double[]) other.elements();
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
                        for (int i = firstIdx; i < lastIdx; i++) {
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
            for (int i = 0; i < size; i++) {
                elements[idx] = elemsOther[idxOther];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return this;
    }

    /**
     * Computes the discrete Fourier transform (DFT) of this matrix. Throws
     * IllegalArgumentException if the size of this matrix is not a power of 2
     * number.
     */
    public void fft() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft == null) {
            fft = new DoubleFFT_1D(size);
        }
        if (isNoView) {
            fft.complexForward(elements);
        } else {
            DComplexMatrix1D copy = this.copy();
            fft.complexForward((double[]) copy.elements());
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public double[] elements() {
        return elements;
    }

    public DoubleMatrix1D getImaginaryPart() {
        final DenseDoubleMatrix1D Im = new DenseDoubleMatrix1D(size);
        final double[] elemsOther = Im.elements();
        final int zeroOther = (int) Im.index(0);
        final int strideOther = Im.stride();
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
                            elemsOther[idxOther] = elements[idx + 1];
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
            for (int i = 0; i < size; i++) {
                elemsOther[idxOther] = elements[idx + 1];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return Im;
    }

    public void getNonZeros(final IntArrayList indexList, final ArrayList<double[]> valueList) {
        indexList.clear();
        valueList.clear();
        int s = (int) size();

        int idx = zero;
        for (int k = 0; k < s; k++) {
            double[] value = new double[2];
            value[0] = elements[idx];
            value[1] = elements[idx + 1];
            if (value[0] != 0 || value[1] != 0) {
                indexList.add(k);
                valueList.add(value);
            }
            idx += stride;
        }
    }

    public double[] getQuick(int index) {
        int idx = zero + index * stride;
        return new double[] { elements[idx], elements[idx + 1] };
    }

    public DoubleMatrix1D getRealPart() {
        final DenseDoubleMatrix1D R = new DenseDoubleMatrix1D(size);
        final double[] elemsOther = R.elements();
        final int zeroOther = (int) R.index(0);
        final int strideOther = R.stride();
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
                            elemsOther[idxOther] = elements[idx];
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
            for (int i = 0; i < size; i++) {
                elemsOther[idxOther] = elements[idx];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return R;
    }

    /**
     * Computes the inverse of the discrete Fourier transform (IDFT) of this
     * matrix. Throws IllegalArgumentException if the size of this matrix is not
     * a power of 2 number.
     * 
     * @param scale
     *            if true, then scaling is performed.
     */
    public void ifft(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft == null) {
            fft = new DoubleFFT_1D(size);
        }
        if (isNoView) {
            fft.complexInverse(elements, scale);
        } else {
            DComplexMatrix1D copy = this.copy();
            fft.complexInverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public DComplexMatrix1D like(int size) {
        return new DenseDComplexMatrix1D(size);
    }

    public DComplexMatrix2D like2D(int rows, int columns) {
        return new DenseDComplexMatrix2D(rows, columns);
    }

    public DComplexMatrix2D reshape(final int rows, final int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        DComplexMatrix2D M = new DenseDComplexMatrix2D(rows, columns);
        final double[] elemsOther = (double[]) M.elements();
        final int zeroOther = (int) M.index(0, 0);
        final int rowStrideOther = M.rowStride();
        final int columnStrideOther = M.columnStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
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
                            idxOther = zeroOther + c * columnStrideOther;
                            idx = zero + (c * rows) * stride;
                            for (int r = 0; r < rows; r++) {
                                elemsOther[idxOther] = elements[idx];
                                elemsOther[idxOther + 1] = elements[idx + 1];
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
                idxOther = zeroOther + c * columnStrideOther;
                for (int r = 0; r < rows; r++) {
                    elemsOther[idxOther] = elements[idx];
                    elemsOther[idxOther + 1] = elements[idx + 1];
                    idxOther += rowStrideOther;
                    idx += stride;
                }
            }
        }
        return M;
    }

    public DComplexMatrix3D reshape(final int slices, final int rows, final int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        DComplexMatrix3D M = new DenseDComplexMatrix3D(slices, rows, columns);
        final double[] elemsOther = (double[]) M.elements();
        final int zeroOther = (int) M.index(0, 0, 0);
        final int sliceStrideOther = M.sliceStride();
        final int rowStrideOther = M.rowStride();
        final int columnStrideOther = M.columnStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
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
                                idxOther = zeroOther + s * sliceStrideOther + c * columnStrideOther;
                                idx = zero + (s * rows * columns + c * rows) * stride;
                                for (int r = 0; r < rows; r++) {
                                    elemsOther[idxOther] = elements[idx];
                                    elemsOther[idxOther + 1] = elements[idx + 1];
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
                    idxOther = zeroOther + s * sliceStrideOther + c * columnStrideOther;
                    for (int r = 0; r < rows; r++) {
                        elemsOther[idxOther] = elements[idx];
                        elemsOther[idxOther + 1] = elements[idx + 1];
                        idxOther += rowStrideOther;
                        idx += stride;
                    }
                }
            }
        }
        return M;
    }

    public void setQuick(int index, double re, double im) {
        int idx = zero + index * stride;
        this.elements[idx] = re;
        this.elements[idx + 1] = im;
    }

    public void setQuick(int index, double[] value) {
        int idx = zero + index * stride;
        this.elements[idx] = value[0];
        this.elements[idx + 1] = value[1];
    }

    public void swap(DComplexMatrix1D other) {
        if (!(other instanceof DenseDComplexMatrix1D)) {
            super.swap(other);
        }
        DenseDComplexMatrix1D y = (DenseDComplexMatrix1D) other;
        if (y == this)
            return;
        checkSize(y);

        final double[] elemsOther = y.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int strideOther = y.stride;
        final int zeroOther = (int) y.index(0);

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
                        double tmp;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            tmp = elements[idx];
                            elements[idx] = elemsOther[idxOther];
                            elemsOther[idxOther] = tmp;
                            tmp = elements[idx + 1];
                            elements[idx + 1] = elemsOther[idxOther + 1];
                            elemsOther[idxOther + 1] = tmp;
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
            double tmp;
            for (int k = 0; k < size; k++) {
                tmp = elements[idx];
                elements[idx] = elemsOther[idxOther];
                elemsOther[idxOther] = tmp;
                tmp = elements[idx + 1];
                elements[idx + 1] = elemsOther[idxOther + 1];
                elemsOther[idxOther + 1] = tmp;
                idx += stride;
                idxOther += strideOther;
            }
        }
    }

    public void toArray(double[] values) {
        if (values.length < 2 * size)
            throw new IllegalArgumentException("values too small");
        if (isNoView)
            System.arraycopy(this.elements, 0, values, 0, this.elements.length);
        else
            super.toArray(values);
    }

    public double[] zDotProduct(final DComplexMatrix1D y, final int from, int length) {
        int size = (int) size();
        if (from < 0 || length <= 0)
            return new double[] { 0, 0 };

        int tail = from + length;
        if (size < tail)
            tail = size;
        if (y.size() < tail)
            tail = (int) y.size();
        length = tail - from;
        final double[] elemsOther = (double[]) y.elements();
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int strideOther = y.stride();
        final int zero = (int) index(from);
        final int zeroOther = (int) y.index(from);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        double[] sum = new double[2];

        if ((nthreads > 1) && (length >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, length);
            Future<?>[] futures = new Future[nthreads];
            double[][] results = new double[nthreads][2];
            int k = length / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? length : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        double[] sum = new double[2];
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            sum[0] += elements[idx] * elemsOther[idxOther] + elements[idx + 1]
                                    * elemsOther[idxOther + 1];
                            sum[1] += elements[idx + 1] * elemsOther[idxOther] - elements[idx]
                                    * elemsOther[idxOther + 1];
                            idx += stride;
                            idxOther += strideOther;
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (double[]) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum = DComplex.plus(sum, results[j]);
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int k = 0; k < length; k++) {
                sum[0] += elements[idx] * elemsOther[idxOther] + elements[idx + 1] * elemsOther[idxOther + 1];
                sum[1] += elements[idx + 1] * elemsOther[idxOther] - elements[idx] * elemsOther[idxOther + 1];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return sum;
    }

    public double[] zSum() {
        double[] sum = new double[2];
        if (this.elements == null)
            throw new InternalError();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            double[][] results = new double[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        double[] sum = new double[2];
                        int idx = zero + firstIdx * stride;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            sum[0] += elements[idx];
                            sum[1] += elements[idx + 1];
                            idx += stride;
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (double[]) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum[0] = sum[0] + results[j][0];
                    sum[1] = sum[1] + results[j][1];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int idx = zero;
            for (int k = 0; k < size; k++) {
                sum[0] += elements[idx];
                sum[1] += elements[idx + 1];
                idx += stride;
            }
        }
        return sum;
    }

    protected int cardinality(int maxCardinality) {
        int cardinality = 0;
        int idx = zero;
        int i = 0;
        while (i < size && cardinality < maxCardinality) {
            if (elements[idx] != 0 || elements[idx + 1] != 0)
                cardinality++;
            idx += stride;
            i++;
        }
        return cardinality;
    }

    protected boolean haveSharedCellsRaw(DComplexMatrix1D other) {
        if (other instanceof SelectedDenseDComplexMatrix1D) {
            SelectedDenseDComplexMatrix1D otherMatrix = (SelectedDenseDComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseDComplexMatrix1D) {
            DenseDComplexMatrix1D otherMatrix = (DenseDComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        return zero + rank * stride;
    }

    protected DComplexMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseDComplexMatrix1D(this.elements, offsets);
    }
}
