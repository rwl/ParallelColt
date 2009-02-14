/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.impl;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 1-d matrix (aka <i>vector</i>) holding <tt>complex</tt> elements.
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contiguous one-dimensional array. Complex data is
 * represented by 2 float values in sequence, i.e. elements[zero + 2 * k *
 * stride] constitute real part and elements[zero + 2 * k * stride + 1]
 * constitute imaginary part (k=0,...,size()-1).
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*2*size()</tt>. Thus, a 1000000 matrix uses 16 MB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 * <p>
 * 
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseFComplexMatrix1D extends FComplexMatrix1D {

    private static final long serialVersionUID = 7295427570770814934L;

    private FloatFFT_1D fft;

    /**
     * The elements of this matrix. Complex data is represented by 2 float
     * values in sequence, i.e. elements[zero + 2 * k * stride] constitute real
     * part and elements[zero + 2 * k * stride] constitute imaginary part
     * (k=0,...,size()-1).
     */
    protected float[] elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in
     * the matrix, and vice-versa. Due to the fact that complex data is
     * represented by 2 float values in sequence: the real and imaginary parts,
     * the size of new matrix will be equal to values.length / 2.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public DenseFComplexMatrix1D(float[] values) {
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
    public DenseFComplexMatrix1D(FloatMatrix1D realPart) {
        this(realPart.size());
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
    public DenseFComplexMatrix1D(int size) {
        setUp(size, 0, 2);
        this.isNoView = true;
        this.elements = new float[2 * size];
    }

    /**
     * Constructs a matrix view with the given parameters.
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
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public DenseFComplexMatrix1D(int size, float[] elements, int zero, int stride) {
        setUp(size, zero, stride);
        this.elements = elements;
        this.isNoView = false;
    }

    public float[] aggregate(final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr, final cern.colt.function.tfcomplex.FComplexFComplexFunction f) {
        float[] b = new float[2];
        if (size == 0) {
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        float[] a = null;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            float[][] results = new float[np][2];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        int idx = zero + startidx * stride;
                        float[] a = f.apply(new float[] { elements[idx], elements[idx + 1] });
                        for (int i = startidx + 1; i < stopidx; i++) {
                            idx += stride;
                            a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(new float[] { elements[zero], elements[zero + 1] });
            int idx = zero;
            for (int i = 1; i < size; i++) {
                idx += stride;
                a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }));
            }
        }
        return a;
    }

    public float[] aggregate(final FComplexMatrix1D other, final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr, final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction f) {
        if (!(other instanceof DenseFComplexMatrix1D)) {
            return super.aggregate(other, aggr, f);
        }
        checkSize(other);
        if (size == 0) {
            float[] b = new float[2];
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        final int zeroOther = (int)other.index(0);
        final int strideOther = other.stride();
        final float[] elemsOther = (float[]) other.elements();
        float[] a = null;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            float[][] results = new float[np][2];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        int idx = zero + startidx * stride;
                        int idxOther = zeroOther + startidx * strideOther;
                        float[] a = f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] });
                        for (int i = startidx + 1; i < stopidx; i++) {
                            idx += stride;
                            idxOther += strideOther;
                            a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] }));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            a = f.apply(new float[] { elements[zero], elements[zero + 1] }, new float[] { elemsOther[zeroOther], elemsOther[zeroOther + 1] });
            for (int i = 1; i < size; i++) {
                idx += stride;
                idxOther += strideOther;
                a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] }));
            }
        }
        return a;
    }

    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
        if (this.elements == null)
            throw new InternalError();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            if (function instanceof cern.jet.math.tfcomplex.FComplexMult) {
                float[] multiplicator = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
                if (multiplicator[0] == 1 && multiplicator[1] == 0)
                    return this;
            }
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
                        float[] tmp = new float[2];
                        int idx = zero + startidx * stride;
                        for (int k = startidx; k < stopidx; k++) {
                            tmp[0] = elements[idx];
                            tmp[1] = elements[idx + 1];
                            tmp = function.apply(tmp);
                            elements[idx] = tmp[0];
                            elements[idx + 1] = tmp[1];
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp = new float[2];
            int idx = zero;
            for (int k = 0; k < size; k++) {
                tmp[0] = elements[idx];
                tmp[1] = elements[idx + 1];
                tmp = function.apply(tmp);
                elements[idx] = tmp[0];
                elements[idx + 1] = tmp[1];
                idx += stride;
            }
        }
        return this;
    }

    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond, final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
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
                        float[] elem = new float[2];
                        int idx = zero + startidx * stride;
                        for (int i = startidx; i < stopidx; i++) {
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
            float[] elem = new float[2];
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

    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond, final float[] value) {
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
                        float[] elem = new float[2];
                        int idx = zero + startidx * stride;
                        for (int i = startidx; i < stopidx; i++) {
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
            float[] elem = new float[2];
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

    public FComplexMatrix1D assign(final cern.colt.function.tfcomplex.FComplexRealFunction function) {
        if (this.elements == null)
            throw new InternalError();
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
                        float[] tmp = new float[2];
                        int idx = zero + startidx * stride;
                        for (int k = startidx; k < stopidx; k++) {
                            tmp[0] = elements[idx];
                            tmp[1] = elements[idx + 1];
                            tmp[0] = function.apply(tmp);
                            elements[idx] = tmp[0];
                            elements[idx + 1] = 0;
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp = new float[2];
            int idx = zero;
            for (int k = 0; k < size; k++) {
                tmp[0] = elements[idx];
                tmp[1] = elements[idx + 1];
                tmp[0] = function.apply(tmp);
                elements[idx] = tmp[0];
                elements[idx + 1] = 0;
                idx += stride;
            }
        }
        return this;
    }

    public FComplexMatrix1D assign(FComplexMatrix1D source) {
        if (!(source instanceof DenseFComplexMatrix1D)) {
            return super.assign(source);
        }
        DenseFComplexMatrix1D other = (DenseFComplexMatrix1D) source;
        if (other == this)
            return this;
        checkSize(other);
        if (isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            FComplexMatrix1D c = other.copy();
            if (!(c instanceof DenseFComplexMatrix1D)) { // should not happen
                return super.assign(source);
            }
            other = (DenseFComplexMatrix1D) c;
        }

        final float[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int strideOther = other.stride;
        final int zeroOther = (int)other.index(0);

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

    public FComplexMatrix1D assign(FComplexMatrix1D y, final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction function) {
        if (!(y instanceof DenseFComplexMatrix1D)) {
            return super.assign(y, function);
        }
        checkSize(y);
        final float[] elemsOther = (float[]) y.elements();
        final int zeroOther = (int)y.index(0);
        final int strideOther = y.stride();

        if (elements == null || elemsOther == null)
            throw new InternalError();
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
                        float[] tmp1 = new float[2];
                        float[] tmp2 = new float[2];
                        int idx = zero + startidx * stride;
                        int idxOther = zeroOther + startidx * strideOther;
                        for (int k = startidx; k < stopidx; k++) {
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
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp1 = new float[2];
            float[] tmp2 = new float[2];
            int idx = zero;
            int idxOther = zeroOther;
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
        return this;
    }

    public FComplexMatrix1D assign(final float re, final float im) {
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

    public FComplexMatrix1D assign(float[] values) {
        if (isNoView) {
            if (values.length != 2 * size)
                throw new IllegalArgumentException("The length of values[] must be equal to 2*size()=" + 2 * size());
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            super.assign(values);
        }
        return this;
    }

    public FComplexMatrix1D assignImaginary(final FloatMatrix1D other) {
        if (!(other instanceof DenseFloatMatrix1D)) {
            return super.assignImaginary(other);
        }
        checkSize(other);
        final int zeroOther = (int)other.index(0);
        final int strideOther = other.stride();
        final float[] elemsOther = (float[]) other.elements();
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
                        for (int i = startidx; i < stopidx; i++) {
//                            elements[idx] = 0;
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
//                elements[idx] = 0;
                elements[idx + 1] = elemsOther[idxOther];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return this;
    }

    public FComplexMatrix1D assignReal(final FloatMatrix1D other) {
        if (!(other instanceof DenseFloatMatrix1D)) {
            return super.assignReal(other);
        }
        checkSize(other);
        final int zeroOther = (int)other.index(0);
        final int strideOther = other.stride();
        final float[] elemsOther = (float[]) other.elements();
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
                        for (int i = startidx; i < stopidx; i++) {
                            elements[idx] = elemsOther[idxOther];
//                            elements[idx + 1] = 0;
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
//                elements[idx + 1] = 0;
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
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (fft == null) {
            fft = new FloatFFT_1D(size);
        }
        
        if (isNoView) {
            fft.complexForward(elements);
        } else {
            FComplexMatrix1D copy = this.copy();
            fft.complexForward((float[]) copy.elements());
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    public float[] elements() {
        return elements;
    }

    public FloatMatrix1D getImaginaryPart() {
        final DenseFloatMatrix1D Im = new DenseFloatMatrix1D(size);
        final float[] elemsOther = (float[]) Im.elements();
        final int zeroOther = (int)Im.index(0);
        final int strideOther = Im.stride();
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

    public void getNonZeros(final IntArrayList indexList, final ArrayList<float[]> valueList) {
        indexList.clear();
        valueList.clear();
        int s = size();

        int idx = zero;
        for (int k = 0; k < s; k++) {
            float[] value = new float[2];
            value[0] = elements[idx];
            value[1] = elements[idx + 1];
            if (value[0] != 0 || value[1] != 0) {
                synchronized (indexList) {
                    indexList.add(k);
                    valueList.add(value);
                }
            }
            idx += stride;
        }
    }

    public float[] getQuick(int index) {
        int idx = zero + index * stride;
        return new float[] { elements[idx], elements[idx + 1] };
    }

    public FloatMatrix1D getRealPart() {
        final DenseFloatMatrix1D R = new DenseFloatMatrix1D(size);
        final float[] elemsOther = (float[]) R.elements();
        final int zeroOther = (int)R.index(0);
        final int strideOther = R.stride();
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
        int oldNp = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNp));
        if (fft == null) {
            fft = new FloatFFT_1D(size);
        }
        if (isNoView) {
            fft.complexInverse(elements, scale);
        } else {
            FComplexMatrix1D copy = this.copy();
            fft.complexInverse((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNp);
    }

    public FComplexMatrix1D like(int size) {
        return new DenseFComplexMatrix1D(size);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return new DenseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix2D reshape(final int rows, final int cols) {
        if (rows * cols != size) {
            throw new IllegalArgumentException("rows*cols != size");
        }
        FComplexMatrix2D M = new DenseFComplexMatrix2D(rows, cols);
        final float[] elemsOther = (float[]) M.elements();
        final int zeroOther = (int)M.index(0, 0);
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
            for (int c = 0; c < cols; c++) {
                idxOther = zeroOther + c * colStrideOther;
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

    public FComplexMatrix3D reshape(final int slices, final int rows, final int cols) {
        if (slices * rows * cols != size) {
            throw new IllegalArgumentException("slices*rows*cols != size");
        }
        FComplexMatrix3D M = new DenseFComplexMatrix3D(slices, rows, cols);
        final float[] elemsOther = (float[]) M.elements();
        final int zeroOther = (int)M.index(0, 0, 0);
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
                for (int c = 0; c < cols; c++) {
                    idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
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

    public void setQuick(int index, float re, float im) {
        int idx = zero + index * stride;
        this.elements[idx] = re;
        this.elements[idx + 1] = im;
    }

    public void setQuick(int index, float[] value) {
        int idx = zero + index * stride;
        this.elements[idx] = value[0];
        this.elements[idx + 1] = value[1];
    }

    public void swap(FComplexMatrix1D other) {
        if (!(other instanceof DenseFComplexMatrix1D)) {
            super.swap(other);
        }
        DenseFComplexMatrix1D y = (DenseFComplexMatrix1D) other;
        if (y == this)
            return;
        checkSize(y);

        final float[] elemsOther = y.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int strideOther = y.stride;
        final int zeroOther = (int)y.index(0);

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
                        float tmp;
                        for (int k = startidx; k < stopidx; k++) {
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
            float tmp;
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

    public void toArray(float[] values) {
        if (values.length < 2 * size)
            throw new IllegalArgumentException("values too small");
        if (this.isNoView)
            System.arraycopy(this.elements, 0, values, 0, this.elements.length);
        else
            super.toArray(values);
    }

    public float[] zSum() {
        float[] sum = new float[2];
        if (this.elements == null)
            throw new InternalError();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            Future<?>[] futures = new Future[np];
            float[][] results = new float[np][2];
            int k = size / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = size;
                } else {
                    stopidx = startidx + k;
                }

                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float[] sum = new float[2];
                        int idx = zero + startidx * stride;
                        for (int k = startidx; k < stopidx; k++) {
                            sum[0] += elements[idx];
                            sum[1] += elements[idx + 1];
                            idx += stride;
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (float[]) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < np; j++) {
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

    protected boolean haveSharedCellsRaw(FComplexMatrix1D other) {
        if (other instanceof SelectedDenseFComplexMatrix1D) {
            SelectedDenseFComplexMatrix1D otherMatrix = (SelectedDenseFComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFComplexMatrix1D) {
            DenseFComplexMatrix1D otherMatrix = (DenseFComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        return zero + rank * stride;
    }

    protected FComplexMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseFComplexMatrix1D(this.elements, offsets);
    }
}
