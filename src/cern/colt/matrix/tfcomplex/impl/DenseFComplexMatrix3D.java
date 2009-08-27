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
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix3D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 3-d matrix holding <tt>complex</tt> elements.
 * <p>
 * Internally holds one single contigous one-dimensional array, addressed in (in
 * decreasing order of significance): slice major, row major, column major.
 * Complex data is represented by 2 float values in sequence, i.e. elements[idx]
 * constitute the real part and elements[idx+1] constitute the imaginary part,
 * where idx = index(0,0,0) + slice * sliceStride + row * rowStride + column *
 * columnStride. Note that this implementation is not synchronized.
 * <p>
 * Applications demanding utmost speed can exploit knowledge about the internal
 * addressing. Setting/getting values in a loop slice-by-slice, row-by-row,
 * column-by-column is quicker than, for example, column-by-column, row-by-row,
 * slice-by-slice. Thus
 * 
 * <pre>
 * for (int slice = 0; slice &lt; slices; slice++) {
 *     for (int row = 0; row &lt; rows; row++) {
 *         for (int column = 0; column &lt; columns; column++) {
 *             matrix.setQuick(slice, row, column, someValue);
 *         }
 *     }
 * }
 * </pre>
 * 
 * is quicker than
 * 
 * <pre>
 * for (int column = 0; column &lt; columns; column++) {
 *     for (int row = 0; row &lt; rows; row++) {
 *         for (int slice = 0; slice &lt; slices; slice++) {
 *             matrix.setQuick(slice, row, column, someValue);
 *         }
 *     }
 * }
 * </pre>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DenseFComplexMatrix3D extends FComplexMatrix3D {
    private static final long serialVersionUID = 1L;

    private FloatFFT_3D fft3;

    /**
     * The elements of this matrix. elements are stored in slice major, then row
     * major, then column major, in order of significance. Complex data is
     * represented by 2 float values in sequence, i.e. elements[idx] constitute
     * the real part and elements[idx+1] constitute the imaginary part, where
     * idx = index(0,0,0) + slice * sliceStride + row * rowStride + column *
     * columnStride.
     */
    protected float[] elements;

    /**
     * Constructs a matrix with a copy of the given values. * <tt>values</tt> is
     * required to have the form
     * <tt>re = values[slice][row][2*column], im = values[slice][row][2*column+1]</tt>
     * and have exactly the same number of rows in every slice and exactly the
     * same number of columns in in every row.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 1 &lt;= slice &lt; values.length: values[slice].length != values[slice-1].length</tt>
     *             .
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 1 &lt;= row &lt; values[0].length: values[slice][row].length != values[slice][row-1].length</tt>
     *             .
     */
    public DenseFComplexMatrix3D(float[][][] values) {
        this(values.length, (values.length == 0 ? 0 : values[0].length), (values.length == 0 ? 0
                : values[0].length == 0 ? 0 : values[0][0].length / 2));
        assign(values);
    }

    /**
     * Constructs a matrix with the same size as <tt>realPart</tt> matrix and
     * fills the real part of this matrix with elements of <tt>realPart</tt>.
     * 
     * @param realPart
     *            a real matrix whose elements become a real part of this matrix
     * @throws IllegalArgumentException
     *             if <tt>(float)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseFComplexMatrix3D(FloatMatrix3D realPart) {
        this(realPart.slices(), realPart.rows(), realPart.columns());
        assignReal(realPart);
    }

    /**
     * Constructs a matrix with a given number of slices, rows and columns. All
     * entries are initially <tt>0</tt>.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>(float)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseFComplexMatrix3D(int slices, int rows, int columns) {
        setUp(slices, rows, columns, 0, 0, 0, rows * 2 * columns, 2 * columns, 2);
        this.elements = new float[slices * rows * 2 * columns];
    }

    /**
     * Constructs a matrix with the given parameters.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param elements
     *            the cells.
     * @param sliceZero
     *            the position of the first element.
     * @param rowZero
     *            the position of the first element.
     * @param columnZero
     *            the position of the first element.
     * @param sliceStride
     *            the number of elements between two slices, i.e.
     *            <tt>index(k+1,i,j)-index(k,i,j)</tt>.
     * @param rowStride
     *            the number of elements between two rows, i.e.
     *            <tt>index(k,i+1,j)-index(k,i,j)</tt>.
     * @param columnStride
     *            the number of elements between two columns, i.e.
     *            <tt>index(k,i,j+1)-index(k,i,j)</tt>.
     * @param isNoView
     *            if false then the view is constructed
     * 
     * @throws IllegalArgumentException
     *             if <tt>(float)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseFComplexMatrix3D(int slices, int rows, int columns, float[] elements, int sliceZero, int rowZero,
            int columnZero, int sliceStride, int rowStride, int columnStride, boolean isNoView) {
        setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = isNoView;
    }

    public float[] aggregate(final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr,
            final cern.colt.function.tfcomplex.FComplexFComplexFunction f) {
        float[] b = new float[2];
        if (size() == 0) {
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        final int zero = (int) index(0, 0, 0);
        float[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {

                    public float[] call() throws Exception {
                        int idx = zero + firstSlice * sliceStride;
                        float[] a = f.apply(new float[] { elements[idx], elements[idx + 1] });
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    idx = zero + s * sliceStride + r * rowStride + c * columnStride;
                                    a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }));
                                }
                                d = 0;
                            }
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(new float[] { elements[zero], elements[zero + 1] });
            int d = 1; // first cell already done
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        idx = zero + s * sliceStride + r * rowStride + c * columnStride;
                        a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }));
                    }
                    d = 0;
                }
            }
        }
        return a;
    }

    public float[] aggregate(final FComplexMatrix3D other,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction f) {
        checkShape(other);
        float[] b = new float[2];
        if (size() == 0) {
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) other.index(0, 0, 0);
        final int sliceStrideOther = other.sliceStride();
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final float[] elemsOther = (float[]) other.elements();

        float[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {

                    public float[] call() throws Exception {
                        int idx = zero + firstSlice * sliceStride;
                        int idxOther = zeroOther + firstSlice * sliceStrideOther;
                        float[] a = f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] {
                                elemsOther[idxOther], elemsOther[idxOther + 1] });
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    idx = zero + s * sliceStride + r * rowStride + c * columnStride;
                                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c
                                            * colStrideOther;
                                    a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] },
                                            new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] }));
                                }
                                d = 0;
                            }
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(new float[] { elements[zero], elements[zero + 1] }, new float[] { elemsOther[zeroOther],
                    elemsOther[zeroOther + 1] });
            int d = 1; // first cell already done
            int idx;
            int idxOther;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        idx = zero + s * sliceStride + r * rowStride + c * columnStride;
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c * colStrideOther;
                        a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] {
                                elemsOther[idxOther], elemsOther[idxOther + 1] }));
                    }
                    d = 0;
                }
            }
        }
        return a;
    }

    public FComplexMatrix3D assign(final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        float[] elem = new float[2];
                        if (function instanceof cern.jet.math.tfcomplex.FComplexMult) {
                            float[] multiplicator = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
                            // x[i] = mult*x[i]
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    for (int c = 0; c < columns; c++) {
                                        elem[0] = elements[idx];
                                        elem[1] = elements[idx + 1];
                                        elements[idx] = elem[0] * multiplicator[0] - elem[1] * multiplicator[1];
                                        elements[idx + 1] = elem[1] * multiplicator[0] + elem[0] * multiplicator[1];
                                        idx += columnStride;
                                    }
                                }
                            }
                        } else {
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    for (int c = 0; c < columns; c++) {
                                        elem[0] = elements[idx];
                                        elem[1] = elements[idx + 1];
                                        elem = function.apply(elem);
                                        elements[idx] = elem[0];
                                        elements[idx + 1] = elem[1];
                                        idx += columnStride;
                                    }
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);

        } else {
            int idx;
            float[] elem = new float[2];
            if (function instanceof cern.jet.math.tfcomplex.FComplexMult) {
                float[] multiplicator = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
                // x[i] = mult*x[i]
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        for (int c = 0; c < columns; c++) {
                            elem[0] = elements[idx];
                            elem[1] = elements[idx + 1];
                            elements[idx] = elem[0] * multiplicator[0] - elem[1] * multiplicator[1];
                            elements[idx + 1] = elem[1] * multiplicator[0] + elem[0] * multiplicator[1];
                            idx += columnStride;
                        }
                    }
                }
            } else {
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        for (int c = 0; c < columns; c++) {
                            elem[0] = elements[idx];
                            elem[1] = elements[idx + 1];
                            elem = function.apply(elem);
                            elements[idx] = elem[0];
                            elements[idx + 1] = elem[1];
                            idx += columnStride;
                        }
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond,
            final cern.colt.function.tfcomplex.FComplexFComplexFunction f) {
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        float[] elem = new float[2];
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    elem[0] = elements[idx];
                                    elem[1] = elements[idx + 1];
                                    if (cond.apply(elem) == true) {
                                        elem = f.apply(elem);
                                        elements[idx] = elem[0];
                                        elements[idx + 1] = elem[1];
                                    }
                                    idx += columnStride;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] elem = new float[2];
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        elem[0] = elements[idx];
                        elem[1] = elements[idx + 1];
                        if (cond.apply(elem) == true) {
                            elem = f.apply(elem);
                            elements[idx] = elem[0];
                            elements[idx + 1] = elem[1];
                        }
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond, final float[] value) {
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        float[] elem = new float[2];
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    elem[0] = elements[idx];
                                    elem[1] = elements[idx + 1];
                                    if (cond.apply(elem) == true) {
                                        elements[idx] = value[0];
                                        elements[idx + 1] = value[1];
                                    }
                                    idx += columnStride;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] elem = new float[2];
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        elem[0] = elements[idx];
                        elem[1] = elements[idx + 1];
                        if (cond.apply(elem) == true) {
                            elements[idx] = value[0];
                            elements[idx + 1] = value[1];
                        }
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(final cern.colt.function.tfcomplex.FComplexRealFunction function) {
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        float[] elem = new float[2];
                        if (function == cern.jet.math.tfcomplex.FComplexFunctions.abs) {
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    for (int c = 0; c < columns; c++) {
                                        elem[0] = elements[idx];
                                        elem[1] = elements[idx + 1];
                                        float absX = Math.abs(elements[idx]);
                                        float absY = Math.abs(elements[idx + 1]);
                                        if (absX == 0 && absY == 0) {
                                            elements[idx] = 0;
                                        } else if (absX >= absY) {
                                            float d = elem[1] / elem[0];
                                            elements[idx] = absX * (float) Math.sqrt(1 + d * d);
                                        } else {
                                            float d = elem[0] / elem[1];
                                            elements[idx] = absY * (float) Math.sqrt(1 + d * d);
                                        }
                                        elements[idx + 1] = 0;
                                        idx += columnStride;
                                    }
                                }
                            }
                        } else {
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    for (int c = 0; c < columns; c++) {
                                        elem[0] = elements[idx];
                                        elem[1] = elements[idx + 1];
                                        elem[0] = function.apply(elem);
                                        elements[idx] = elem[0];
                                        elements[idx + 1] = 0;
                                        idx += columnStride;
                                    }
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);

        } else {
            int idx;
            float[] elem = new float[2];
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        elem[0] = elements[idx];
                        elem[1] = elements[idx + 1];
                        elem[0] = function.apply(elem);
                        elements[idx] = elem[0];
                        elements[idx + 1] = 0;
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(FComplexMatrix3D source) {
        // overriden for performance only
        if (!(source instanceof DenseFComplexMatrix3D)) {
            super.assign(source);
            return this;
        }
        DenseFComplexMatrix3D other = (DenseFComplexMatrix3D) source;
        if (other == this)
            return this;
        checkShape(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            FComplexMatrix3D c = other.copy();
            if (!(c instanceof DenseFComplexMatrix3D)) { // should not happen
                super.assign(source);
                return this;
            }
            other = (DenseFComplexMatrix3D) c;
        }
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) other.index(0, 0, 0);
        final int sliceStrideOther = other.sliceStride;
        final int rowStrideOther = other.rowStride;
        final int columnStrideOther = other.columnStride;
        final float[] elemsOther = other.elements;
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                for (int c = 0; c < columns; c++) {
                                    elements[idx] = elemsOther[idxOther];
                                    elements[idx + 1] = elemsOther[idxOther + 1];
                                    idx += columnStride;
                                    idxOther += columnStrideOther;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                    for (int c = 0; c < columns; c++) {
                        elements[idx] = elemsOther[idxOther];
                        elements[idx + 1] = elemsOther[idxOther + 1];
                        idx += columnStride;
                        idxOther += columnStrideOther;
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(final FComplexMatrix3D y,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction function) {
        checkShape(y);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) y.index(0, 0, 0);
        final int colStrideOther = y.columnStride();
        final int sliceStrideOther = y.sliceStride();
        final int rowStrideOther = y.rowStride();
        final float[] elemsOther = (float[]) y.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
                        float[] tmp1 = new float[2];
                        float[] tmp2 = new float[2];
                        if (function == cern.jet.math.tfcomplex.FComplexFunctions.mult) {
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                    for (int c = 0; c < columns; c++) {
                                        tmp1[0] = elements[idx];
                                        tmp1[1] = elements[idx + 1];
                                        tmp2[0] = elemsOther[idxOther];
                                        tmp2[1] = elemsOther[idxOther + 1];
                                        elements[idx] = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1];
                                        elements[idx + 1] = tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                                        idx += columnStride;
                                        idxOther += colStrideOther;
                                    }
                                }
                            }
                        } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjFirst) {
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                    for (int c = 0; c < columns; c++) {
                                        tmp1[0] = elements[idx];
                                        tmp1[1] = elements[idx + 1];
                                        tmp2[0] = elemsOther[idxOther];
                                        tmp2[1] = elemsOther[idxOther + 1];
                                        elements[idx] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                                        elements[idx + 1] = -tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                                        idx += columnStride;
                                        idxOther += colStrideOther;
                                    }
                                }
                            }

                        } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjSecond) {
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                    for (int c = 0; c < columns; c++) {
                                        tmp1[0] = elements[idx];
                                        tmp1[1] = elements[idx + 1];
                                        tmp2[0] = elemsOther[idxOther];
                                        tmp2[1] = elemsOther[idxOther + 1];
                                        elements[idx] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                                        elements[idx + 1] = tmp1[1] * tmp2[0] - tmp1[0] * tmp2[1];
                                        idx += columnStride;
                                        idxOther += colStrideOther;
                                    }
                                }
                            }
                        } else {
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                    for (int c = 0; c < columns; c++) {
                                        tmp1[0] = elements[idx];
                                        tmp1[1] = elements[idx + 1];
                                        tmp2[0] = elemsOther[idxOther];
                                        tmp2[1] = elemsOther[idxOther + 1];
                                        tmp1 = function.apply(tmp1, tmp2);
                                        elements[idx] = tmp1[0];
                                        elements[idx + 1] = tmp1[1];
                                        idx += columnStride;
                                        idxOther += colStrideOther;
                                    }
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            float[] tmp1 = new float[2];
            float[] tmp2 = new float[2];
            if (function == cern.jet.math.tfcomplex.FComplexFunctions.mult) {
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                        for (int c = 0; c < columns; c++) {
                            tmp1[0] = elements[idx];
                            tmp1[1] = elements[idx + 1];
                            tmp2[0] = elemsOther[idxOther];
                            tmp2[1] = elemsOther[idxOther + 1];
                            elements[idx] = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1];
                            elements[idx + 1] = tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                            idx += columnStride;
                            idxOther += colStrideOther;
                        }
                    }
                }
            } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjFirst) {
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                        for (int c = 0; c < columns; c++) {
                            tmp1[0] = elements[idx];
                            tmp1[1] = elements[idx + 1];
                            tmp2[0] = elemsOther[idxOther];
                            tmp2[1] = elemsOther[idxOther + 1];
                            elements[idx] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                            elements[idx + 1] = -tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                            idx += columnStride;
                            idxOther += colStrideOther;
                        }
                    }
                }

            } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjSecond) {
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                        for (int c = 0; c < columns; c++) {
                            tmp1[0] = elements[idx];
                            tmp1[1] = elements[idx + 1];
                            tmp2[0] = elemsOther[idxOther];
                            tmp2[1] = elemsOther[idxOther + 1];
                            elements[idx] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                            elements[idx + 1] = tmp1[1] * tmp2[0] - tmp1[0] * tmp2[1];
                            idx += columnStride;
                            idxOther += colStrideOther;
                        }
                    }
                }
            } else {
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                        for (int c = 0; c < columns; c++) {
                            tmp1[0] = elements[idx];
                            tmp1[1] = elements[idx + 1];
                            tmp2[0] = elemsOther[idxOther];
                            tmp2[1] = elemsOther[idxOther + 1];
                            tmp1 = function.apply(tmp1, tmp2);
                            elements[idx] = tmp1[0];
                            elements[idx + 1] = tmp1[1];
                            idx += columnStride;
                            idxOther += colStrideOther;
                        }
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(final float re, final float im) {
        if (this.isNoView == false) {
            return super.assign(re, im);
        }
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    elements[idx] = re;
                                    elements[idx + 1] = im;
                                    idx += columnStride;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);

        } else {
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        elements[idx] = re;
                        elements[idx + 1] = im;
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(final float[] values) {
        if (values.length != slices * rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length
                    + "slices()*rows()*2*columns()=" + slices() * rows() * 2 * columns());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            System.arraycopy(values, 0, elements, 0, values.length);
        } else {
            final int zero = (int) index(0, 0, 0);
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
                nthreads = Math.min(nthreads, slices);
                Future<?>[] futures = new Future[nthreads];
                int k = slices / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstSlice = j * k;
                    final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int idxOther = firstSlice * 2 * rows * columns;
                            int idx;
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    for (int c = 0; c < columns; c++) {
                                        elements[idx] = values[idxOther++];
                                        elements[idx + 1] = values[idxOther++];
                                        idx += columnStride;
                                    }
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idxOther = 0;
                int idx;
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        for (int c = 0; c < columns; c++) {
                            elements[idx] = values[idxOther++];
                            elements[idx + 1] = values[idxOther++];
                            idx += columnStride;
                        }
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assign(final float[][][] values) {
        if (values.length != slices)
            throw new IllegalArgumentException("Must have same number of slices: slices=" + values.length + "slices()="
                    + slices());
        final int length = 2 * columns;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
                nthreads = Math.min(nthreads, slices);
                Future<?>[] futures = new Future[nthreads];
                int k = slices / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstSlice = j * k;
                    final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int i = firstSlice * sliceStride;
                            for (int s = firstSlice; s < lastSlice; s++) {
                                float[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    float[] currentRow = currentSlice[r];
                                    if (currentRow.length != length)
                                        throw new IllegalArgumentException(
                                                "Must have same number of columns in every row: columns="
                                                        + currentRow.length + "2 * columns()=" + length);
                                    System.arraycopy(currentRow, 0, elements, i, length);
                                    i += length;
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int i = 0;
                for (int s = 0; s < slices; s++) {
                    float[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        float[] currentRow = currentSlice[r];
                        if (currentRow.length != length)
                            throw new IllegalArgumentException(
                                    "Must have same number of columns in every row: columns=" + currentRow.length
                                            + "2 * columns()=" + length);
                        System.arraycopy(currentRow, 0, elements, i, length);
                        i += length;
                    }
                }
            }
        } else {
            final int zero = (int) index(0, 0, 0);
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
                nthreads = Math.min(nthreads, slices);
                Future<?>[] futures = new Future[nthreads];
                int k = slices / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstSlice = j * k;
                    final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            int idx;
                            for (int s = firstSlice; s < lastSlice; s++) {
                                float[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    float[] currentRow = currentSlice[r];
                                    if (currentRow.length != length)
                                        throw new IllegalArgumentException(
                                                "Must have same number of columns in every row: columns="
                                                        + currentRow.length + "2*columns()=" + length);
                                    for (int c = 0; c < columns; c++) {
                                        elements[idx] = currentRow[2 * c];
                                        elements[idx + 1] = currentRow[2 * c + 1];
                                        idx += columnStride;
                                    }
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);

            } else {
                int idx;
                for (int s = 0; s < slices; s++) {
                    float[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        float[] currentRow = currentSlice[r];
                        if (currentRow.length != length)
                            throw new IllegalArgumentException(
                                    "Must have same number of columns in every row: columns=" + currentRow.length
                                            + "2*columns()=" + length);
                        for (int c = 0; c < columns; c++) {
                            elements[idx] = currentRow[2 * c];
                            elements[idx + 1] = currentRow[2 * c + 1];
                            idx += columnStride;
                        }
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assignImaginary(final FloatMatrix3D other) {
        checkShape(other);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) other.index(0, 0, 0);
        final int sliceStrideOther = other.sliceStride();
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final float[] elemsOther = (float[]) other.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                for (int c = 0; c < columns; c++) {
                                    elements[idx + 1] = elemsOther[idxOther];
                                    idx += columnStride;
                                    idxOther += colStrideOther;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                    for (int c = 0; c < columns; c++) {
                        elements[idx + 1] = elemsOther[idxOther];
                        idx += columnStride;
                        idxOther += colStrideOther;
                    }
                }
            }
        }
        return this;
    }

    public FComplexMatrix3D assignReal(final FloatMatrix3D other) {
        checkShape(other);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) other.index(0, 0, 0);
        final int sliceStrideOther = other.sliceStride();
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final float[] elemsOther = (float[]) other.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                for (int c = 0; c < columns; c++) {
                                    elements[idx] = elemsOther[idxOther];
                                    idx += columnStride;
                                    idxOther += colStrideOther;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                    for (int c = 0; c < columns; c++) {
                        elements[idx] = elemsOther[idxOther];
                        idx += columnStride;
                        idxOther += colStrideOther;
                    }
                }
            }
        }
        return this;
    }

    public int cardinality() {
        int cardinality = 0;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    if ((elements[idx] != 0.0) || (elements[idx + 1] != 0.0)) {
                                        cardinality++;
                                    }
                                    idx += columnStride;
                                }
                            }
                        }
                        return Integer.valueOf(cardinality);
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
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        if ((elements[idx] != 0.0) || (elements[idx + 1] != 0.0)) {
                            cardinality++;
                        }
                        idx += columnStride;
                    }
                }
            }
        }
        return cardinality;
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of each slice of this
     * matrix.
     */
    public void fft2Slices() {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            ConcurrencyUtils.setThreadsBeginN_2D(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseFComplexMatrix2D) viewSlice(s)).fft2();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseFComplexMatrix2D) viewSlice(s)).fft2();
            }
        }
    }

    /**
     * Computes the 3D discrete Fourier transform (DFT) of this matrix.
     */
    public void fft3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new FloatFFT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            fft3.complexForward(elements);
        } else {
            FComplexMatrix3D copy = this.copy();
            fft3.complexForward((float[]) copy.elements());
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public float[] elements() {
        return elements;
    }

    public FloatMatrix3D getImaginaryPart() {
        final DenseFloatMatrix3D Im = new DenseFloatMatrix3D(slices, rows, columns);
        final float[] elemsOther = Im.elements();
        final int sliceStrideOther = Im.sliceStride();
        final int rowStrideOther = Im.rowStride();
        final int columnStrideOther = Im.columnStride();
        final int zeroOther = (int) Im.index(0, 0, 0);
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                for (int c = 0; c < columns; c++) {
                                    elemsOther[idxOther] = elements[idx + 1];
                                    idx += columnStride;
                                    idxOther += columnStrideOther;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                    for (int c = 0; c < columns; c++) {
                        elemsOther[idxOther] = elements[idx + 1];
                        idx += columnStride;
                        idxOther += columnStrideOther;
                    }
                }
            }
        }
        return Im;
    }

    public void getNonZeros(final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList,
            final ArrayList<float[]> valueList) {
        sliceList.clear();
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int zero = (int) index(0, 0, 0);

        int idx;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                idx = zero + s * sliceStride + r * rowStride;
                for (int c = 0; c < columns; c++) {
                    float[] elem = new float[2];
                    elem[0] = elements[idx];
                    elem[1] = elements[idx + 1];
                    if (elem[0] != 0 || elem[1] != 0) {
                        sliceList.add(s);
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(elem);
                    }
                    idx += columnStride;
                }
            }

        }
    }

    public float[] getQuick(int slice, int row, int column) {
        int idx = sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
        return new float[] { elements[idx], elements[idx + 1] };
    }

    public FloatMatrix3D getRealPart() {
        final DenseFloatMatrix3D R = new DenseFloatMatrix3D(slices, rows, columns);
        final float[] elemsOther = R.elements();
        final int sliceStrideOther = R.sliceStride();
        final int rowStrideOther = R.rowStride();
        final int columnStrideOther = R.columnStride();
        final int zeroOther = (int) R.index(0, 0, 0);
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                                for (int c = 0; c < columns; c++) {
                                    elemsOther[idxOther] = elements[idx];
                                    idx += columnStride;
                                    idxOther += columnStrideOther;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                    for (int c = 0; c < columns; c++) {
                        elemsOther[idxOther] = elements[idx];
                        idx += columnStride;
                        idxOther += columnStrideOther;
                    }
                }
            }
        }
        return R;
    }

    /**
     * Computes the 2D inverse of the discrete Fourier transform (IDFT) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void ifft2Slices(final boolean scale) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            ConcurrencyUtils.setThreadsBeginN_2D(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseFComplexMatrix2D) viewSlice(s)).ifft2(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseFComplexMatrix2D) viewSlice(s)).ifft2(scale);
            }
        }
    }

    /**
     * Computes the 3D inverse of the discrete Fourier transform (IDFT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void ifft3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new FloatFFT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            fft3.complexInverse(elements, scale);
        } else {
            FComplexMatrix3D copy = this.copy();
            fft3.complexInverse((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public FComplexMatrix3D like(int slices, int rows, int columns) {
        return new DenseFComplexMatrix3D(slices, rows, columns);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return new DenseFComplexMatrix2D(rows, columns);
    }

    public void setQuick(int slice, int row, int column, float re, float im) {
        int idx = sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = re;
        elements[idx + 1] = im;
    }

    public void setQuick(int slice, int row, int column, float[] value) {
        int idx = sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = value[0];
        elements[idx + 1] = value[1];
    }

    public float[][][] toArray() {
        final int zero = (int) index(0, 0, 0);
        final float[][][] values = new float[slices][rows][2 * columns];
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            float[][] currentSlice = values[s];
                            for (int r = 0; r < rows; r++) {
                                float[] currentRow = currentSlice[r];
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    currentRow[2 * c] = elements[idx];
                                    currentRow[2 * c + 1] = elements[idx + 1];
                                    idx += columnStride;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            for (int s = 0; s < slices; s++) {
                float[][] currentSlice = values[s];
                for (int r = 0; r < rows; r++) {
                    float[] currentRow = currentSlice[r];
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        currentRow[2 * c] = elements[idx];
                        currentRow[2 * c + 1] = elements[idx + 1];
                        idx += columnStride;
                    }
                }
            }
        }
        return values;
    }

    public FComplexMatrix1D vectorize() {
        FComplexMatrix1D v = new DenseFComplexMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            FComplexMatrix2D slice = viewSlice(s);
            v.viewPart(s * length, length).assign(slice.vectorize());
        }
        return v;
    }

    public float[] zSum() {
        float[] sum = new float[2];
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {

                    public float[] call() throws Exception {
                        float[] sum = new float[2];
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    sum[0] += elements[idx];
                                    sum[1] += elements[idx + 1];
                                    idx += columnStride;
                                }
                            }
                        }
                        return sum;
                    }
                });
            }
            float[] tmp;
            try {
                for (int j = 0; j < nthreads; j++) {
                    tmp = (float[]) futures[j].get();
                    sum[0] += tmp[0];
                    sum[1] += tmp[1];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        sum[0] += elements[idx];
                        sum[1] += elements[idx + 1];
                        idx += columnStride;
                    }
                }
            }
        }
        return sum;
    }

    protected boolean haveSharedCellsRaw(FComplexMatrix3D other) {
        if (other instanceof SelectedDenseFComplexMatrix3D) {
            SelectedDenseFComplexMatrix3D otherMatrix = (SelectedDenseFComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFComplexMatrix3D) {
            DenseFComplexMatrix3D otherMatrix = (DenseFComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int slice, int row, int column) {
        return sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
    }

    protected FComplexMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
            int columnStride) {
        return new DenseFComplexMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride,
                false);
    }

    protected FComplexMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseFComplexMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
    }
}
