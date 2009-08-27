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
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>complex</tt> elements. <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array, addressed in row
 * major. Complex data is represented by 2 float values in sequence, i.e.
 * elements[idx] constitute the real part and elements[idx+1] constitute the
 * imaginary part, where idx = index(0,0) + row * rowStride + column *
 * columnStride. Note that this implementation is not synchronized.
 * <p>
 * Cells are internally addressed in row-major. Applications demanding utmost
 * speed can exploit this fact. Setting/getting values in a loop row-by-row is
 * quicker than column-by-column. Thus
 * 
 * <pre>
 * for (int row = 0; row &lt; rows; row++) {
 *     for (int column = 0; column &lt; columns; column++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * is quicker than
 * 
 * <pre>
 * for (int column = 0; column &lt; columns; column++) {
 *     for (int row = 0; row &lt; rows; row++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseFComplexMatrix2D extends FComplexMatrix2D {
    static final long serialVersionUID = 1020177651L;

    private FloatFFT_2D fft2;

    /**
     * The elements of this matrix. elements are stored in row major. Complex
     * data is represented by 2 float values in sequence, i.e. elements[idx]
     * constitute the real part and elements[idx+1] constitute the imaginary
     * part, where idx = index(0,0) + row * rowStride + column * columnStride.
     */
    protected float[] elements;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form
     * <tt>re = values[row][2*column]; im = values[row][2*column+1]</tt> and
     * have exactly the same number of rows and columns as the receiver. Due to
     * the fact that complex data is represented by 2 float values in sequence:
     * the real and imaginary parts, the new matrix will be of the size
     * values.length by values[0].length / 2.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 1 &lt;= row &lt; values.length: values[row].length != values[row-1].length</tt>
     *             .
     */
    public DenseFComplexMatrix2D(float[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length / 2);
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
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public DenseFComplexMatrix2D(FloatMatrix2D realPart) {
        this(realPart.rows(), realPart.columns());
        assignReal(realPart);
    }

    /**
     * Constructs a matrix with a given number of rows and columns. All entries
     * are initially <tt>0</tt>.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public DenseFComplexMatrix2D(int rows, int columns) {
        setUp(rows, columns, 0, 0, 2 * columns, 2);
        this.elements = new float[rows * 2 * columns];
    }

    /**
     * Constructs a matrix with the given parameters.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param elements
     *            the cells.
     * @param rowZero
     *            the position of the first element.
     * @param columnZero
     *            the position of the first element.
     * @param rowStride
     *            the number of elements between two rows, i.e.
     *            <tt>index(i+1,j)-index(i,j)</tt>.
     * @param columnStride
     *            the number of elements between two columns, i.e.
     *            <tt>index(i,j+1)-index(i,j)</tt>.
     * @param isNoView
     *            if false then the view is constructed
     * 
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    public DenseFComplexMatrix2D(int rows, int columns, float[] elements, int rowZero, int columnZero, int rowStride,
            int columnStride, boolean isNoView) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
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
        final int zero = (int) index(0, 0);
        float[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        int idx = zero + firstRow * rowStride;
                        float[] a = f.apply(elements[idx], elements[idx + 1]);
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                idx = zero + r * rowStride + c * columnStride;
                                a = aggr.apply(a, f.apply(elements[idx], elements[idx + 1]));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero], elements[zero + 1]);
            int d = 1; // first cell already done
            int idx;
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    idx = zero + r * rowStride + c * columnStride;
                    a = aggr.apply(a, f.apply(elements[idx], elements[idx + 1]));
                }
                d = 0;
            }
        }
        return a;
    }

    public float[] aggregate(final FComplexMatrix2D other,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction f) {
        if (!(other instanceof DenseFComplexMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        float[] b = new float[2];
        if (size() == 0) {
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int columnStrideOther = other.columnStride();
        final float[] elemsOther = (float[]) other.elements();
        float[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {

                    public float[] call() throws Exception {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        float[] a = f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] {
                                elemsOther[idxOther], elemsOther[idxOther + 1] });
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                idx = zero + r * rowStride + c * columnStride;
                                idxOther = zeroOther + r * rowStrideOther + c * columnStrideOther;
                                a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] },
                                        new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] }));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            int idx;
            int idxOther;
            a = f.apply(new float[] { elements[zero], elements[zero + 1] }, new float[] { elemsOther[zeroOther],
                    elemsOther[zeroOther + 1] });
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    idx = zero + r * rowStride + c * columnStride;
                    idxOther = zeroOther + r * rowStrideOther + c * columnStrideOther;
                    a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] {
                            elemsOther[idxOther], elemsOther[idxOther + 1] }));
                }
                d = 0;
            }
        }
        return a;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tfcomplex.FComplexMult) {
                float[] multiplicator = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
                if (multiplicator[0] == 1 && multiplicator[1] == 0)
                    return this;
                if (multiplicator[0] == 0 && multiplicator[1] == 0)
                    return assign(0, 0);
            }
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        float[] tmp = new float[2];
                        if (function instanceof cern.jet.math.tfcomplex.FComplexMult) {
                            float[] multiplicator = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
                            // x[i] = mult*x[i]
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    tmp[0] = elements[i];
                                    tmp[1] = elements[i + 1];
                                    elements[i] = tmp[0] * multiplicator[0] - tmp[1] * multiplicator[1];
                                    elements[i + 1] = tmp[1] * multiplicator[0] + tmp[0] * multiplicator[1];
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        } else {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    tmp = function.apply(elements[i], elements[i + 1]);
                                    elements[i] = tmp[0];
                                    elements[i + 1] = tmp[1];
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            float[] tmp = new float[2];
            if (function instanceof cern.jet.math.tfcomplex.FComplexMult) {
                float[] multiplicator = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
                // x[i] = mult*x[i]
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        tmp[0] = elements[i];
                        tmp[1] = elements[i + 1];
                        elements[i] = tmp[0] * multiplicator[0] - tmp[1] * multiplicator[1];
                        elements[i + 1] = tmp[1] * multiplicator[0] + tmp[0] * multiplicator[1];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        tmp = function.apply(elements[i], elements[i + 1]);
                        elements[i] = tmp[0];
                        elements[i + 1] = tmp[1];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond,
            final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        float[] elem = new float[2];
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem[0] = elements[i];
                                elem[1] = elements[i + 1];
                                if (cond.apply(elem) == true) {
                                    elem = function.apply(elem);
                                    elements[i] = elem[0];
                                    elements[i + 1] = elem[1];
                                }
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] elem = new float[2];
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem[0] = elements[i];
                    elem[1] = elements[i + 1];
                    if (cond.apply(elem) == true) {
                        elem = function.apply(elem);
                        elements[i] = elem[0];
                        elements[i + 1] = elem[1];
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond, final float[] value) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        float[] elem = new float[2];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem[0] = elements[i];
                                elem[1] = elements[i + 1];
                                if (cond.apply(elem) == true) {
                                    elements[i] = value[0];
                                    elements[i + 1] = value[1];
                                }
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] elem = new float[2];
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem[0] = elements[i];
                    elem[1] = elements[i + 1];
                    if (cond.apply(elem) == true) {
                        elements[i] = value[0];
                        elements[i + 1] = value[1];
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexRealFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        float[] tmp = new float[2];
                        if (function == cern.jet.math.tfcomplex.FComplexFunctions.abs) {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    tmp[0] = elements[i];
                                    tmp[1] = elements[i + 1];
                                    float absX = Math.abs(elements[i]);
                                    float absY = Math.abs(elements[i + 1]);
                                    if (absX == 0 && absY == 0) {
                                        elements[i] = 0;
                                    } else if (absX >= absY) {
                                        float d = tmp[1] / tmp[0];
                                        elements[i] = absX * (float) Math.sqrt(1 + d * d);
                                    } else {
                                        float d = tmp[0] / tmp[1];
                                        elements[i] = absY * (float) Math.sqrt(1 + d * d);
                                    }
                                    elements[i + 1] = 0;
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        } else {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    tmp[0] = elements[i];
                                    tmp[1] = elements[i + 1];
                                    tmp[0] = function.apply(tmp);
                                    elements[i] = tmp[0];
                                    elements[i + 1] = 0;
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            float[] tmp = new float[2];
            if (function == cern.jet.math.tfcomplex.FComplexFunctions.abs) {
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        tmp[0] = elements[i];
                        tmp[1] = elements[i + 1];
                        float absX = Math.abs(tmp[0]);
                        float absY = Math.abs(tmp[1]);
                        if (absX == 0 && absY == 0) {
                            elements[i] = 0;
                        } else if (absX >= absY) {
                            float d = tmp[1] / tmp[0];
                            elements[i] = absX * (float) Math.sqrt(1 + d * d);
                        } else {
                            float d = tmp[0] / tmp[1];
                            elements[i] = absY * (float) Math.sqrt(1 + d * d);
                        }
                        elements[i + 1] = 0;
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        tmp[0] = elements[i];
                        tmp[1] = elements[i + 1];
                        tmp[0] = function.apply(tmp);
                        elements[i] = tmp[0];
                        elements[i + 1] = 0;
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final FComplexMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof DenseFComplexMatrix2D)) {
            super.assign(source);
            return this;
        }
        DenseFComplexMatrix2D other = (DenseFComplexMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            FComplexMatrix2D c = other.copy();
            if (!(c instanceof DenseFComplexMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseFComplexMatrix2D) c;
        }

        final float[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elements[i] = elemsOther[j];
                                elements[i + 1] = elemsOther[j + 1];
                                i += columnStride;
                                j += columnStrideOther;
                            }
                            idx += rowStride;
                            idxOther += rowStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                    elements[i] = elemsOther[j];
                    elements[i + 1] = elemsOther[j + 1];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final FComplexMatrix2D y,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseFComplexMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final float[] elemsOther = ((DenseFComplexMatrix2D) y).elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int columnStrideOther = y.columnStride();
        final int rowStrideOther = y.rowStride();
        final int zeroOther = (int) y.index(0, 0);
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        float[] tmp1 = new float[2];
                        float[] tmp2 = new float[2];
                        if (function == cern.jet.math.tfcomplex.FComplexFunctions.mult) {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    elements[i] = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1];
                                    elements[i + 1] = tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjFirst) {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                                    elements[i + 1] = -tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }

                        } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjSecond) {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                                    elements[i + 1] = tmp1[1] * tmp2[0] - tmp1[0] * tmp2[1];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    tmp1 = function.apply(tmp1, tmp2);
                                    elements[i] = tmp1[0];
                                    elements[i + 1] = tmp1[1];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
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
            if (function == cern.jet.math.tfcomplex.FComplexFunctions.mult) {
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        elements[i] = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1];
                        elements[i + 1] = tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjFirst) {
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                        elements[i + 1] = -tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }

            } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.multConjSecond) {
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                        elements[i + 1] = tmp1[1] * tmp2[0] - tmp1[0] * tmp2[1];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        tmp1 = function.apply(tmp1, tmp2);
                        elements[i] = tmp1[0];
                        elements[i + 1] = tmp1[1];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final float re, final float im) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elements[i] = re;
                                elements[i + 1] = im;
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elements[i] = re;
                    elements[i + 1] = im;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final float[] values) {
        if (values.length != rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + "rows()*2*columns()="
                    + rows() * 2 * columns());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            final int zero = (int) index(0, 0);
            if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, rows);
                Future<?>[] futures = new Future[nthreads];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int idxOther = firstRow * columns * 2;
                            int idx = zero + firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = values[idxOther++];
                                    elements[i + 1] = values[idxOther++];
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idxOther = 0;
                int idx = zero;
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        elements[i] = values[idxOther++];
                        elements[i + 1] = values[idxOther++];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final float[][] values) {
        if (values.length != rows)
            throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "rows()="
                    + rows());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, rows);
                Future<?>[] futures = new Future[nthreads];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int idx = 2 * columns;
                            int i = firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                float[] currentRow = values[r];
                                if (currentRow.length != idx)
                                    throw new IllegalArgumentException(
                                            "Must have same number of columns in every row: columns="
                                                    + currentRow.length + "2*columns()=" + idx);
                                System.arraycopy(currentRow, 0, elements, i, idx);
                                i += idx;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = 2 * columns;
                int i = 0;
                for (int r = 0; r < rows; r++) {
                    float[] currentRow = values[r];
                    if (currentRow.length != idx)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "2*columns()=" + idx);
                    System.arraycopy(currentRow, 0, this.elements, i, idx);
                    i += idx;
                }
            }
        } else {
            final int zero = (int) index(0, 0);
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, rows);
                Future<?>[] futures = new Future[nthreads];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int idx = zero + firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                float[] currentRow = values[r];
                                if (currentRow.length != 2 * columns)
                                    throw new IllegalArgumentException(
                                            "Must have same number of columns in every row: columns="
                                                    + currentRow.length + "2*columns()=" + 2 * columns());
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = currentRow[2 * c];
                                    elements[i + 1] = currentRow[2 * c + 1];
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = zero;
                for (int r = 0; r < rows; r++) {
                    float[] currentRow = values[r];
                    if (currentRow.length != 2 * columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "2*columns()=" + 2 * columns());
                    for (int i = idx, c = 0; c < columns; c++) {
                        elements[i] = currentRow[2 * c];
                        elements[i + 1] = currentRow[2 * c + 1];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FComplexMatrix2D assignImaginary(final FloatMatrix2D other) {
        checkShape(other);
        final int columnStrideOther = other.columnStride();
        final int rowStrideOther = other.rowStride();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final float[] elemsOther = ((DenseFloatMatrix2D) other).elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elements[i + 1] = elemsOther[j];
                                i += columnStride;
                                j += columnStrideOther;
                            }
                            idx += rowStride;
                            idxOther += rowStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                    elements[i + 1] = elemsOther[j];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return this;
    }

    public FComplexMatrix2D assignReal(final FloatMatrix2D other) {
        checkShape(other);
        final int columnStrideOther = other.columnStride();
        final int rowStrideOther = other.rowStride();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final float[] elemsOther = ((DenseFloatMatrix2D) other).elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elements[i] = elemsOther[j];
                                i += columnStride;
                                j += columnStrideOther;
                            }
                            idx += rowStride;
                            idxOther += rowStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                    elements[i] = elemsOther[j];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return this;
    }

    public int cardinality() {
        int cardinality = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                if ((elements[i] != 0.0) || (elements[i + 1] != 0.0))
                                    cardinality++;
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        return Integer.valueOf(cardinality);
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Integer) futures[j].get();
                }
                cardinality = results[0].intValue();
                for (int j = 1; j < nthreads; j++) {
                    cardinality += results[j].intValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    if ((elements[i] != 0.0) || (elements[i + 1] != 0.0))
                        cardinality++;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return cardinality;
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of this matrix.
     */
    public void fft2() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        if (isNoView == true) {
            fft2.complexForward(elements);
        } else {
            FComplexMatrix2D copy = this.copy();
            fft2.complexForward((float[]) copy.elements());
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the discrete Fourier transform (DFT) of each column of this
     * matrix.
     */
    public void fftColumns() {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFComplexMatrix1D) viewColumn(c)).fft();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFComplexMatrix1D) viewColumn(c)).fft();
            }
        }
    }

    /**
     * Computes the discrete Fourier transform (DFT) of each row of this matrix.
     */
    public void fftRows() {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFComplexMatrix1D) viewRow(r)).fft();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFComplexMatrix1D) viewRow(r)).fft();
            }
        }
    }

    public FComplexMatrix2D forEachNonZero(final cern.colt.function.tfcomplex.IntIntFComplexFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        float[] value = new float[2];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                value[0] = elements[i];
                                value[1] = elements[i + 1];
                                if (value[0] != 0 || value[1] != 0) {
                                    float[] v = function.apply(r, c, value);
                                    elements[i] = v[0];
                                    elements[i + 1] = v[1];
                                }
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            float[] value = new float[2];
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    value[0] = elements[i];
                    value[1] = elements[i + 1];
                    if (value[0] != 0 || value[1] != 0) {
                        float[] v = function.apply(r, c, value);
                        elements[i] = v[0];
                        elements[i + 1] = v[1];
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D getConjugateTranspose() {
        FComplexMatrix2D transpose = this.viewDice().copy();
        final float[] elemsOther = ((DenseFComplexMatrix2D) transpose).elements;
        final int zeroOther = (int) transpose.index(0, 0);
        final int columnStrideOther = transpose.columnStride();
        final int rowStrideOther = transpose.rowStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int columnsOther = transpose.columns();
        final int rowsOther = transpose.rows();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rowsOther);
            Future<?>[] futures = new Future[nthreads];
            int k = rowsOther / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rowsOther : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columnsOther; c++) {
                                elemsOther[idxOther + 1] = -elemsOther[idxOther + 1];
                                idxOther += columnStrideOther;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxOther = zeroOther;
            for (int r = 0; r < rowsOther; r++) {
                for (int c = 0; c < columnsOther; c++) {
                    elemsOther[idxOther + 1] = -elemsOther[idxOther + 1];
                    idxOther += columnStrideOther;
                }
            }
        }
        return transpose;
    }

    public float[] elements() {
        return elements;
    }

    public FloatMatrix2D getImaginaryPart() {
        final DenseFloatMatrix2D Im = new DenseFloatMatrix2D(rows, columns);
        final float[] elemsOther = Im.elements();
        final int columnStrideOther = Im.columnStride();
        final int rowStrideOther = Im.rowStride();
        final int zeroOther = (int) Im.index(0, 0);
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elemsOther[j] = elements[i + 1];
                                i += columnStride;
                                j += columnStrideOther;
                            }
                            idx += rowStride;
                            idxOther += rowStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                    elemsOther[j] = elements[i + 1];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return Im;
    }

    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList,
            final ArrayList<float[]> valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                float[] value = new float[2];
                value[0] = elements[i];
                value[1] = elements[i + 1];
                if (value[0] != 0 || value[1] != 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += columnStride;
            }
            idx += rowStride;
        }

    }

    public float[] getQuick(int row, int column) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        return new float[] { elements[idx], elements[idx + 1] };
    }

    public FloatMatrix2D getRealPart() {
        final DenseFloatMatrix2D R = new DenseFloatMatrix2D(rows, columns);
        final float[] elemsOther = R.elements();
        final int columnStrideOther = R.columnStride();
        final int rowStrideOther = R.rowStride();
        final int zeroOther = (int) R.index(0, 0);
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elemsOther[j] = elements[i];
                                i += columnStride;
                                j += columnStrideOther;
                            }
                            idx += rowStride;
                            idxOther += rowStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                    elemsOther[j] = elements[i];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return R;
    }

    /**
     * Computes the 2D inverse of the discrete Fourier transform (IDFT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void ifft2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        if (isNoView == true) {
            fft2.complexInverse(elements, scale);
        } else {
            FComplexMatrix2D copy = this.copy();
            fft2.complexInverse((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete Fourier transform (IDFT) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void ifftColumns(final boolean scale) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFComplexMatrix1D) viewColumn(c)).ifft(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFComplexMatrix1D) viewColumn(c)).ifft(scale);
            }
        }
    }

    /**
     * Computes the inverse of the discrete Fourier transform (IDFT) of each row
     * of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void ifftRows(final boolean scale) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFComplexMatrix1D) viewRow(r)).ifft(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFComplexMatrix1D) viewRow(r)).ifft(scale);
            }
        }
    }

    public FComplexMatrix2D like(int rows, int columns) {
        return new DenseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix1D like1D(int size) {
        return new DenseFComplexMatrix1D(size);
    }

    public void setQuick(int row, int column, float re, float im) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = re;
        elements[idx + 1] = im;
    }

    public void setQuick(int row, int column, float[] value) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = value[0];
        elements[idx + 1] = value[1];
    }

    public float[][] toArray() {
        final float[][] values = new float[rows][2 * columns];
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                values[r][2 * c] = elements[i];
                                values[r][2 * c + 1] = elements[i + 1];
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    values[r][2 * c] = elements[i];
                    values[r][2 * c + 1] = elements[i + 1];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return values;
    }

    public FComplexMatrix1D vectorize() {
        final FComplexMatrix1D v = new DenseFComplexMatrix1D((int) size());
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) v.index(0);
        final int strideOther = v.stride();
        final float[] elemsOther = (float[]) v.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                final int firstIdx = j * k * rows;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = 0;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            idx = zero + c * columnStride;
                            for (int r = 0; r < rows; r++) {
                                elemsOther[idxOther] = elements[idx];
                                elemsOther[idxOther + 1] = elements[idx + 1];
                                idx += rowStride;
                                idxOther += strideOther;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = 0;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                idx = zero + c * columnStride;
                for (int r = 0; r < rows; r++) {
                    elemsOther[idxOther] = elements[idx];
                    elemsOther[idxOther + 1] = elements[idx + 1];
                    idx += rowStride;
                    idxOther += strideOther;
                }
            }
        }
        return v;
    }

    public FComplexMatrix1D zMult(final FComplexMatrix1D y, FComplexMatrix1D z, final float[] alpha,
            final float[] beta, boolean transposeA) {
        if (transposeA)
            return getConjugateTranspose().zMult(y, z, alpha, beta, false);
        final FComplexMatrix1D zz;
        if (z == null) {
            zz = new DenseFComplexMatrix1D(this.rows);
        } else {
            zz = z;
        }
        if (columns != y.size() || rows > zz.size())
            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort()
                    + ", " + zz.toStringShort());
        final float[] elemsY = (float[]) y.elements();
        final float[] elemsZ = (float[]) zz.elements();
        if (elements == null || elemsY == null || elemsZ == null)
            throw new InternalError();
        final int strideY = y.stride();
        final int strideZ = zz.stride();
        final int zero = (int) index(0, 0);
        final int zeroY = (int) y.index(0);
        final int zeroZ = (int) zz.index(0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idxZero = zero + firstRow * rowStride;
                        int idxZeroZ = zeroZ + firstRow * strideZ;
                        float reS;
                        float imS;
                        float reA;
                        float imA;
                        float reY;
                        float imY;
                        float reZ;
                        float imZ;
                        for (int r = firstRow; r < lastRow; r++) {
                            reS = 0;
                            imS = 0;
                            int idx = idxZero;
                            int idxY = zeroY;
                            for (int c = 0; c < columns; c++) {
                                reA = elements[idx];
                                imA = elements[idx + 1];
                                reY = elemsY[idxY];
                                imY = elemsY[idxY + 1];
                                reS += reA * reY - imA * imY;
                                imS += imA * reY + reA * imY;
                                idx += columnStride;
                                idxY += strideY;
                            }
                            reZ = elemsZ[idxZeroZ];
                            imZ = elemsZ[idxZeroZ + 1];
                            elemsZ[idxZeroZ] = reS * alpha[0] - imS * alpha[1] + reZ * beta[0] - imZ * beta[1];
                            elemsZ[idxZeroZ + 1] = imS * alpha[0] + reS * alpha[1] + imZ * beta[0] + reZ * beta[1];
                            idxZero += rowStride;
                            idxZeroZ += strideZ;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxZero = zero;
            int idxZeroZ = zeroZ;
            float reS;
            float imS;
            float reA;
            float imA;
            float reY;
            float imY;
            float reZ;
            float imZ;

            for (int r = 0; r < rows; r++) {
                reS = 0;
                imS = 0;
                int idx = idxZero;
                int idxY = zeroY;
                for (int c = 0; c < columns; c++) {
                    reA = elements[idx];
                    imA = elements[idx + 1];
                    reY = elemsY[idxY];
                    imY = elemsY[idxY + 1];
                    reS += reA * reY - imA * imY;
                    imS += imA * reY + reA * imY;
                    idx += columnStride;
                    idxY += strideY;
                }
                reZ = elemsZ[idxZeroZ];
                imZ = elemsZ[idxZeroZ + 1];
                elemsZ[idxZeroZ] = reS * alpha[0] - imS * alpha[1] + reZ * beta[0] - imZ * beta[1];
                elemsZ[idxZeroZ + 1] = imS * alpha[0] + reS * alpha[1] + imZ * beta[0] + reZ * beta[1];
                idxZero += rowStride;
                idxZeroZ += strideZ;
            }
        }
        return zz;
    }

    public FComplexMatrix2D zMult(final FComplexMatrix2D B, FComplexMatrix2D C, final float[] alpha,
            final float[] beta, final boolean transposeA, final boolean transposeB) {
        final int rowsA = rows;
        final int columnsA = columns;
        final int rowsB = B.rows();
        final int columnsB = B.columns();
        final int rowsC = transposeA ? columnsA : rowsA;
        final int columnsC = transposeB ? rowsB : columnsB;

        if (C == null)
            C = new DenseFComplexMatrix2D(rowsC, columnsC);

        if (transposeA)
            return getConjugateTranspose().zMult(B, C, alpha, beta, false, transposeB);
        if (transposeB)
            return this.zMult(B.getConjugateTranspose(), C, alpha, beta, transposeA, false);
        if (B.rows() != columnsA)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + B.toStringShort());
        if (C.rows() != rowsA || C.columns() != columnsB)
            throw new IllegalArgumentException("Incompatibe result matrix: " + toStringShort() + ", "
                    + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");
        long flops = 2L * rowsA * columnsA * columnsB;
        int noOfTasks = (int) Math.min(flops / 30000, ConcurrencyUtils.getNumberOfThreads()); // each
        /* thread should process at least 30000 flops */
        boolean splitB = (columnsB >= noOfTasks);
        int width = splitB ? columnsB : rowsA;
        noOfTasks = Math.min(width, noOfTasks);

        if (noOfTasks < 2) {
            return this.zMultSeq(B, C, alpha, beta, transposeA, transposeB);
        }
        // set up concurrent tasks
        int span = width / noOfTasks;
        final Future<?>[] subTasks = new Future[noOfTasks];
        for (int i = 0; i < noOfTasks; i++) {
            final int offset = i * span;
            if (i == noOfTasks - 1)
                span = width - span * i; // last span may be a bit larger
            final FComplexMatrix2D AA, BB, CC;
            if (splitB) {
                // split B along columns into blocks
                AA = this;
                BB = B.viewPart(0, offset, columnsA, span);
                CC = C.viewPart(0, offset, rowsA, span);
            } else {
                // split A along rows into blocks
                AA = this.viewPart(offset, 0, span, columnsA);
                BB = B;
                CC = C.viewPart(offset, 0, span, columnsB);
            }

            subTasks[i] = ConcurrencyUtils.submit(new Runnable() {
                public void run() {
                    ((DenseFComplexMatrix2D) AA).zMultSeq(BB, CC, alpha, beta, transposeA, transposeB);
                }
            });
        }
        ConcurrencyUtils.waitForCompletion(subTasks);

        return C;
    }

    protected FComplexMatrix2D zMultSeq(FComplexMatrix2D B, FComplexMatrix2D C, float[] alpha, float[] beta,
            boolean transposeA, boolean transposeB) {
        if (transposeA)
            return getConjugateTranspose().zMult(B, C, alpha, beta, false, transposeB);
        if (transposeB)
            return this.zMult(B.getConjugateTranspose(), C, alpha, beta, transposeA, false);
        int m = rows;
        int n = columns;
        int p = B.columns();
        if (C == null)
            C = new DenseFComplexMatrix2D(m, p);
        if (!(C instanceof DenseFComplexMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + B.toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", "
                    + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        DenseFComplexMatrix2D BB = (DenseFComplexMatrix2D) B;
        DenseFComplexMatrix2D CC = (DenseFComplexMatrix2D) C;
        final float[] AElems = this.elements;
        final float[] BElems = BB.elements;
        final float[] CElems = CC.elements;
        if (AElems == null || BElems == null || CElems == null)
            throw new InternalError();

        int cA = this.columnStride;
        int cB = BB.columnStride;
        int cC = CC.columnStride;

        int rA = this.rowStride;
        int rB = BB.rowStride;
        int rC = CC.rowStride;

        /*
         * A is blocked to hide memory latency xxxxxxx B xxxxxxx xxxxxxx A xxx
         * xxxxxxx C xxx xxxxxxx --- ------- xxx xxxxxxx xxx xxxxxxx --- -------
         * xxx xxxxxxx
         */
        final int BLOCK_SIZE = 30000; // * 8 == Level 2 cache in bytes
        int m_optimal = (BLOCK_SIZE - n) / (n + 1);
        if (m_optimal <= 0)
            m_optimal = 1;
        int blocks = m / m_optimal;
        int rr = 0;
        if (m % m_optimal != 0)
            blocks++;
        float reS;
        float imS;
        float reA;
        float imA;
        float reB;
        float imB;
        float reC;
        float imC;
        for (; --blocks >= 0;) {
            int jB = (int) BB.index(0, 0);
            int indexA = (int) index(rr, 0);
            int jC = (int) CC.index(rr, 0);
            rr += m_optimal;
            if (blocks == 0)
                m_optimal += m - rr;

            for (int j = p; --j >= 0;) {
                int iA = indexA;
                int iC = jC;
                for (int i = m_optimal; --i >= 0;) {
                    int kA = iA;
                    int kB = jB;
                    reS = 0;
                    imS = 0;
                    // loop unrolled
                    kA -= cA;
                    kB -= rB;
                    for (int k = n % 4; --k >= 0;) {
                        kA += cA;
                        kB += rB;
                        reA = AElems[kA];
                        imA = AElems[kA + 1];
                        reB = BElems[kB];
                        imB = BElems[kB + 1];
                        reS += reA * reB - imA * imB;
                        imS += imA * reB + reA * imB;
                    }
                    for (int k = n / 4; --k >= 0;) {
                        kA += cA;
                        kB += rB;
                        reA = AElems[kA];
                        imA = AElems[kA + 1];
                        reB = BElems[kB];
                        imB = BElems[kB + 1];
                        reS += reA * reB - imA * imB;
                        imS += imA * reB + reA * imB;
                        kA += cA;
                        kB += rB;
                        reA = AElems[kA];
                        imA = AElems[kA + 1];
                        reB = BElems[kB];
                        imB = BElems[kB + 1];
                        reS += reA * reB - imA * imB;
                        imS += imA * reB + reA * imB;
                        kA += cA;
                        kB += rB;
                        reA = AElems[kA];
                        imA = AElems[kA + 1];
                        reB = BElems[kB];
                        imB = BElems[kB + 1];
                        reS += reA * reB - imA * imB;
                        imS += imA * reB + reA * imB;
                        kA += cA;
                        kB += rB;
                        reA = AElems[kA];
                        imA = AElems[kA + 1];
                        reB = BElems[kB];
                        imB = BElems[kB + 1];
                        reS += reA * reB - imA * imB;
                        imS += imA * reB + reA * imB;
                    }
                    reC = CElems[iC];
                    imC = CElems[iC + 1];
                    CElems[iC] = alpha[0] * reS - alpha[1] * imS + beta[0] * reC - beta[1] * imC;
                    CElems[iC + 1] = alpha[1] * reS + alpha[0] * imS + beta[1] * reC + beta[0] * imC;
                    iA += rA;
                    iC += rC;
                }
                jB += cB;
                jC += cC;
            }
        }
        return C;
    }

    public float[] zSum() {
        float[] sum = new float[2];
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {

                    public float[] call() throws Exception {
                        float[] sum = new float[2];
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                sum[0] += elements[i];
                                sum[1] += elements[i + 1];
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        return sum;
                    }
                });
            }
            try {
                float[] tmp;
                for (int j = 0; j < nthreads; j++) {
                    tmp = (float[]) futures[j].get();
                    sum[0] = sum[0] + tmp[0];
                    sum[1] = sum[1] + tmp[1];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    sum[0] += elements[i];
                    sum[1] += elements[i + 1];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return sum;
    }

    protected boolean haveSharedCellsRaw(FComplexMatrix2D other) {
        if (other instanceof SelectedDenseFComplexMatrix2D) {
            SelectedDenseFComplexMatrix2D otherMatrix = (SelectedDenseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFComplexMatrix2D) {
            DenseFComplexMatrix2D otherMatrix = (DenseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    protected FComplexMatrix1D like1D(int size, int zero, int stride) {
        return new DenseFComplexMatrix1D(size, this.elements, zero, stride, false);
    }

    protected FComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseFComplexMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }
}
