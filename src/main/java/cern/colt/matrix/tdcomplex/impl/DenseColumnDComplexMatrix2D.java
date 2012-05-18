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
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>complex</tt> elements. <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array, addressed in row
 * major. Complex data is represented by 2 double values in sequence, i.e.
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
public class DenseColumnDComplexMatrix2D extends DComplexMatrix2D {
    static final long serialVersionUID = 1020177651L;

    private DoubleFFT_2D fft2;

    /**
     * The elements of this matrix. elements are stored in row major. Complex
     * data is represented by 2 double values in sequence, i.e. elements[idx]
     * constitute the real part and elements[idx+1] constitute the imaginary
     * part, where idx = index(0,0) + row * rowStride + column * columnStride.
     */
    protected double[] elements;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form
     * <tt>re = values[row][2*column]; im = values[row][2*column+1]</tt> and
     * have exactly the same number of rows and columns as the receiver. Due to
     * the fact that complex data is represented by 2 double values in sequence:
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
    public DenseColumnDComplexMatrix2D(double[][] values) {
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
    public DenseColumnDComplexMatrix2D(DoubleMatrix2D realPart) {
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
    public DenseColumnDComplexMatrix2D(int rows, int columns) {
        setUp(rows, columns, 0, 0, 2, 2 * rows);
        this.elements = new double[rows * 2 * columns];
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
    public DenseColumnDComplexMatrix2D(int rows, int columns, double[] elements, int rowZero, int columnZero,
            int rowStride, int columnStride, boolean isNoView) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = isNoView;
    }

    public double[] aggregate(final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction aggr,
            final cern.colt.function.tdcomplex.DComplexDComplexFunction f) {
        double[] b = new double[2];
        if (size() == 0) {
            b[0] = Double.NaN;
            b[1] = Double.NaN;
            return b;
        }
        final int zero = (int) index(0, 0);
        double[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int idx = zero + firstColumn * columnStride;
                        double[] a = f.apply(elements[idx], elements[idx + 1]);
                        int d = 1;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = d; r < rows; r++) {
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
            for (int c = 0; c < columns; c++) {
                for (int r = d; r < rows; r++) {
                    idx = zero + r * rowStride + c * columnStride;
                    a = aggr.apply(a, f.apply(elements[idx], elements[idx + 1]));
                }
                d = 0;
            }
        }
        return a;
    }

    public double[] aggregate(final DComplexMatrix2D other,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction aggr,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction f) {
        if (!(other instanceof DenseColumnDComplexMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        double[] b = new double[2];
        if (size() == 0) {
            b[0] = Double.NaN;
            b[1] = Double.NaN;
            return b;
        }
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int columnStrideOther = other.columnStride();
        final double[] elemsOther = (double[]) other.elements();
        double[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {

                    public double[] call() throws Exception {
                        int idx = zero + firstColumn * columnStride;
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        double[] a = f.apply(new double[] { elements[idx], elements[idx + 1] }, new double[] {
                                elemsOther[idxOther], elemsOther[idxOther + 1] });
                        int d = 1;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = d; r < rows; r++) {
                                idx = zero + r * rowStride + c * columnStride;
                                idxOther = zeroOther + r * rowStrideOther + c * columnStrideOther;
                                a = aggr.apply(a, f.apply(new double[] { elements[idx], elements[idx + 1] },
                                        new double[] { elemsOther[idxOther], elemsOther[idxOther + 1] }));
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
            a = f.apply(new double[] { elements[zero], elements[zero + 1] }, new double[] { elemsOther[zeroOther],
                    elemsOther[zeroOther + 1] });
            int d = 1; // first cell already done
            for (int c = 0; c < columns; c++) {
                for (int r = d; r < rows; r++) {
                    idx = zero + r * rowStride + c * columnStride;
                    idxOther = zeroOther + r * rowStrideOther + c * columnStrideOther;
                    a = aggr.apply(a, f.apply(new double[] { elements[idx], elements[idx + 1] }, new double[] {
                            elemsOther[idxOther], elemsOther[idxOther + 1] }));
                }
                d = 0;
            }
        }
        return a;
    }

    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexDComplexFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tdcomplex.DComplexMult) {
                double[] multiplicator = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
                if (multiplicator[0] == 1 && multiplicator[1] == 0)
                    return this;
                if (multiplicator[0] == 0 && multiplicator[1] == 0)
                    return assign(0, 0);
            }
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        double[] tmp = new double[2];
                        if (function instanceof cern.jet.math.tdcomplex.DComplexMult) {
                            double[] multiplicator = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
                            // x[i] = mult*x[i]
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, r = 0; r < rows; r++) {
                                    tmp[0] = elements[i];
                                    tmp[1] = elements[i + 1];
                                    elements[i] = tmp[0] * multiplicator[0] - tmp[1] * multiplicator[1];
                                    elements[i + 1] = tmp[1] * multiplicator[0] + tmp[0] * multiplicator[1];
                                    i += rowStride;
                                }
                                idx += columnStride;
                            }
                        } else {
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, r = 0; r < rows; r++) {
                                    tmp = function.apply(elements[i], elements[i + 1]);
                                    elements[i] = tmp[0];
                                    elements[i + 1] = tmp[1];
                                    i += rowStride;
                                }
                                idx += columnStride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            double[] tmp = new double[2];
            if (function instanceof cern.jet.math.tdcomplex.DComplexMult) {
                double[] multiplicator = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
                // x[i] = mult*x[i]
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, r = 0; r < rows; r++) {
                        tmp[0] = elements[i];
                        tmp[1] = elements[i + 1];
                        elements[i] = tmp[0] * multiplicator[0] - tmp[1] * multiplicator[1];
                        elements[i + 1] = tmp[1] * multiplicator[0] + tmp[0] * multiplicator[1];
                        i += rowStride;
                    }
                    idx += columnStride;
                }
            } else {
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, r = 0; r < rows; r++) {
                        tmp = function.apply(elements[i], elements[i + 1]);
                        elements[i] = tmp[0];
                        elements[i + 1] = tmp[1];
                        i += rowStride;
                    }
                    idx += columnStride;
                }
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond,
            final cern.colt.function.tdcomplex.DComplexDComplexFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        double[] elem = new double[2];
                        int idx = zero + firstColumn * columnStride;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                elem[0] = elements[i];
                                elem[1] = elements[i + 1];
                                if (cond.apply(elem) == true) {
                                    elem = function.apply(elem);
                                    elements[i] = elem[0];
                                    elements[i + 1] = elem[1];
                                }
                                i += rowStride;
                            }
                            idx += columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] elem = new double[2];
            int idx = zero;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    elem[0] = elements[i];
                    elem[1] = elements[i + 1];
                    if (cond.apply(elem) == true) {
                        elem = function.apply(elem);
                        elements[i] = elem[0];
                        elements[i + 1] = elem[1];
                    }
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond, final double[] value) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        double[] elem = new double[2];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                elem[0] = elements[i];
                                elem[1] = elements[i + 1];
                                if (cond.apply(elem) == true) {
                                    elements[i] = value[0];
                                    elements[i + 1] = value[1];
                                }
                                i += rowStride;
                            }
                            idx += columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] elem = new double[2];
            int idx = zero;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    elem[0] = elements[i];
                    elem[1] = elements[i + 1];
                    if (cond.apply(elem) == true) {
                        elements[i] = value[0];
                        elements[i + 1] = value[1];
                    }
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexRealFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        double[] tmp = new double[2];
                        if (function == cern.jet.math.tdcomplex.DComplexFunctions.abs) {
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, r = 0; r < rows; r++) {
                                    tmp[0] = elements[i];
                                    tmp[1] = elements[i + 1];
                                    double absX = Math.abs(elements[i]);
                                    double absY = Math.abs(elements[i + 1]);
                                    if (absX == 0 && absY == 0) {
                                        elements[i] = 0;
                                    } else if (absX >= absY) {
                                        double d = tmp[1] / tmp[0];
                                        elements[i] = absX * Math.sqrt(1 + d * d);
                                    } else {
                                        double d = tmp[0] / tmp[1];
                                        elements[i] = absY * Math.sqrt(1 + d * d);
                                    }
                                    elements[i + 1] = 0;
                                    i += rowStride;
                                }
                                idx += columnStride;
                            }
                        } else {
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, r = 0; r < rows; r++) {
                                    tmp[0] = elements[i];
                                    tmp[1] = elements[i + 1];
                                    tmp[0] = function.apply(tmp);
                                    elements[i] = tmp[0];
                                    elements[i + 1] = 0;
                                    i += rowStride;
                                }
                                idx += columnStride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            double[] tmp = new double[2];
            if (function == cern.jet.math.tdcomplex.DComplexFunctions.abs) {
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, r = 0; r < rows; r++) {
                        tmp[0] = elements[i];
                        tmp[1] = elements[i + 1];
                        double absX = Math.abs(tmp[0]);
                        double absY = Math.abs(tmp[1]);
                        if (absX == 0 && absY == 0) {
                            elements[i] = 0;
                        } else if (absX >= absY) {
                            double d = tmp[1] / tmp[0];
                            elements[i] = absX * Math.sqrt(1 + d * d);
                        } else {
                            double d = tmp[0] / tmp[1];
                            elements[i] = absY * Math.sqrt(1 + d * d);
                        }
                        elements[i + 1] = 0;
                        i += rowStride;
                    }
                    idx += columnStride;
                }
            } else {
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, r = 0; r < rows; r++) {
                        tmp[0] = elements[i];
                        tmp[1] = elements[i + 1];
                        tmp[0] = function.apply(tmp);
                        elements[i] = tmp[0];
                        elements[i + 1] = 0;
                        i += rowStride;
                    }
                    idx += columnStride;
                }
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final DComplexMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof DenseColumnDComplexMatrix2D)) {
            super.assign(source);
            return this;
        }
        DenseColumnDComplexMatrix2D other = (DenseColumnDComplexMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            DComplexMatrix2D c = other.copy();
            if (!(c instanceof DenseColumnDComplexMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseColumnDComplexMatrix2D) c;
        }

        final double[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                elements[i] = elemsOther[j];
                                elements[i + 1] = elemsOther[j + 1];
                                i += rowStride;
                                j += rowStrideOther;
                            }
                            idx += columnStride;
                            idxOther += columnStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                    elements[i] = elemsOther[j];
                    elements[i + 1] = elemsOther[j + 1];
                    i += rowStride;
                    j += rowStrideOther;
                }
                idx += columnStride;
                idxOther += columnStrideOther;
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final DComplexMatrix2D y,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseColumnDComplexMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final double[] elemsOther = ((DenseColumnDComplexMatrix2D) y).elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int columnStrideOther = y.columnStride();
        final int rowStrideOther = y.rowStride();
        final int zeroOther = (int) y.index(0, 0);
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        double[] tmp1 = new double[2];
                        double[] tmp2 = new double[2];
                        if (function == cern.jet.math.tdcomplex.DComplexFunctions.mult) {
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    elements[i] = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1];
                                    elements[i + 1] = tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                                    i += rowStride;
                                    j += rowStrideOther;
                                }
                                idx += columnStride;
                                idxOther += columnStrideOther;
                            }
                        } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjFirst) {
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                                    elements[i + 1] = -tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                                    i += rowStride;
                                    j += rowStrideOther;
                                }
                                idx += columnStride;
                                idxOther += columnStrideOther;
                            }

                        } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjSecond) {
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                                    elements[i + 1] = tmp1[1] * tmp2[0] - tmp1[0] * tmp2[1];
                                    i += rowStride;
                                    j += rowStrideOther;
                                }
                                idx += columnStride;
                                idxOther += columnStrideOther;
                            }
                        } else {
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                    tmp1[0] = elements[i];
                                    tmp1[1] = elements[i + 1];
                                    tmp2[0] = elemsOther[j];
                                    tmp2[1] = elemsOther[j + 1];
                                    tmp1 = function.apply(tmp1, tmp2);
                                    elements[i] = tmp1[0];
                                    elements[i + 1] = tmp1[1];
                                    i += rowStride;
                                    j += rowStrideOther;
                                }
                                idx += columnStride;
                                idxOther += columnStrideOther;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] tmp1 = new double[2];
            double[] tmp2 = new double[2];
            int idx = zero;
            int idxOther = zeroOther;
            if (function == cern.jet.math.tdcomplex.DComplexFunctions.mult) {
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        elements[i] = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1];
                        elements[i + 1] = tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                        i += rowStride;
                        j += rowStrideOther;
                    }
                    idx += columnStride;
                    idxOther += columnStrideOther;
                }
            } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjFirst) {
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                        elements[i + 1] = -tmp1[1] * tmp2[0] + tmp1[0] * tmp2[1];
                        i += rowStride;
                        j += rowStrideOther;
                    }
                    idx += columnStride;
                    idxOther += columnStrideOther;
                }

            } else if (function == cern.jet.math.tdcomplex.DComplexFunctions.multConjSecond) {
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        elements[i] = tmp1[0] * tmp2[0] + tmp1[1] * tmp2[1];
                        elements[i + 1] = tmp1[1] * tmp2[0] - tmp1[0] * tmp2[1];
                        i += rowStride;
                        j += rowStrideOther;
                    }
                    idx += columnStride;
                    idxOther += columnStrideOther;
                }
            } else {
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                        tmp1[0] = elements[i];
                        tmp1[1] = elements[i + 1];
                        tmp2[0] = elemsOther[j];
                        tmp2[1] = elemsOther[j + 1];
                        tmp1 = function.apply(tmp1, tmp2);
                        elements[i] = tmp1[0];
                        elements[i + 1] = tmp1[1];
                        i += rowStride;
                        j += rowStrideOther;
                    }
                    idx += columnStride;
                    idxOther += columnStrideOther;
                }
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final double re, final double im) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                elements[i] = re;
                                elements[i + 1] = im;
                                i += rowStride;
                            }
                            idx += columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    elements[i] = re;
                    elements[i + 1] = im;
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final double[] values) {
        if (values.length != rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + "rows()*2*columns()="
                    + rows() * 2 * columns());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            final int zero = (int) index(0, 0);
            if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, columns);
                Future<?>[] futures = new Future[nthreads];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = j * k;
                    final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int idxOther = firstColumn * rows * 2;
                            int idx = zero + firstColumn * columnStride;
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int i = idx, r = 0; r < rows; r++) {
                                    elements[i] = values[idxOther++];
                                    elements[i + 1] = values[idxOther++];
                                    i += rowStride;
                                }
                                idx += columnStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idxOther = 0;
                int idx = zero;
                for (int c = 0; c < columns; c++) {
                    for (int i = idx, r = 0; r < rows; r++) {
                        elements[i] = values[idxOther++];
                        elements[i + 1] = values[idxOther++];
                        i += rowStride;
                    }
                    idx += columnStride;
                }
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final double[][] values) {
        if (values.length != columns)
            throw new IllegalArgumentException("Must have same number of columns: values.length=" + values.length
                    + "columns()=" + columns());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, columns);
                Future<?>[] futures = new Future[nthreads];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = j * k;
                    final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int idx = 2 * rows;
                            int i = firstColumn * columnStride;
                            for (int c = firstColumn; c < lastColumn; c++) {
                                double[] currentColumn = values[c];
                                if (currentColumn.length != idx)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every column: rows="
                                                    + currentColumn.length + "2*rows()=" + idx);
                                System.arraycopy(currentColumn, 0, elements, i, idx);
                                i += idx;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = 2 * rows;
                int i = 0;
                for (int c = 0; c < columns; c++) {
                    double[] currentColumn = values[c];
                    if (currentColumn.length != idx)
                        throw new IllegalArgumentException("Must have same number of rows in every column: rows="
                                + currentColumn.length + "2*rows()=" + idx);
                    System.arraycopy(currentColumn, 0, this.elements, i, idx);
                    i += idx;
                }
            }
        } else {
            final int zero = (int) index(0, 0);
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, columns);
                Future<?>[] futures = new Future[nthreads];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = j * k;
                    final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int idx = zero + firstColumn * columnStride;
                            for (int c = firstColumn; c < lastColumn; c++) {
                                double[] currentColumn = values[c];
                                if (currentColumn.length != 2 * rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every column: rows="
                                                    + currentColumn.length + "2*rows()=" + idx);
                                for (int i = idx, r = 0; r < rows; r++) {
                                    elements[i] = currentColumn[2 * r];
                                    elements[i + 1] = currentColumn[2 * r + 1];
                                    i += rowStride;
                                }
                                idx += columnStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = zero;
                for (int c = 0; c < columns; c++) {
                    double[] currentColumn = values[c];
                    if (currentColumn.length != 2 * rows)
                        throw new IllegalArgumentException("Must have same number of rows in every column: rows="
                                + currentColumn.length + "2*rows()=" + idx);
                    for (int i = idx, r = 0; r < rows; r++) {
                        elements[i] = currentColumn[2 * r];
                        elements[i + 1] = currentColumn[2 * r + 1];
                        i += rowStride;
                    }
                    idx += columnStride;
                }
            }
        }
        return this;
    }

    public DComplexMatrix2D assign(final float[] values) {
        if (values.length != rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + "rows()*2*columns()="
                    + rows() * 2 * columns());
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idxOther = firstColumn * rows * 2;
                        int idx = zero + firstColumn * columnStride;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                elements[i] = values[idxOther++];
                                elements[i + 1] = values[idxOther++];
                                i += rowStride;
                            }
                            idx += columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxOther = 0;
            int idx = zero;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    elements[i] = values[idxOther++];
                    elements[i + 1] = values[idxOther++];
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return this;
    }

    public DComplexMatrix2D assignImaginary(final DoubleMatrix2D other) {
        checkShape(other);
        final int columnStrideOther = other.columnStride();
        final int rowStrideOther = other.rowStride();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final double[] elemsOther = ((DenseDoubleMatrix2D) other).elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                elements[i + 1] = elemsOther[j];
                                i += rowStride;
                                j += rowStrideOther;
                            }
                            idx += columnStride;
                            idxOther += columnStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                    elements[i + 1] = elemsOther[j];
                    i += rowStride;
                    j += rowStrideOther;
                }
                idx += columnStride;
                idxOther += columnStrideOther;
            }
        }
        return this;
    }

    public DComplexMatrix2D assignReal(final DoubleMatrix2D other) {
        checkShape(other);
        final int columnStrideOther = other.columnStride();
        final int rowStrideOther = other.rowStride();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final double[] elemsOther = ((DenseDoubleMatrix2D) other).elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                elements[i] = elemsOther[j];
                                i += rowStride;
                                j += rowStrideOther;
                            }
                            idx += columnStride;
                            idxOther += columnStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                    elements[i] = elemsOther[j];
                    i += rowStride;
                    j += rowStrideOther;
                }
                idx += columnStride;
                idxOther += columnStrideOther;
            }
        }
        return this;
    }

    public int cardinality() {
        int cardinality = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        int idx = zero + firstColumn * columnStride;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                if ((elements[i] != 0.0) || (elements[i + 1] != 0.0))
                                    cardinality++;
                                i += rowStride;
                            }
                            idx += columnStride;
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
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    if ((elements[i] != 0.0) || (elements[i + 1] != 0.0))
                        cardinality++;
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return cardinality;
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of this matrix.
     */
    public void fft2() {
        DComplexMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new DoubleFFT_2D(rows, columns);
        }
        fft2.complexForward((double[]) transpose.elements());
        this.assign(transpose.viewDice().copy());
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
                            ((DenseDComplexMatrix1D) viewColumn(c)).fft();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseDComplexMatrix1D) viewColumn(c)).fft();
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
                            ((DenseDComplexMatrix1D) viewRow(r)).fft();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseDComplexMatrix1D) viewRow(r)).fft();
            }
        }
    }

    public DComplexMatrix2D forEachNonZero(final cern.colt.function.tdcomplex.IntIntDComplexFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        double[] value = new double[2];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                value[0] = elements[i];
                                value[1] = elements[i + 1];
                                if (value[0] != 0 || value[1] != 0) {
                                    double[] v = function.apply(r, c, value);
                                    elements[i] = v[0];
                                    elements[i + 1] = v[1];
                                }
                                i += rowStride;
                            }
                            idx += columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            double[] value = new double[2];
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    value[0] = elements[i];
                    value[1] = elements[i + 1];
                    if (value[0] != 0 || value[1] != 0) {
                        double[] v = function.apply(r, c, value);
                        elements[i] = v[0];
                        elements[i + 1] = v[1];
                    }
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return this;
    }

    public DComplexMatrix2D getConjugateTranspose() {
        DComplexMatrix2D transpose = this.viewDice().copy();
        final double[] elemsOther = ((DenseColumnDComplexMatrix2D) transpose).elements;
        final int zeroOther = (int) transpose.index(0, 0);
        final int columnStrideOther = transpose.columnStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int columnsOther = transpose.columns();
        final int rowsOther = transpose.rows();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columnsOther);
            Future<?>[] futures = new Future[nthreads];
            int k = columnsOther / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columnsOther : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = 0; r < rowsOther; r++) {
                                elemsOther[idxOther + 1] = -elemsOther[idxOther + 1];
                                idxOther += 2;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxOther = zeroOther;
            for (int c = 0; c < columnsOther; c++) {
                for (int r = 0; r < rowsOther; r++) {
                    elemsOther[idxOther + 1] = -elemsOther[idxOther + 1];
                    idxOther += 2;
                }
            }
        }
        return transpose;
    }

    public double[] elements() {
        return elements;
    }

    public DoubleMatrix2D getImaginaryPart() {
        final DenseColumnDoubleMatrix2D Im = new DenseColumnDoubleMatrix2D(rows, columns);
        final double[] elemsOther = Im.elements();
        final int columnStrideOther = Im.columnStride();
        final int rowStrideOther = Im.rowStride();
        final int zeroOther = (int) Im.index(0, 0);
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                elemsOther[j] = elements[i + 1];
                                i += rowStride;
                                j += rowStrideOther;
                            }
                            idx += columnStride;
                            idxOther += columnStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                    elemsOther[j] = elements[i + 1];
                    i += rowStride;
                    j += rowStrideOther;
                }
                idx += columnStride;
                idxOther += columnStrideOther;
            }
        }
        return Im;
    }

    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList,
            final ArrayList<double[]> valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int c = 0; c < columns; c++) {
            for (int i = idx, r = 0; r < rows; r++) {
                double[] value = new double[2];
                value[0] = elements[i];
                value[1] = elements[i + 1];
                if (value[0] != 0 || value[1] != 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += rowStride;
            }
            idx += columnStride;
        }

    }

    public double[] getQuick(int row, int column) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        return new double[] { elements[idx], elements[idx + 1] };
    }

    public DoubleMatrix2D getRealPart() {
        final DenseColumnDoubleMatrix2D R = new DenseColumnDoubleMatrix2D(rows, columns);
        final double[] elemsOther = R.elements();
        final int columnStrideOther = R.columnStride();
        final int rowStrideOther = R.rowStride();
        final int zeroOther = (int) R.index(0, 0);
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        int idxOther = zeroOther + firstColumn * columnStrideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                                elemsOther[j] = elements[i];
                                i += rowStride;
                                j += rowStrideOther;
                            }
                            idx += columnStride;
                            idxOther += columnStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, j = idxOther, r = 0; r < rows; r++) {
                    elemsOther[j] = elements[i];
                    i += rowStride;
                    j += rowStrideOther;
                }
                idx += columnStride;
                idxOther += columnStrideOther;
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
        DComplexMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new DoubleFFT_2D(rows, columns);
        }
        fft2.complexInverse((double[]) transpose.elements(), scale);
        this.assign(transpose.viewDice().copy());
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
                            ((DenseDComplexMatrix1D) viewColumn(c)).ifft(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseDComplexMatrix1D) viewColumn(c)).ifft(scale);
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
                            ((DenseDComplexMatrix1D) viewRow(r)).ifft(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseDComplexMatrix1D) viewRow(r)).ifft(scale);
            }
        }
    }

    public DComplexMatrix2D like(int rows, int columns) {
        return new DenseColumnDComplexMatrix2D(rows, columns);
    }

    public DComplexMatrix1D like1D(int size) {
        return new DenseDComplexMatrix1D(size);
    }

    public void setQuick(int row, int column, double re, double im) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = re;
        elements[idx + 1] = im;
    }

    public void setQuick(int row, int column, double[] value) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = value[0];
        elements[idx + 1] = value[1];
    }

    public double[][] toArray() {
        final double[][] values = new double[rows][2 * columns];
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstColumn * columnStride;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                values[r][2 * c] = elements[i];
                                values[r][2 * c + 1] = elements[i + 1];
                                i += rowStride;
                            }
                            idx += columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    values[r][2 * c] = elements[i];
                    values[r][2 * c + 1] = elements[i + 1];
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return values;
    }

    public DComplexMatrix1D vectorize() {
        final DComplexMatrix1D v = new DenseDComplexMatrix1D((int) size());
        if (isNoView == true) {
            System.arraycopy(elements, 0, v.elements(), 0, elements.length);
        } else {
            final int zero = (int) index(0, 0);
            final int zeroOther = (int) v.index(0);
            final int strideOther = v.stride();
            final double[] elemsOther = (double[]) v.elements();
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
        }
        return v;
    }

    //    public DComplexMatrix1D zMult(final DComplexMatrix1D y, DComplexMatrix1D z, final double[] alpha,
    //            final double[] beta, boolean transposeA) {
    //        if (transposeA)
    //            return getConjugateTranspose().zMult(y, z, alpha, beta, false);
    //        final DComplexMatrix1D zz;
    //        if (z == null) {
    //            zz = new DenseDComplexMatrix1D(this.rows);
    //        } else {
    //            zz = z;
    //        }
    //        if (columns != y.size() || rows > zz.size())
    //            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort()
    //                    + ", " + zz.toStringShort());
    //        final double[] elemsY = (double[]) y.elements();
    //        final double[] elemsZ = (double[]) zz.elements();
    //        if (elements == null || elemsY == null || elemsZ == null)
    //            throw new InternalError();
    //        final int strideY = y.stride();
    //        final int strideZ = zz.stride();
    //        final int zero = (int) index(0, 0);
    //        final int zeroY = (int) y.index(0);
    //        final int zeroZ = (int) zz.index(0);
    //        int nthreads = ConcurrencyUtils.getNumberOfThreads();
    //        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
    //            nthreads = Math.min(nthreads, columns);
    //            Future<?>[] futures = new Future[nthreads];
    //            int k = columns / nthreads;
    //            for (int j = 0; j < nthreads; j++) {
    //                final int firstColumn = j * k;
    //                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
    //                futures[j] = ConcurrencyUtils.submit(new Runnable() {
    //
    //                    public void run() {
    //                        int idxZero = zero + firstRow * rowStride;
    //                        int idxZeroZ = zeroZ + firstRow * strideZ;
    //                        double reS;
    //                        double imS;
    //                        double reA;
    //                        double imA;
    //                        double reY;
    //                        double imY;
    //                        double reZ;
    //                        double imZ;
    //                        for (int c = firstColumn; c < lastColumn; c++) {
    //                            reS = 0;
    //                            imS = 0;
    //                            int idx = idxZero;
    //                            int idxY = zeroY;
    //                            for (int c = 0; c < columns; c++) {
    //                                reA = elements[idx];
    //                                imA = elements[idx + 1];
    //                                reY = elemsY[idxY];
    //                                imY = elemsY[idxY + 1];
    //                                reS += reA * reY - imA * imY;
    //                                imS += imA * reY + reA * imY;
    //                                idx += columnStride;
    //                                idxY += strideY;
    //                            }
    //                            reZ = elemsZ[idxZeroZ];
    //                            imZ = elemsZ[idxZeroZ + 1];
    //                            elemsZ[idxZeroZ] = reS * alpha[0] - imS * alpha[1] + reZ * beta[0] - imZ * beta[1];
    //                            elemsZ[idxZeroZ + 1] = imS * alpha[0] + reS * alpha[1] + imZ * beta[0] + reZ * beta[1];
    //                            idxZero += rowStride;
    //                            idxZeroZ += strideZ;
    //                        }
    //                    }
    //                });
    //            }
    //            ConcurrencyUtils.waitForCompletion(futures);
    //        } else {
    //            int idxZero = zero;
    //            int idxZeroZ = zeroZ;
    //            double reS;
    //            double imS;
    //            double reA;
    //            double imA;
    //            double reY;
    //            double imY;
    //            double reZ;
    //            double imZ;
    //
    //            for (int c = 0; c < columns; c++) {
    //                reS = 0;
    //                imS = 0;
    //                int idx = idxZero;
    //                int idxY = zeroY;
    //                for (int c = 0; c < columns; c++) {
    //                    reA = elements[idx];
    //                    imA = elements[idx + 1];
    //                    reY = elemsY[idxY];
    //                    imY = elemsY[idxY + 1];
    //                    reS += reA * reY - imA * imY;
    //                    imS += imA * reY + reA * imY;
    //                    idx += columnStride;
    //                    idxY += strideY;
    //                }
    //                reZ = elemsZ[idxZeroZ];
    //                imZ = elemsZ[idxZeroZ + 1];
    //                elemsZ[idxZeroZ] = reS * alpha[0] - imS * alpha[1] + reZ * beta[0] - imZ * beta[1];
    //                elemsZ[idxZeroZ + 1] = imS * alpha[0] + reS * alpha[1] + imZ * beta[0] + reZ * beta[1];
    //                idxZero += rowStride;
    //                idxZeroZ += strideZ;
    //            }
    //        }
    //        return zz;
    //    }

    public DComplexMatrix2D zMult(final DComplexMatrix2D B, DComplexMatrix2D C, final double[] alpha,
            final double[] beta, final boolean transposeA, final boolean transposeB) {
        final int rowsA = rows;
        final int columnsA = columns;
        final int rowsB = B.rows();
        final int columnsB = B.columns();
        final int rowsC = transposeA ? columnsA : rowsA;
        final int columnsC = transposeB ? rowsB : columnsB;

        if (C == null)
            C = new DenseColumnDComplexMatrix2D(rowsC, columnsC);

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
            final DComplexMatrix2D AA, BB, CC;
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
                    ((DenseColumnDComplexMatrix2D) AA).zMultSeq(BB, CC, alpha, beta, transposeA, transposeB);
                }
            });
        }
        ConcurrencyUtils.waitForCompletion(subTasks);

        return C;
    }

    protected DComplexMatrix2D zMultSeq(DComplexMatrix2D B, DComplexMatrix2D C, double[] alpha, double[] beta,
            boolean transposeA, boolean transposeB) {
        if (transposeA)
            return getConjugateTranspose().zMult(B, C, alpha, beta, false, transposeB);
        if (transposeB)
            return this.zMult(B.getConjugateTranspose(), C, alpha, beta, transposeA, false);
        int m = rows;
        int n = columns;
        int p = B.columns();
        if (C == null)
            C = new DenseColumnDComplexMatrix2D(m, p);
        if (!(C instanceof DenseColumnDComplexMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + B.toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", "
                    + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        DenseColumnDComplexMatrix2D BB = (DenseColumnDComplexMatrix2D) B;
        DenseColumnDComplexMatrix2D CC = (DenseColumnDComplexMatrix2D) C;
        final double[] AElems = this.elements;
        final double[] BElems = BB.elements;
        final double[] CElems = CC.elements;
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
        double reS;
        double imS;
        double reA;
        double imA;
        double reB;
        double imB;
        double reC;
        double imC;
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

    public double[] zSum() {
        double[] sum = new double[2];
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {

                    public double[] call() throws Exception {
                        double[] sum = new double[2];
                        int idx = zero + firstColumn * columnStride;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int i = idx, r = 0; r < rows; r++) {
                                sum[0] += elements[i];
                                sum[1] += elements[i + 1];
                                i += rowStride;
                            }
                            idx += columnStride;
                        }
                        return sum;
                    }
                });
            }
            try {
                double[] tmp;
                for (int j = 0; j < nthreads; j++) {
                    tmp = (double[]) futures[j].get();
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
            for (int c = 0; c < columns; c++) {
                for (int i = idx, r = 0; r < rows; r++) {
                    sum[0] += elements[i];
                    sum[1] += elements[i + 1];
                    i += rowStride;
                }
                idx += columnStride;
            }
        }
        return sum;
    }

    protected boolean haveSharedCellsRaw(DComplexMatrix2D other) {
        if (other instanceof SelectedDenseColumnDComplexMatrix2D) {
            SelectedDenseColumnDComplexMatrix2D otherMatrix = (SelectedDenseColumnDComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseColumnDComplexMatrix2D) {
            DenseColumnDComplexMatrix2D otherMatrix = (DenseColumnDComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    protected DComplexMatrix1D like1D(int size, int zero, int stride) {
        return new DenseDComplexMatrix1D(size, this.elements, zero, stride, false);
    }

    protected DComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseColumnDComplexMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }
}
