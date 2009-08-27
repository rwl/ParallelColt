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
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import edu.emory.mathcs.jtransforms.dct.DoubleDCT_3D;
import edu.emory.mathcs.jtransforms.dht.DoubleDHT_3D;
import edu.emory.mathcs.jtransforms.dst.DoubleDST_3D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 3-d matrix holding <tt>double</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contiguous one-dimensional array, addressed in
 * (in decreasing order of significance): slice major, row major, column major.
 * Note that this implementation is not synchronized.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
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
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DenseDoubleMatrix3D extends DoubleMatrix3D {
    private static final long serialVersionUID = 1L;

    private DoubleFFT_3D fft3;

    private DoubleDCT_3D dct3;

    private DoubleDST_3D dst3;

    private DoubleDHT_3D dht3;

    protected double[] elements;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form <tt>values[slice][row][column]</tt> and have
     * exactly the same number of rows in in every slice and exactly the same
     * number of columns in in every row.
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
    public DenseDoubleMatrix3D(double[][][] values) {
        this(values.length, (values.length == 0 ? 0 : values[0].length), (values.length == 0 ? 0
                : values[0].length == 0 ? 0 : values[0][0].length));
        assign(values);
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
     *             if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseDoubleMatrix3D(int slices, int rows, int columns) {
        setUp(slices, rows, columns);
        this.elements = new double[slices * rows * columns];
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
     * @param isView
     *            if true then a matrix view is constructed
     * @throws IllegalArgumentException
     *             if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseDoubleMatrix3D(int slices, int rows, int columns, double[] elements, int sliceZero, int rowZero,
            int columnZero, int sliceStride, int rowStride, int columnStride, boolean isView) {
        setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    public double aggregate(final cern.colt.function.tdouble.DoubleDoubleFunction aggr,
            final cern.colt.function.tdouble.DoubleFunction f) {
        if (size() == 0)
            return Double.NaN;
        double a = 0;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {

                    public Double call() throws Exception {
                        double a = f.apply(elements[zero + firstSlice * sliceStride]);
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    a = aggr.apply(a, f.apply(elements[zero + s * sliceStride + r * rowStride + c
                                            * columnStride]));
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
            a = f.apply(elements[zero]);
            int d = 1; // first cell already done
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        a = aggr.apply(a, f.apply(elements[zero + s * sliceStride + r * rowStride + c * columnStride]));
                    }
                    d = 0;
                }
            }
        }
        return a;
    }

    public double aggregate(final cern.colt.function.tdouble.DoubleDoubleFunction aggr,
            final cern.colt.function.tdouble.DoubleFunction f, final cern.colt.function.tdouble.DoubleProcedure cond) {
        if (size() == 0)
            return Double.NaN;
        double a = 0;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {

                    public Double call() throws Exception {
                        double elem = elements[zero + firstSlice * sliceStride];
                        double a = 0;
                        if (cond.apply(elem) == true) {
                            a = aggr.apply(a, f.apply(elem));
                        }
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
                                    if (cond.apply(elem) == true) {
                                        a = aggr.apply(a, f.apply(elem));
                                    }
                                    d = 0;
                                }
                            }
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            double elem = elements[zero];
            if (cond.apply(elem) == true) {
                a = aggr.apply(a, f.apply(elem));
            }
            int d = 1;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
                        if (cond.apply(elem) == true) {
                            a = aggr.apply(a, f.apply(elem));
                        }
                        d = 0;
                    }
                }
            }
        }
        return a;
    }

    public double aggregate(final cern.colt.function.tdouble.DoubleDoubleFunction aggr,
            final cern.colt.function.tdouble.DoubleFunction f, final IntArrayList sliceList,
            final IntArrayList rowList, final IntArrayList columnList) {
        if (size() == 0)
            return Double.NaN;
        if (sliceList.size() == 0 || rowList.size() == 0 || columnList.size() == 0)
            return Double.NaN;
        final int size = sliceList.size();
        final int[] sliceElements = sliceList.elements();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final int zero = (int) index(0, 0, 0);
        double a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {

                    public Double call() throws Exception {
                        double a = f.apply(elements[zero + sliceElements[firstIdx] * sliceStride
                                + rowElements[firstIdx] * rowStride + columnElements[firstIdx] * columnStride]);
                        double elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            elem = elements[zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride
                                    + columnElements[i] * columnStride];
                            a = aggr.apply(a, f.apply(elem));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero + sliceElements[0] * sliceStride + rowElements[0] * rowStride + columnElements[0]
                    * columnStride]);
            double elem;
            for (int i = 1; i < size; i++) {
                elem = elements[zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i]
                        * columnStride];
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    public double aggregate(final DoubleMatrix3D other, final cern.colt.function.tdouble.DoubleDoubleFunction aggr,
            final cern.colt.function.tdouble.DoubleDoubleFunction f) {
        if (!(other instanceof DenseDoubleMatrix3D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            return Double.NaN;
        double a = 0;
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) other.index(0, 0, 0);
        final int sliceStrideOther = other.sliceStride();
        final int rowStrideOther = other.rowStride();
        final int columnStrideOther = other.columnStride();
        final double[] elementsOther = (double[]) other.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {
                    public Double call() throws Exception {
                        int idx = zero + firstSlice * sliceStride;
                        int idxOther = zeroOther + firstSlice * sliceStrideOther;
                        double a = f.apply(elements[idx], elementsOther[idxOther]);
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    idx = zero + s * sliceStride + r * rowStride + c * columnStride;
                                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c
                                            * columnStrideOther;
                                    a = aggr.apply(a, f.apply(elements[idx], elementsOther[idxOther]));
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
            a = f.apply(getQuick(0, 0, 0), other.getQuick(0, 0, 0));
            int d = 1; // first cell already done
            int idx;
            int idxOther;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        idx = zero + s * sliceStride + r * rowStride + c * columnStride;
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c * columnStrideOther;
                        a = aggr.apply(a, f.apply(elements[idx], elementsOther[idxOther]));
                    }
                    d = 0;
                }
            }
        }
        return a;
    }

    public DoubleMatrix3D assign(final cern.colt.function.tdouble.DoubleFunction function) {
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    elements[idx] = function.apply(elements[idx]);
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
                        elements[idx] = function.apply(elements[idx]);
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public DoubleMatrix3D assign(final cern.colt.function.tdouble.DoubleProcedure cond,
            final cern.colt.function.tdouble.DoubleFunction f) {
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
                        double elem;
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    elem = elements[idx];
                                    if (cond.apply(elem) == true) {
                                        elements[idx] = f.apply(elem);
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
            double elem;
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        elem = elements[idx];
                        if (cond.apply(elem) == true) {
                            elements[idx] = f.apply(elem);
                        }
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public DoubleMatrix3D assign(final cern.colt.function.tdouble.DoubleProcedure cond, final double value) {
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
                        double elem;
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    elem = elements[idx];
                                    if (cond.apply(elem) == true) {
                                        elements[idx] = value;
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
            double elem;
            int idx;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        elem = elements[idx];
                        if (cond.apply(elem) == true) {
                            elements[idx] = value;
                        }
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public DoubleMatrix3D assign(final double value) {
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    elements[idx] = value;
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
                        elements[idx] = value;
                        idx += columnStride;
                    }
                }
            }
        }
        return this;
    }

    public DoubleMatrix3D assign(final double[] values) {
        if (values.length != size())
            throw new IllegalArgumentException("Must have same length: length=" + values.length
                    + "slices()*rows()*columns()=" + slices() * rows() * columns());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            System.arraycopy(values, 0, this.elements, 0, values.length);
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
                            int idxOther = firstSlice * rows * columns;
                            int idx;
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    for (int c = 0; c < columns; c++) {
                                        elements[idx] = values[idxOther++];
                                        idx += columnStride;
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        futures[j].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                int idxOther = 0;
                int idx;
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        for (int c = 0; c < columns; c++) {
                            elements[idx] = values[idxOther++];
                            idx += columnStride;
                        }
                    }
                }
            }
        }
        return this;
    }

    public DoubleMatrix3D assign(final double[][][] values) {
        if (values.length != slices)
            throw new IllegalArgumentException("Must have same number of slices: slices=" + values.length + "slices()="
                    + slices());
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
                                double[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    double[] currentRow = currentSlice[r];
                                    if (currentRow.length != columns)
                                        throw new IllegalArgumentException(
                                                "Must have same number of columns in every row: columns="
                                                        + currentRow.length + "columns()=" + columns());
                                    System.arraycopy(currentRow, 0, elements, i, columns);
                                    i += columns;
                                }
                            }
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        futures[j].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                int i = 0;
                for (int s = 0; s < slices; s++) {
                    double[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        double[] currentRow = currentSlice[r];
                        if (currentRow.length != columns)
                            throw new IllegalArgumentException(
                                    "Must have same number of columns in every row: columns=" + currentRow.length
                                            + "columns()=" + columns());
                        System.arraycopy(currentRow, 0, this.elements, i, columns);
                        i += columns;
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
                                double[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    double[] currentRow = currentSlice[r];
                                    if (currentRow.length != columns)
                                        throw new IllegalArgumentException(
                                                "Must have same number of columns in every row: columns="
                                                        + currentRow.length + "columns()=" + columns());
                                    for (int c = 0; c < columns; c++) {
                                        elements[idx] = currentRow[c];
                                        idx += columnStride;
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        futures[j].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            } else {
                int idx;
                for (int s = 0; s < slices; s++) {
                    double[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        double[] currentRow = currentSlice[r];
                        if (currentRow.length != columns)
                            throw new IllegalArgumentException(
                                    "Must have same number of columns in every row: columns=" + currentRow.length
                                            + "columns()=" + columns());
                        for (int c = 0; c < columns; c++) {
                            elements[idx] = currentRow[c];
                            idx += columnStride;
                        }
                    }
                }
            }
        }
        return this;
    }

    public DoubleMatrix3D assign(DoubleMatrix3D source) {
        // overriden for performance only
        if (!(source instanceof DenseDoubleMatrix3D)) {
            super.assign(source);
            return this;
        }
        DenseDoubleMatrix3D other = (DenseDoubleMatrix3D) source;
        if (other == this)
            return this;
        checkShape(other);
        if (haveSharedCells(other)) {
            DoubleMatrix3D c = other.copy();
            if (!(c instanceof DenseDoubleMatrix3D)) { // should not happen
                super.assign(source);
                return this;
            }
            other = (DenseDoubleMatrix3D) c;
        }

        final DenseDoubleMatrix3D other_final = other;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other.isNoView) { // quickest
            System.arraycopy(other_final.elements, 0, this.elements, 0, this.elements.length);
            return this;
        } else {
            final int zero = (int) index(0, 0, 0);
            final int zeroOther = (int) other_final.index(0, 0, 0);
            final int sliceStrideOther = other_final.sliceStride;
            final int rowStrideOther = other_final.rowStride;
            final int columnStrideOther = other_final.columnStride;
            final double[] elementsOther = other_final.elements;
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
                                        elements[idx] = elementsOther[idxOther];
                                        idx += columnStride;
                                        idxOther += columnStrideOther;
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        futures[j].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                int idx;
                int idxOther;
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
                        for (int c = 0; c < columns; c++) {
                            elements[idx] = elementsOther[idxOther];
                            idx += columnStride;
                            idxOther += columnStrideOther;
                        }
                    }
                }
            }
            return this;
        }
    }

    public DoubleMatrix3D assign(final DoubleMatrix3D y, final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        if (!(y instanceof DenseDoubleMatrix3D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) y.index(0, 0, 0);
        final int sliceStrideOther = y.sliceStride();
        final int rowStrideOther = y.rowStride();
        final int columnStrideOther = y.columnStride();
        final double[] elementsOther = (double[]) y.elements();
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
                                    elements[idx] = function.apply(elements[idx], elementsOther[idxOther]);
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
                        elements[idx] = function.apply(elements[idx], elementsOther[idxOther]);
                        idx += columnStride;
                        idxOther += columnStrideOther;
                    }
                }
            }
        }

        return this;
    }

    public DoubleMatrix3D assign(final DoubleMatrix3D y,
            final cern.colt.function.tdouble.DoubleDoubleFunction function, final IntArrayList sliceList,
            final IntArrayList rowList, final IntArrayList columnList) {
        if (!(y instanceof DenseDoubleMatrix3D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) y.index(0, 0, 0);
        final int sliceStrideOther = y.sliceStride();
        final int rowStrideOther = y.rowStride();
        final int columnStrideOther = y.columnStride();
        final double[] elementsOther = (double[]) y.elements();
        int size = sliceList.size();
        final int[] sliceElements = sliceList.elements();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            int idx = zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride
                                    + columnElements[i] * columnStride;
                            int idxOther = zeroOther + sliceElements[i] * sliceStrideOther + rowElements[i]
                                    * rowStrideOther + columnElements[i] * columnStrideOther;
                            elements[idx] = function.apply(elements[idx], elementsOther[idxOther]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                int idx = zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i]
                        * columnStride;
                int idxOther = zeroOther + sliceElements[i] * sliceStrideOther + rowElements[i] * rowStrideOther
                        + columnElements[i] * columnStrideOther;
                elements[idx] = function.apply(elements[idx], elementsOther[idxOther]);
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
                                    if (elements[idx] != 0) {
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
                        if (elements[idx] != 0) {
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
     * Computes the 2D discrete cosine transform (DCT-II) of each slice of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dct2Slices(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseDoubleMatrix2D) viewSlice(s)).dct2(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseDoubleMatrix2D) viewSlice(s)).dct2(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D discrete cosine transform (DCT-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dct3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dct3 == null) {
            dct3 = new DoubleDCT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            dct3.forward(elements, scale);
        } else {
            DoubleMatrix3D copy = this.copy();
            dct3.forward((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D discrete Hartley transform (DHT) of each slice of this
     * matrix.
     * 
     */
    public void dht2Slices() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseDoubleMatrix2D) viewSlice(s)).dht2();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseDoubleMatrix2D) viewSlice(s)).dht2();
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D discrete Hartley transform (DHT) of this matrix.
     * 
     */
    public void dht3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dht3 == null) {
            dht3 = new DoubleDHT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            dht3.forward(elements);
        } else {
            DoubleMatrix3D copy = this.copy();
            dht3.forward((double[]) copy.elements());
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D discrete sine transform (DST-II) of each slice of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dst2Slices(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseDoubleMatrix2D) viewSlice(s)).dst2(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseDoubleMatrix2D) viewSlice(s)).dst2(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D discrete sine transform (DST-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dst3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dst3 == null) {
            dst3 = new DoubleDST_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            dst3.forward(elements, scale);
        } else {
            DoubleMatrix3D copy = this.copy();
            dst3.forward((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public double[] elements() {
        return elements;
    }

    /**
     * Computes the 3D discrete Fourier transform (DFT) of this matrix. The
     * physical layout of the output data is as follows:
     * 
     * <pre>
     * this[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * this[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * this[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * this[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * this[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * this[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * this[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * this[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * this[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * this[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * this[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * this[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * this[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * this[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * this[0][0][0] = Re[0][0][0], 
     * this[0][0][1] = Re[0][0][n3/2], 
     * this[0][n2/2][0] = Re[0][n2/2][0], 
     * this[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * this[n1/2][0][0] = Re[n1/2][0][0], 
     * this[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * this[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * this[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>getFft3</code>. To get back the original
     * data, use <code>ifft3</code>.
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     */
    public void fft3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            fft3.realForward(elements);
        } else {
            DoubleMatrix3D copy = this.copy();
            fft3.realForward((double[]) copy.elements());
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Returns new complex matrix which is the 2D discrete Fourier transform
     * (DFT) of each slice of this matrix.
     * 
     * @return the 2D discrete Fourier transform (DFT) of each slice of this
     *         matrix.
     * 
     */
    public DenseDComplexMatrix3D getFft2Slices() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        final DenseDComplexMatrix3D C = new DenseDComplexMatrix3D(slices, rows, columns);
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            C.viewSlice(s).assign(((DenseDoubleMatrix2D) viewSlice(s)).getFft2());
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                C.viewSlice(s).assign(((DenseDoubleMatrix2D) viewSlice(s)).getFft2());
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the 3D discrete Fourier transform
     * (DFT) of this matrix.
     * 
     * @return the 3D discrete Fourier transform (DFT) of this matrix.
     */
    public DenseDComplexMatrix3D getFft3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        DenseDComplexMatrix3D C = new DenseDComplexMatrix3D(slices, rows, columns);
        final int sliceStride = rows * columns;
        final int rowStride = columns;
        final double[] elems;
        if (isNoView == true) {
            elems = elements;
        } else {
            elems = (double[]) this.copy().elements();
        }
        final double[] cElems = (C).elements();
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
                            for (int r = 0; r < rows; r++) {
                                idx = s * sliceStride + r * rowStride;
                                System.arraycopy(elems, idx, cElems, idx, columns);
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
                    idx = s * sliceStride + r * rowStride;
                    System.arraycopy(elems, idx, cElems, idx, columns);
                }
            }
        }
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.realForwardFull(cElems);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the 2D inverse of the discrete
     * Fourier transform (IDFT) of each slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @return the 2D inverse of the discrete Fourier transform (IDFT) of each
     *         slice of this matrix.
     */
    public DenseDComplexMatrix3D getIfft2Slices(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        final DenseDComplexMatrix3D C = new DenseDComplexMatrix3D(slices, rows, columns);
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            C.viewSlice(s).assign(((DenseDoubleMatrix2D) viewSlice(s)).getIfft2(scale));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                C.viewSlice(s).assign(((DenseDoubleMatrix2D) viewSlice(s)).getIfft2(scale));
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the 3D inverse of the discrete
     * Fourier transform (IDFT) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @return the 3D inverse of the discrete Fourier transform (IDFT) of this
     *         matrix.
     * 
     */
    public DenseDComplexMatrix3D getIfft3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        DenseDComplexMatrix3D C = new DenseDComplexMatrix3D(slices, rows, columns);
        final int sliceStride = rows * columns;
        final int rowStride = columns;
        final double[] cElems = (C).elements();
        final double[] elems;
        if (isNoView == true) {
            elems = elements;
        } else {
            elems = (double[]) this.copy().elements();
        }
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
                            for (int r = 0; r < rows; r++) {
                                idx = s * sliceStride + r * rowStride;
                                System.arraycopy(elems, idx, cElems, idx, columns);
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
                    idx = s * sliceStride + r * rowStride;
                    System.arraycopy(elems, idx, cElems, idx, columns);
                }
            }
        }
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.realInverseFull(cElems, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    public double[] getMaxLocation() {
        final int zero = (int) index(0, 0, 0);
        int slice_loc = 0;
        int row_loc = 0;
        int col_loc = 0;
        double maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            double[][] results = new double[nthreads][2];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int slice_loc = firstSlice;
                        int row_loc = 0;
                        int col_loc = 0;
                        double maxValue = elements[zero + firstSlice * sliceStride];
                        int d = 1;
                        double elem;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
                                    if (maxValue < elem) {
                                        maxValue = elem;
                                        slice_loc = s;
                                        row_loc = r;
                                        col_loc = c;
                                    }
                                }
                                d = 0;
                            }
                        }
                        return new double[] { maxValue, slice_loc, row_loc, col_loc };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (double[]) futures[j].get();
                }
                maxValue = results[0][0];
                slice_loc = (int) results[0][1];
                row_loc = (int) results[0][2];
                col_loc = (int) results[0][3];
                for (int j = 1; j < nthreads; j++) {
                    if (maxValue < results[j][0]) {
                        maxValue = results[j][0];
                        slice_loc = (int) results[j][1];
                        row_loc = (int) results[j][2];
                        col_loc = (int) results[j][3];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            maxValue = elements[zero];
            double elem;
            int d = 1;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
                        if (maxValue < elem) {
                            maxValue = elem;
                            slice_loc = s;
                            row_loc = r;
                            col_loc = c;
                        }
                    }
                    d = 0;
                }
            }
        }
        return new double[] { maxValue, slice_loc, row_loc, col_loc };
    }

    public double[] getMinLocation() {
        final int zero = (int) index(0, 0, 0);
        int slice_loc = 0;
        int row_loc = 0;
        int col_loc = 0;
        double minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            double[][] results = new double[nthreads][2];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int slice_loc = firstSlice;
                        int row_loc = 0;
                        int col_loc = 0;
                        double minValue = elements[zero + slice_loc * sliceStride];
                        int d = 1;
                        double elem;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
                                    if (minValue > elem) {
                                        minValue = elem;
                                        slice_loc = s;
                                        row_loc = r;
                                        col_loc = c;
                                    }
                                }
                                d = 0;
                            }
                        }
                        return new double[] { minValue, slice_loc, row_loc, col_loc };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (double[]) futures[j].get();
                }
                minValue = results[0][0];
                slice_loc = (int) results[0][1];
                row_loc = (int) results[0][2];
                col_loc = (int) results[0][3];
                for (int j = 1; j < nthreads; j++) {
                    if (minValue > results[j][0]) {
                        minValue = results[j][0];
                        slice_loc = (int) results[j][1];
                        row_loc = (int) results[j][2];
                        col_loc = (int) results[j][3];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            minValue = elements[zero];
            double elem;
            int d = 1;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
                        if (minValue > elem) {
                            minValue = elem;
                            slice_loc = s;
                            row_loc = r;
                            col_loc = c;
                        }
                    }
                    d = 0;
                }
            }
        }
        return new double[] { minValue, slice_loc, row_loc, col_loc };
    }

    public void getNegativeValues(final IntArrayList sliceList, final IntArrayList rowList,
            final IntArrayList columnList, final DoubleArrayList valueList) {
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
                    double value = elements[idx];
                    if (value < 0) {
                        sliceList.add(s);
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(value);
                    }
                    idx += columnStride;
                }
            }
        }

    }

    public void getNonZeros(final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList,
            final DoubleArrayList valueList) {
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
                    double value = elements[idx];
                    if (value != 0) {
                        sliceList.add(s);
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(value);
                    }
                    idx += columnStride;
                }
            }

        }
    }

    public void getPositiveValues(final IntArrayList sliceList, final IntArrayList rowList,
            final IntArrayList columnList, final DoubleArrayList valueList) {
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
                    double value = elements[idx];
                    if (value > 0) {
                        sliceList.add(s);
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(value);
                    }
                    idx += columnStride;
                }
            }
        }

    }

    public double getQuick(int slice, int row, int column) {
        return elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column
                * columnStride];
    }

    /**
     * Computes the 2D inverse of the discrete cosine transform (DCT-III) of
     * each slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idct2Slices(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseDoubleMatrix2D) viewSlice(s)).idct2(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseDoubleMatrix2D) viewSlice(s)).idct2(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D inverse of the discrete cosine transform (DCT-III) of
     * this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idct3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dct3 == null) {
            dct3 = new DoubleDCT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            dct3.inverse(elements, scale);
        } else {
            DoubleMatrix3D copy = this.copy();
            dct3.inverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete Hartley transform (IDHT) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idht2Slices(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseDoubleMatrix2D) viewSlice(s)).idht2(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseDoubleMatrix2D) viewSlice(s)).idht2(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D inverse of the discrete Hartley transform (IDHT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idht3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dht3 == null) {
            dht3 = new DoubleDHT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            dht3.inverse(elements, scale);
        } else {
            DoubleMatrix3D copy = this.copy();
            dht3.inverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete sine transform (DST-III) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idst2Slices(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            ((DenseDoubleMatrix2D) viewSlice(s)).idst2(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                ((DenseDoubleMatrix2D) viewSlice(s)).idst2(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D inverse of the discrete sine transform (DST-III) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idst3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dst3 == null) {
            dst3 = new DoubleDST_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            dst3.inverse(elements, scale);
        } else {
            DoubleMatrix3D copy = this.copy();
            dst3.inverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D inverse of the discrete Fourier transform (IDFT) of this
     * matrix. The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * this[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * this[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * this[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * this[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * this[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * this[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * this[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * this[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * this[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * this[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * this[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * this[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * this[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * this[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * this[0][0][0] = Re[0][0][0], 
     * this[0][0][1] = Re[0][0][n3/2], 
     * this[0][n2/2][0] = Re[0][n2/2][0], 
     * this[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * this[n1/2][0][0] = Re[n1/2][0][0], 
     * this[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * this[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * this[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>getIfft3</code>.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     */
    public void ifft3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        if (isNoView == true) {
            fft3.realInverse(elements, scale);
        } else {
            DoubleMatrix3D copy = this.copy();
            fft3.realInverse((double[]) copy.elements(), scale);
            this.assign((double[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public long index(int slice, int row, int column) {
        return sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
    }

    public DoubleMatrix3D like(int slices, int rows, int columns) {
        return new DenseDoubleMatrix3D(slices, rows, columns);
    }

    public DoubleMatrix2D like2D(int rows, int columns) {
        return new DenseDoubleMatrix2D(rows, columns);
    }

    public void setQuick(int slice, int row, int column, double value) {
        elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride] = value;
    }

    public double[][][] toArray() {
        final double[][][] values = new double[slices][rows][columns];
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
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
                            double[][] currentSlice = values[s];
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                double[] currentRow = currentSlice[r];
                                for (int c = 0; c < columns; c++) {
                                    currentRow[c] = elements[idx];
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
                double[][] currentSlice = values[s];
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    double[] currentRow = currentSlice[r];
                    for (int c = 0; c < columns; c++) {
                        currentRow[c] = elements[idx];
                        idx += columnStride;
                    }
                }
            }
        }
        return values;
    }

    public DoubleMatrix1D vectorize() {
        DoubleMatrix1D v = new DenseDoubleMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    public void zAssign27Neighbors(DoubleMatrix3D B, cern.colt.function.tdouble.Double27Function function) {
        // overridden for performance only
        if (!(B instanceof DenseDoubleMatrix3D)) {
            super.zAssign27Neighbors(B, function);
            return;
        }
        if (function == null)
            throw new NullPointerException("function must not be null.");
        checkShape(B);
        int r = rows - 1;
        int c = columns - 1;
        if (rows < 3 || columns < 3 || slices < 3)
            return; // nothing to do

        DenseDoubleMatrix3D BB = (DenseDoubleMatrix3D) B;
        int A_ss = sliceStride;
        int A_rs = rowStride;
        int B_rs = BB.rowStride;
        int A_cs = columnStride;
        int B_cs = BB.columnStride;
        double[] elems = this.elements;
        double[] B_elems = BB.elements;
        if (elems == null || B_elems == null)
            throw new InternalError();

        for (int k = 1; k < slices - 1; k++) {
            int A_index = (int) index(k, 1, 1);
            int B_index = (int) BB.index(k, 1, 1);

            for (int i = 1; i < r; i++) {
                int A002 = A_index - A_ss - A_rs - A_cs;
                int A012 = A002 + A_rs;
                int A022 = A012 + A_rs;

                int A102 = A002 + A_ss;
                int A112 = A102 + A_rs;
                int A122 = A112 + A_rs;

                int A202 = A102 + A_ss;
                int A212 = A202 + A_rs;
                int A222 = A212 + A_rs;

                double a000, a001, a002;
                double a010, a011, a012;
                double a020, a021, a022;

                double a100, a101, a102;
                double a110, a111, a112;
                double a120, a121, a122;

                double a200, a201, a202;
                double a210, a211, a212;
                double a220, a221, a222;

                a000 = elems[A002];
                A002 += A_cs;
                a001 = elems[A002];
                a010 = elems[A012];
                A012 += A_cs;
                a011 = elems[A012];
                a020 = elems[A022];
                A022 += A_cs;
                a021 = elems[A022];

                a100 = elems[A102];
                A102 += A_cs;
                a101 = elems[A102];
                a110 = elems[A112];
                A112 += A_cs;
                a111 = elems[A112];
                a120 = elems[A122];
                A122 += A_cs;
                a121 = elems[A122];

                a200 = elems[A202];
                A202 += A_cs;
                a201 = elems[A202];
                a210 = elems[A212];
                A212 += A_cs;
                a211 = elems[A212];
                a220 = elems[A222];
                A222 += A_cs;
                a221 = elems[A222];

                int B11 = B_index;
                for (int j = 1; j < c; j++) {
                    // in each step 18 cells can be remembered in registers -
                    // they don't need to be reread from slow memory
                    // in each step 9 instead of 27 cells need to be read from
                    // memory.
                    a002 = elems[A002 += A_cs];
                    a012 = elems[A012 += A_cs];
                    a022 = elems[A022 += A_cs];

                    a102 = elems[A102 += A_cs];
                    a112 = elems[A112 += A_cs];
                    a122 = elems[A122 += A_cs];

                    a202 = elems[A202 += A_cs];
                    a212 = elems[A212 += A_cs];
                    a222 = elems[A222 += A_cs];

                    B_elems[B11] = function.apply(a000, a001, a002, a010, a011, a012, a020, a021, a022,

                    a100, a101, a102, a110, a111, a112, a120, a121, a122,

                    a200, a201, a202, a210, a211, a212, a220, a221, a222);
                    B11 += B_cs;

                    // move remembered cells
                    a000 = a001;
                    a001 = a002;
                    a010 = a011;
                    a011 = a012;
                    a020 = a021;
                    a021 = a022;

                    a100 = a101;
                    a101 = a102;
                    a110 = a111;
                    a111 = a112;
                    a120 = a121;
                    a121 = a122;

                    a200 = a201;
                    a201 = a202;
                    a210 = a211;
                    a211 = a212;
                    a220 = a221;
                    a221 = a222;
                }
                A_index += A_rs;
                B_index += B_rs;
            }
        }
    }

    public double zSum() {
        double sum = 0;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Double>() {

                    public Double call() throws Exception {
                        double sum = 0;
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    sum += elements[idx];
                                    idx += columnStride;
                                }
                            }
                        }
                        return Double.valueOf(sum);
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    sum = sum + ((Double) futures[j].get());
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
                        sum += elements[idx];
                        idx += columnStride;
                    }
                }
            }
        }
        return sum;
    }

    protected boolean haveSharedCellsRaw(DoubleMatrix3D other) {
        if (other instanceof SelectedDenseDoubleMatrix3D) {
            SelectedDenseDoubleMatrix3D otherMatrix = (SelectedDenseDoubleMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseDoubleMatrix3D) {
            DenseDoubleMatrix3D otherMatrix = (DenseDoubleMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    protected DoubleMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
        return new DenseDoubleMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride, true);
    }

    protected DoubleMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseDoubleMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
    }
}
