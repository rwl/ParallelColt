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
 * Dense 3-d matrix holding <tt>int</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Longernally holds one single contiguous one-dimensional array, addressed in
 * (in decreasing order of significance): slice major, row major, column major.
 * Note that this implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*slices()*rows()*columns()</tt>. Thus, a 100*100*100
 * matrix uses 8 MB.
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
public class DenseLongMatrix3D extends LongMatrix3D {
    private static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix. elements are stored in slice major, then row
     * major, then column major, in order of significance, i.e.
     * index==slice*sliceStride+ row*rowStride + column*columnStride i.e.
     * {slice0 row0..m}, {slice1 row0..m}, ..., {sliceN row0..m} with each row
     * storead as {row0 column0..m}, {row1 column0..m}, ..., {rown column0..m}
     */
    protected long[] elements;

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
    public DenseLongMatrix3D(long[][][] values) {
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
     *             if <tt>(int)slices*columns*rows > Long.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseLongMatrix3D(int slices, int rows, int columns) {
        setUp(slices, rows, columns);
        this.elements = new long[slices * rows * columns];
    }

    /**
     * Constructs a view with the given parameters.
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
     *             if <tt>(int)slices*columns*rows > Long.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseLongMatrix3D(int slices, int rows, int columns, long[] elements, int sliceZero, int rowZero,
            int columnZero, int sliceStride, int rowStride, int columnStride, boolean isView) {
        setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        long a = 0;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + firstSlice * sliceStride]);
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

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f, final cern.colt.function.tlong.LongProcedure cond) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        long a = 0;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long elem = elements[zero + firstSlice * sliceStride];
                        long a = 0;
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
            long elem = elements[zero];
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

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f, final IntArrayList sliceList, final IntArrayList rowList,
            final IntArrayList columnList) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        if (sliceList.size() == 0 || rowList.size() == 0 || columnList.size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int size = sliceList.size();
        final int[] sliceElements = sliceList.elements();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final int zero = (int) index(0, 0, 0);
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + sliceElements[firstIdx] * sliceStride + rowElements[firstIdx]
                                * rowStride + columnElements[firstIdx] * columnStride]);
                        long elem;
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
            long elem;
            for (int i = 1; i < size; i++) {
                elem = elements[zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i]
                        * columnStride];
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    public long aggregate(final LongMatrix3D other, final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongLongFunction f) {
        if (!(other instanceof DenseLongMatrix3D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        long a = 0;
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) other.index(0, 0, 0);
        final int sliceStrideOther = other.sliceStride();
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final long[] elemsOther = (long[]) other.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        int idx = zero + firstSlice * sliceStride;
                        int idxOther = zeroOther + firstSlice * sliceStrideOther;
                        long a = f.apply(elements[idx], elemsOther[idxOther]);
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    idx = zero + s * sliceStride + r * rowStride + c * columnStride;
                                    idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c
                                            * colStrideOther;
                                    a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
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
                        idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c * colStrideOther;
                        a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
                    }
                    d = 0;
                }
            }
        }
        return a;
    }

    public LongMatrix3D assign(final cern.colt.function.tlong.LongFunction function) {
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

    public LongMatrix3D assign(final long value) {
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

    public LongMatrix3D assign(final long[] values) {
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
                ConcurrencyUtils.waitForCompletion(futures);
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

    public LongMatrix3D assign(final int[] values) {
        if (values.length != size())
            throw new IllegalArgumentException("Must have same length: length=" + values.length
                    + "slices()*rows()*columns()=" + slices() * rows() * columns());
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
            ConcurrencyUtils.waitForCompletion(futures);
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
        return this;
    }

    public LongMatrix3D assign(final long[][][] values) {
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
                                long[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    long[] currentRow = currentSlice[r];
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
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int i = 0;
                for (int s = 0; s < slices; s++) {
                    long[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        long[] currentRow = currentSlice[r];
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
                                long[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    long[] currentRow = currentSlice[r];
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
                ConcurrencyUtils.waitForCompletion(futures);

            } else {
                int idx;
                for (int s = 0; s < slices; s++) {
                    long[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        long[] currentRow = currentSlice[r];
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

    public LongMatrix3D assign(final cern.colt.function.tlong.LongProcedure cond,
            final cern.colt.function.tlong.LongFunction f) {
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
                        long elem;
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
            long elem;
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

    public LongMatrix3D assign(final cern.colt.function.tlong.LongProcedure cond, final long value) {
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
                        long elem;
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
            long elem;
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

    public LongMatrix3D assign(LongMatrix3D source) {
        // overriden for performance only
        if (!(source instanceof DenseLongMatrix3D)) {
            super.assign(source);
            return this;
        }
        DenseLongMatrix3D other = (DenseLongMatrix3D) source;
        if (other == this)
            return this;
        checkShape(other);
        if (haveSharedCells(other)) {
            LongMatrix3D c = other.copy();
            if (!(c instanceof DenseLongMatrix3D)) { // should not happen
                super.assign(source);
                return this;
            }
            other = (DenseLongMatrix3D) c;
        }

        final DenseLongMatrix3D other_final = other;
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
            final long[] elemsOther = other_final.elements;
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
                            idx += columnStride;
                            idxOther += columnStrideOther;
                        }
                    }
                }
            }
            return this;
        }
    }

    public LongMatrix3D assign(final LongMatrix3D y, final cern.colt.function.tlong.LongLongFunction function) {
        if (!(y instanceof DenseLongMatrix3D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) y.index(0, 0, 0);
        final int sliceStrideOther = y.sliceStride();
        final int rowStrideOther = y.rowStride();
        final int columnStrideOther = y.columnStride();
        final long[] elemsOther = (long[]) y.elements();
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
                                    elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
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
                        elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
                        idx += columnStride;
                        idxOther += columnStrideOther;
                    }
                }
            }
        }

        return this;
    }

    public LongMatrix3D assign(final LongMatrix3D y, final cern.colt.function.tlong.LongLongFunction function,
            final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList) {
        if (!(y instanceof DenseLongMatrix3D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) y.index(0, 0, 0);
        final int sliceStrideOther = y.sliceStride();
        final int rowStrideOther = y.rowStride();
        final int columnStrideOther = y.columnStride();
        final long[] elemsOther = (long[]) y.elements();
        int size = sliceList.size();
        final int[] sliceElements = sliceList.elements();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, size);
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
                            elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
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
                elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
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

    public long[] elements() {
        return elements;
    }

    public void getNegativeValues(final IntArrayList sliceList, final IntArrayList rowList,
            final IntArrayList columnList, final LongArrayList valueList) {
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
                    long value = elements[idx];
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
            final LongArrayList valueList) {
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
                    long value = elements[idx];
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
            final IntArrayList columnList, final LongArrayList valueList) {
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
                    long value = elements[idx];
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

    public long getQuick(int slice, int row, int column) {
        return elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column
                * columnStride];
    }

    public long index(int slice, int row, int column) {
        return sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
    }

    public LongMatrix3D like(int slices, int rows, int columns) {
        return new DenseLongMatrix3D(slices, rows, columns);
    }

    public LongMatrix2D like2D(int rows, int columns) {
        return new DenseLongMatrix2D(rows, columns);
    }

    public long[] getMaxLocation() {
        final int zero = (int) index(0, 0, 0);
        int slice_loc = 0;
        int row_loc = 0;
        int col_loc = 0;
        long maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        int slice_loc = firstSlice;
                        int row_loc = 0;
                        int col_loc = 0;
                        long maxValue = elements[zero + firstSlice * sliceStride];
                        int d = 1;
                        long elem;
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
                        return new long[] { maxValue, slice_loc, row_loc, col_loc };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
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
            long elem;
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
        return new long[] { maxValue, slice_loc, row_loc, col_loc };
    }

    public long[] getMinLocation() {
        final int zero = (int) index(0, 0, 0);
        int slice_loc = 0;
        int row_loc = 0;
        int col_loc = 0;
        long minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        int slice_loc = firstSlice;
                        int row_loc = 0;
                        int col_loc = 0;
                        long minValue = elements[zero + slice_loc * sliceStride];
                        int d = 1;
                        long elem;
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
                        return new long[] { minValue, slice_loc, row_loc, col_loc };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
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
            long elem;
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
        return new long[] { minValue, slice_loc, row_loc, col_loc };
    }

    public void setQuick(int slice, int row, int column, long value) {
        elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride] = value;
    }

    public long[][][] toArray() {
        final long[][][] values = new long[slices][rows][columns];
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
                            long[][] currentSlice = values[s];
                            for (int r = 0; r < rows; r++) {
                                idx = zero + s * sliceStride + r * rowStride;
                                long[] currentRow = currentSlice[r];
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
                long[][] currentSlice = values[s];
                for (int r = 0; r < rows; r++) {
                    idx = zero + s * sliceStride + r * rowStride;
                    long[] currentRow = currentSlice[r];
                    for (int c = 0; c < columns; c++) {
                        currentRow[c] = elements[idx];
                        idx += columnStride;
                    }
                }
            }
        }
        return values;
    }

    public LongMatrix1D vectorize() {
        LongMatrix1D v = new DenseLongMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    public long zSum() {
        long sum = 0;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long sum = 0;
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
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    sum = sum + ((Long) futures[j].get());
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

    protected boolean haveSharedCellsRaw(LongMatrix3D other) {
        if (other instanceof SelectedDenseLongMatrix3D) {
            SelectedDenseLongMatrix3D otherMatrix = (SelectedDenseLongMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseLongMatrix3D) {
            DenseLongMatrix3D otherMatrix = (DenseLongMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    protected LongMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
        return new DenseLongMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride, true);
    }

    protected LongMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseLongMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
    }
}
