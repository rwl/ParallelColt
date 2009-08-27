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
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>int</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Longernally holds one single contigous one-dimensional array, addressed in
 * row major. Note that this implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*rows()*columns()</tt>. Thus, a 1000*1000 matrix uses 8
 * MB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
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
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseLongMatrix2D extends LongMatrix2D {
    static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix. elements are stored in row major, i.e.
     * index==row*columns + column columnOf(index)==index%columns
     * rowOf(index)==index/columns i.e. {row0 column0..m}, {row1 column0..m},
     * ..., {rown column0..m}
     */
    protected long[] elements;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form <tt>values[row][column]</tt> and have exactly
     * the same number of columns in every row.
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
    public DenseLongMatrix2D(long[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length);
        assign(values);
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
     *             <tt>rows<0 || columns<0 || (int)columns*rows > Long.MAX_VALUE</tt>
     *             .
     */
    public DenseLongMatrix2D(int rows, int columns) {
        setUp(rows, columns);
        this.elements = new long[rows * columns];
    }

    /**
     * Constructs a view with the given parameters.
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
     * @param isView
     *            if true then a matrix view is constructed
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (int)columns*rows > Long.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    public DenseLongMatrix2D(int rows, int columns, long[] elements, int rowZero, int columnZero, int rowStride,
            int columnStride, boolean isView) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int zero = (int) index(0, 0);
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + firstRow * rowStride]);
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride]));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero]);
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride]));
                }
                d = 0;
            }
        }
        return a;
    }

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f, final cern.colt.function.tlong.LongProcedure cond) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int zero = (int) index(0, 0);
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long elem = elements[zero + firstRow * rowStride];
                        long a = 0;
                        if (cond.apply(elem) == true) {
                            a = f.apply(elem);
                        }
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                elem = elements[zero + r * rowStride + c * columnStride];
                                if (cond.apply(elem) == true) {
                                    a = aggr.apply(a, f.apply(elem));
                                }
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            long elem = elements[zero];
            if (cond.apply(elem) == true) {
                a = f.apply(elements[zero]);
            }
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    elem = elements[zero + r * rowStride + c * columnStride];
                    if (cond.apply(elem) == true) {
                        a = aggr.apply(a, f.apply(elem));
                    }
                }
                d = 0;
            }
        }
        return a;
    }

    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f, final IntArrayList rowList, final IntArrayList columnList) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int zero = (int) index(0, 0);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + rowElements[firstIdx] * rowStride + columnElements[firstIdx]
                                * columnStride]);
                        long elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            elem = elements[zero + rowElements[i] * rowStride + columnElements[i] * columnStride];
                            a = aggr.apply(a, f.apply(elem));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            long elem;
            a = f.apply(elements[zero + rowElements[0] * rowStride + columnElements[0] * columnStride]);
            for (int i = 1; i < size; i++) {
                elem = elements[zero + rowElements[i] * rowStride + columnElements[i] * columnStride];
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    public long aggregate(final LongMatrix2D other, final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongLongFunction f) {
        if (!(other instanceof DenseLongMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final long[] elemsOther = (long[]) other.elements();
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + firstRow * rowStride], elemsOther[zeroOther + firstRow
                                * rowStrideOther]);
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride],
                                        elemsOther[zeroOther + r * rowStrideOther + c * colStrideOther]));
                            }
                            d = 0;
                        }
                        return Long.valueOf(a);
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            int d = 1; // first cell already done
            a = f.apply(elements[zero], elemsOther[zeroOther]);
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride], elemsOther[zeroOther
                            + r * rowStrideOther + c * colStrideOther]));
                }
                d = 0;
            }
        }
        return a;
    }

    public LongMatrix2D assign(final cern.colt.function.tlong.LongFunction function) {
        final long[] elems = this.elements;
        if (elems == null)
            throw new InternalError();
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tlong.LongMult) { // x[i] =
                // mult*x[i]
                long multiplicator = ((cern.jet.math.tlong.LongMult) function).multiplicator;
                if (multiplicator == 1)
                    return this;
                if (multiplicator == 0)
                    return assign(0);
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
                        // specialization for speed
                        if (function instanceof cern.jet.math.tlong.LongMult) {
                            // x[i] = mult*x[i]
                            long multiplicator = ((cern.jet.math.tlong.LongMult) function).multiplicator;
                            if (multiplicator == 1)
                                return;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elems[i] *= multiplicator;
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        } else {
                            // the general case x[i] = f(x[i])
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elems[i] = function.apply(elems[i]);
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
            // specialization for speed
            if (function instanceof cern.jet.math.tlong.LongMult) { // x[i] =
                // mult*x[i]
                long multiplicator = ((cern.jet.math.tlong.LongMult) function).multiplicator;
                if (multiplicator == 1)
                    return this;
                if (multiplicator == 0)
                    return assign(0);
                for (int r = 0; r < rows; r++) { // the general case
                    for (int i = idx, c = 0; c < columns; c++) {
                        elems[i] *= multiplicator;
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            } else { // the general case x[i] = f(x[i])
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        elems[i] = function.apply(elems[i]);
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public LongMatrix2D assign(final cern.colt.function.tlong.LongProcedure cond,
            final cern.colt.function.tlong.LongFunction function) {
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
                        long elem;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem = elements[i];
                                if (cond.apply(elem) == true) {
                                    elements[i] = function.apply(elem);
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
            long elem;
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem = elements[i];
                    if (cond.apply(elem) == true) {
                        elements[i] = function.apply(elem);
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public LongMatrix2D assign(final cern.colt.function.tlong.LongProcedure cond, final long value) {
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
                        long elem;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem = elements[i];
                                if (cond.apply(elem) == true) {
                                    elements[i] = value;
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
            long elem;
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem = elements[i];
                    if (cond.apply(elem) == true) {
                        elements[i] = value;
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public LongMatrix2D assign(final long value) {
        final long[] elems = this.elements;
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
                                elems[i] = value;
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
                    elems[i] = value;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public LongMatrix2D assign(final long[] values) {
        if (values.length != size())
            throw new IllegalArgumentException("Must have same length: length=" + values.length + " rows()*columns()="
                    + rows() * columns());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            System.arraycopy(values, 0, this.elements, 0, values.length);
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
                            int idxOther = firstRow * columns;
                            int idx = zero + firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = values[idxOther++];
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
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public LongMatrix2D assign(final int[] values) {
        if (values.length != size())
            throw new IllegalArgumentException("Must have same length: length=" + values.length + " rows()*columns()="
                    + rows() * columns());
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
                        int idxOther = firstRow * columns;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elements[i] = values[idxOther++];
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
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public LongMatrix2D assign(final long[][] values) {
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
                            int i = firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                long[] currentRow = values[r];
                                if (currentRow.length != columns)
                                    throw new IllegalArgumentException(
                                            "Must have same number of columns in every row: columns="
                                                    + currentRow.length + "columns()=" + columns());
                                System.arraycopy(currentRow, 0, elements, i, columns);
                                i += columns;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int i = 0;
                for (int r = 0; r < rows; r++) {
                    long[] currentRow = values[r];
                    if (currentRow.length != columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "columns()=" + columns());
                    System.arraycopy(currentRow, 0, this.elements, i, columns);
                    i += columns;
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
                                long[] currentRow = values[r];
                                if (currentRow.length != columns)
                                    throw new IllegalArgumentException(
                                            "Must have same number of columns in every row: columns="
                                                    + currentRow.length + "columns()=" + columns());
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = currentRow[c];
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
                    long[] currentRow = values[r];
                    if (currentRow.length != columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "columns()=" + columns());
                    for (int i = idx, c = 0; c < columns; c++) {
                        elements[i] = currentRow[c];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
            return this;
        }
        return this;
    }

    public LongMatrix2D assign(final LongMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof DenseLongMatrix2D)) {
            super.assign(source);
            return this;
        }
        final DenseLongMatrix2D other_final = (DenseLongMatrix2D) source;
        if (other_final == this)
            return this; // nothing to do
        checkShape(other_final);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other_final.isNoView) { // quickest
            System.arraycopy(other_final.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        DenseLongMatrix2D other = (DenseLongMatrix2D) source;
        if (haveSharedCells(other)) {
            LongMatrix2D c = other.copy();
            if (!(c instanceof DenseLongMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseLongMatrix2D) c;
        }

        final long[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
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

    public LongMatrix2D assign(final LongMatrix2D y, final cern.colt.function.tlong.LongLongFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseLongMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        DenseLongMatrix2D other = (DenseLongMatrix2D) y;
        checkShape(y);
        final long[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
                long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
                if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                    return this;
                }
            }
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx;
                        int idxOther;
                        // specialized for speed
                        if (function == cern.jet.math.tlong.LongFunctions.mult) {
                            // x[i] = x[i]*y[i]
                            idx = zero + firstRow * rowStride;
                            idxOther = zeroOther + firstRow * rowStrideOther;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] *= elemsOther[j];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else if (function == cern.jet.math.tlong.LongFunctions.div) {
                            // x[i] = x[i] / y[i]
                            idx = zero + firstRow * rowStride;
                            idxOther = zeroOther + firstRow * rowStrideOther;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] /= elemsOther[j];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
                            long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
                            if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                idx = zero + firstRow * rowStride;
                                idxOther = zeroOther + firstRow * rowStrideOther;
                                for (int r = firstRow; r < lastRow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] += elemsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = x[i] - y[i]
                                idx = zero + firstRow * rowStride;
                                idxOther = zeroOther + firstRow * rowStrideOther;
                                for (int r = firstRow; r < lastRow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] -= elemsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            } else { // the general case
                                // x[i] = x[i] + mult*y[i]
                                idx = zero + firstRow * rowStride;
                                idxOther = zeroOther + firstRow * rowStrideOther;
                                for (int r = firstRow; r < lastRow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] += multiplicator * elemsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            }
                        } else { // the general case x[i] = f(x[i],y[i])
                            idx = zero + firstRow * rowStride;
                            idxOther = zeroOther + firstRow * rowStrideOther;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] = function.apply(elements[i], elemsOther[j]);
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
            int idx;
            int idxOther;
            // specialized for speed
            if (function == cern.jet.math.tlong.LongFunctions.mult) {
                // x[i] = x[i] * y[i]
                idx = zero;
                idxOther = zeroOther;
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        elements[i] *= elemsOther[j];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            } else if (function == cern.jet.math.tlong.LongFunctions.div) {
                // x[i] = x[i] / y[i]
                idx = zero;
                idxOther = zeroOther;
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        elements[i] /= elemsOther[j];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            } else if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
                long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
                if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                    return this;
                } else if (multiplicator == 1) { // x[i] = x[i] + y[i]
                    idx = zero;
                    idxOther = zeroOther;
                    for (int r = 0; r < rows; r++) {
                        for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                            elements[i] += elemsOther[j];
                            i += columnStride;
                            j += columnStrideOther;
                        }
                        idx += rowStride;
                        idxOther += rowStrideOther;
                    }

                } else if (multiplicator == -1) { // x[i] = x[i] - y[i]
                    idx = zero;
                    idxOther = zeroOther;
                    for (int r = 0; r < rows; r++) {
                        for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                            elements[i] -= elemsOther[j];
                            i += columnStride;
                            j += columnStrideOther;
                        }
                        idx += rowStride;
                        idxOther += rowStrideOther;
                    }
                } else { // the general case
                    // x[i] = x[i] + mult*y[i]
                    idx = zero;
                    idxOther = zeroOther;
                    for (int r = 0; r < rows; r++) {
                        for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                            elements[i] += multiplicator * elemsOther[j];
                            i += columnStride;
                            j += columnStrideOther;
                        }
                        idx += rowStride;
                        idxOther += rowStrideOther;
                    }
                }
            } else { // the general case x[i] = f(x[i],y[i])
                idx = zero;
                idxOther = zeroOther;
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        elements[i] = function.apply(elements[i], elemsOther[j]);
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

    public LongMatrix2D assign(final LongMatrix2D y, final cern.colt.function.tlong.LongLongFunction function,
            IntArrayList rowList, IntArrayList columnList) {
        checkShape(y);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final long[] elemsOther = (long[]) y.elements();
        final int zeroOther = (int) y.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = y.columnStride();
        final int rowStrideOther = y.rowStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx;
                        int idxOther;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                            idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i]
                                    * columnStrideOther;
                            elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
                        }
                    }

                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int i = 0; i < size; i++) {
                idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i] * columnStrideOther;
                elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
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
            Long[] results = new Long[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        int cardinality = 0;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                if (elements[i] != 0)
                                    cardinality++;
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        return Long.valueOf(cardinality);
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Long) futures[j].get();
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
                    if (elements[i] != 0)
                        cardinality++;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return cardinality;
    }

    public long[] elements() {
        return elements;
    }

    public LongMatrix2D forEachNonZero(final cern.colt.function.tlong.IntIntLongFunction function) {
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
                                long value = elements[i];
                                if (value != 0) {
                                    elements[i] = function.apply(r, c, value);
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
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    long value = elements[i];
                    if (value != 0) {
                        elements[i] = function.apply(r, c, value);
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public void getNegativeValues(final IntArrayList rowList, final IntArrayList columnList,
            final LongArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                long value = elements[i];
                if (value < 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += columnStride;
            }
            idx += rowStride;
        }
    }

    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList, final LongArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                long value = elements[i];
                if (value != 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += columnStride;
            }
            idx += rowStride;
        }
    }

    public void getPositiveValues(final IntArrayList rowList, final IntArrayList columnList,
            final LongArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                long value = elements[i];
                if (value > 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += columnStride;
            }
            idx += rowStride;
        }
    }

    public long getQuick(int row, int column) {
        return elements[rowZero + row * rowStride + columnZero + column * columnStride];
    }

    public long index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    public LongMatrix2D like(int rows, int columns) {
        return new DenseLongMatrix2D(rows, columns);
    }

    public LongMatrix1D like1D(int size) {
        return new DenseLongMatrix1D(size);
    }

    public long[] getMaxLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = (int) index(0, 0);
        long maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        long maxValue = elements[zero + firstRow * rowStride];
                        int rowLocation = firstRow;
                        int colLocation = 0;
                        long elem;
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                elem = elements[zero + r * rowStride + c * columnStride];
                                if (maxValue < elem) {
                                    maxValue = elem;
                                    rowLocation = r;
                                    colLocation = c;
                                }
                            }
                            d = 0;
                        }
                        return new long[] { maxValue, rowLocation, colLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                maxValue = results[0][0];
                rowLocation = (int) results[0][1];
                columnLocation = (int) results[0][2];
                for (int j = 1; j < nthreads; j++) {
                    if (maxValue < results[j][0]) {
                        maxValue = results[j][0];
                        rowLocation = (int) results[j][1];
                        columnLocation = (int) results[j][2];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            maxValue = elements[zero];
            int d = 1;
            long elem;
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    elem = elements[zero + r * rowStride + c * columnStride];
                    if (maxValue < elem) {
                        maxValue = elem;
                        rowLocation = r;
                        columnLocation = c;
                    }
                }
                d = 0;
            }
        }
        return new long[] { maxValue, rowLocation, columnLocation };
    }

    public long[] getMinLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = (int) index(0, 0);
        long minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        int rowLocation = firstRow;
                        int columnLocation = 0;
                        long minValue = elements[zero + firstRow * rowStride];
                        long elem;
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                elem = elements[zero + r * rowStride + c * columnStride];
                                if (minValue > elem) {
                                    minValue = elem;
                                    rowLocation = r;
                                    columnLocation = c;
                                }
                            }
                            d = 0;
                        }
                        return new long[] { minValue, rowLocation, columnLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                minValue = results[0][0];
                rowLocation = (int) results[0][1];
                columnLocation = (int) results[0][2];
                for (int j = 1; j < nthreads; j++) {
                    if (minValue > results[j][0]) {
                        minValue = results[j][0];
                        rowLocation = (int) results[j][1];
                        columnLocation = (int) results[j][2];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            minValue = elements[zero];
            int d = 1;
            long elem;
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    elem = elements[zero + r * rowStride + c * columnStride];
                    if (minValue > elem) {
                        minValue = elem;
                        rowLocation = r;
                        columnLocation = c;
                    }
                }
                d = 0;
            }
        }
        return new long[] { minValue, rowLocation, columnLocation };
    }

    public void setQuick(int row, int column, long value) {
        elements[rowZero + row * rowStride + columnZero + column * columnStride] = value;
    }

    public long[][] toArray() {
        final long[][] values = new long[rows][columns];
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
                            long[] currentRow = values[r];
                            for (int i = idx, c = 0; c < columns; c++) {
                                currentRow[c] = elements[i];
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
                long[] currentRow = values[r];
                for (int i = idx, c = 0; c < columns; c++) {
                    currentRow[c] = elements[i];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return values;
    }

    public LongMatrix1D vectorize() {
        final DenseLongMatrix1D v = new DenseLongMatrix1D((int) size());
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) v.index(0);
        final int strideOther = v.stride();
        final long[] elemsOther = v.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                final int startidx = j * k * rows;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = 0;
                        int idxOther = zeroOther + startidx * strideOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            idx = zero + c * columnStride;
                            for (int r = 0; r < rows; r++) {
                                elemsOther[idxOther] = elements[idx];
                                idx += rowStride;
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
            for (int c = 0; c < columns; c++) {
                idx = zero + c * columnStride;
                for (int r = 0; r < rows; r++) {
                    elemsOther[idxOther] = elements[idx];
                    idx += rowStride;
                    idxOther += strideOther;
                }
            }
        }
        return v;
    }

    public LongMatrix1D zMult(final LongMatrix1D y, LongMatrix1D z, final long alpha, final long beta,
            final boolean transposeA) {
        if (transposeA)
            return viewDice().zMult(y, z, alpha, beta, false);
        if (z == null) {
            z = new DenseLongMatrix1D(rows);
        }
        if (!(y instanceof DenseLongMatrix1D && z instanceof DenseLongMatrix1D))
            return super.zMult(y, z, alpha, beta, transposeA);

        if (columns != y.size() || rows > z.size())
            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort()
                    + ", " + z.toStringShort());

        final long[] elemsY = (long[]) y.elements();
        final long[] elemsZ = (long[]) z.elements();
        if (elements == null || elemsY == null || elemsZ == null)
            throw new InternalError();
        final int strideY = y.stride();
        final int strideZ = z.stride();
        final int zero = (int) index(0, 0);
        final int zeroY = (int) y.index(0);
        final int zeroZ = (int) z.index(0);
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
                        for (int r = firstRow; r < lastRow; r++) {
                            long sum = 0;
                            int idx = idxZero;
                            int idxY = zeroY;
                            for (int c = 0; c < columns; c++) {
                                sum += elements[idx] * elemsY[idxY];
                                idx += columnStride;
                                idxY += strideY;
                            }
                            elemsZ[idxZeroZ] = alpha * sum + beta * elemsZ[idxZeroZ];
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
            for (int r = 0; r < rows; r++) {
                long sum = 0;
                int idx = idxZero;
                int idxY = zeroY;
                for (int c = 0; c < columns; c++) {
                    sum += elements[idx] * elemsY[idxY];
                    idx += columnStride;
                    idxY += strideY;
                }
                elemsZ[idxZeroZ] = alpha * sum + beta * elemsZ[idxZeroZ];
                idxZero += rowStride;
                idxZeroZ += strideZ;
            }
        }
        return z;
    }

    public LongMatrix2D zMult(final LongMatrix2D B, LongMatrix2D C, final long alpha, final long beta,
            final boolean transposeA, final boolean transposeB) {
        final int rowsA = rows;
        final int columnsA = columns;
        final int rowsB = B.rows();
        final int columnsB = B.columns();
        final int rowsC = transposeA ? columnsA : rowsA;
        final int columnsC = transposeB ? rowsB : columnsB;

        if (C == null) {
            C = new DenseLongMatrix2D(rowsC, columnsC);
        }

        /*
        * determine how to split and parallelize best into blocks if more
        * B.columns than tasks --> split B.columns, as follows:
        * 
        * xx|xx|xxx B xx|xx|xxx xx|xx|xxx A xxx xx|xx|xxx C xxx xx|xx|xxx xxx
        * xx|xx|xxx xxx xx|xx|xxx xxx xx|xx|xxx
        * 
        * if less B.columns than tasks --> split A.rows, as follows:
        * 
        * xxxxxxx B xxxxxxx xxxxxxx A xxx xxxxxxx C xxx xxxxxxx --- ------- xxx
        * xxxxxxx xxx xxxxxxx --- ------- xxx xxxxxxx
        */
        if (transposeA)
            return viewDice().zMult(B, C, alpha, beta, false, transposeB);
        if (B instanceof SparseLongMatrix2D || B instanceof SparseRCLongMatrix2D) {
            // exploit quick sparse mult
            // A*B = (B' * A')'
            if (C == null) {
                return B.zMult(this, null, alpha, beta, !transposeB, true).viewDice();
            } else {
                B.zMult(this, C.viewDice(), alpha, beta, !transposeB, true);
                return C;
            }
        }
        if (transposeB)
            return this.zMult(B.viewDice(), C, alpha, beta, transposeA, false);

        if (!(C instanceof DenseLongMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);

        if (B.rows() != columnsA)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + this.toStringShort() + ", "
                    + B.toStringShort());
        if (C.rows() != rowsA || C.columns() != columnsB)
            throw new IllegalArgumentException("Incompatibe result matrix: " + this.toStringShort() + ", "
                    + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        long flops = 2L * rowsA * columnsA * columnsB;
        int noOfTasks = (int) Math.min(flops / 30000, ConcurrencyUtils.getNumberOfThreads()); // each
        /* thread should process at least 30000 flops */
        boolean splitB = (columnsB >= noOfTasks);
        int width = splitB ? columnsB : rowsA;
        noOfTasks = Math.min(width, noOfTasks);

        if (noOfTasks < 2) { //parallelization doesn't pay off (too much start up overhead)
            return this.zMultSequential(B, C, alpha, beta, transposeA, transposeB);
        }

        // set up concurrent tasks
        int span = width / noOfTasks;
        final Future<?>[] subTasks = new Future[noOfTasks];
        for (int i = 0; i < noOfTasks; i++) {
            final int offset = i * span;
            if (i == noOfTasks - 1)
                span = width - span * i; // last span may be a bit larger

            final LongMatrix2D AA, BB, CC;
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
                    ((DenseLongMatrix2D) AA).zMultSequential(BB, CC, alpha, beta, transposeA, transposeB);
                }
            });
        }

        ConcurrencyUtils.waitForCompletion(subTasks);
        return C;
    }

    public long zSum() {
        long sum = 0;
        if (elements == null)
            throw new InternalError();
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long sum = 0;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                sum += elements[i];
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    sum += (Long) futures[j].get();
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
                    sum += elements[i];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return sum;
    }

    private LongMatrix2D zMultSequential(LongMatrix2D B, LongMatrix2D C, long alpha, long beta, boolean transposeA,
            boolean transposeB) {
        if (transposeA)
            return viewDice().zMult(B, C, alpha, beta, false, transposeB);
        if (B instanceof SparseLongMatrix2D || B instanceof SparseRCLongMatrix2D || B instanceof SparseCCLongMatrix2D) {
            // exploit quick sparse mult
            // A*B = (B' * A')'
            if (C == null) {
                return B.zMult(this, null, alpha, beta, !transposeB, true).viewDice();
            } else {
                B.zMult(this, C.viewDice(), alpha, beta, !transposeB, true);
                return C;
            }
        }
        if (transposeB)
            return this.zMult(B.viewDice(), C, alpha, beta, transposeA, false);

        int rowsA = rows;
        int columnsA = columns;
        int p = B.columns();
        if (C == null) {
            C = new DenseLongMatrix2D(rowsA, p);
        }
        if (!(B instanceof DenseLongMatrix2D) || !(C instanceof DenseLongMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        if (B.rows() != columnsA)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + B.toStringShort());
        if (C.rows() != rowsA || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", "
                    + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        DenseLongMatrix2D BB = (DenseLongMatrix2D) B;
        DenseLongMatrix2D CC = (DenseLongMatrix2D) C;
        final long[] AElems = this.elements;
        final long[] BElems = BB.elements;
        final long[] CElems = CC.elements;
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
        int m_optimal = (BLOCK_SIZE - columnsA) / (columnsA + 1);
        if (m_optimal <= 0)
            m_optimal = 1;
        int blocks = rowsA / m_optimal;
        int rr = 0;
        if (rowsA % m_optimal != 0)
            blocks++;
        for (; --blocks >= 0;) {
            int jB = (int) BB.index(0, 0);
            int indexA = (int) index(rr, 0);
            int jC = (int) CC.index(rr, 0);
            rr += m_optimal;
            if (blocks == 0)
                m_optimal += rowsA - rr;

            for (int j = p; --j >= 0;) {
                int iA = indexA;
                int iC = jC;
                for (int i = m_optimal; --i >= 0;) {
                    int kA = iA;
                    int kB = jB;
                    long s = 0;

                    // loop unrolled
                    kA -= cA;
                    kB -= rB;

                    for (int k = columnsA % 4; --k >= 0;) {
                        s += AElems[kA += cA] * BElems[kB += rB];
                    }
                    for (int k = columnsA / 4; --k >= 0;) {
                        s += AElems[kA += cA] * BElems[kB += rB] + AElems[kA += cA] * BElems[kB += rB]
                                + AElems[kA += cA] * BElems[kB += rB] + AElems[kA += cA] * BElems[kB += rB];
                    }

                    CElems[iC] = alpha * s + beta * CElems[iC];
                    iA += rA;
                    iC += rC;
                }
                jB += cB;
                jC += cC;
            }
        }
        return C;
    }

    protected boolean haveSharedCellsRaw(LongMatrix2D other) {
        if (other instanceof SelectedDenseLongMatrix2D) {
            SelectedDenseLongMatrix2D otherMatrix = (SelectedDenseLongMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseLongMatrix2D) {
            DenseLongMatrix2D otherMatrix = (DenseLongMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    protected LongMatrix1D like1D(int size, int zero, int stride) {
        return new DenseLongMatrix1D(size, this.elements, zero, stride, true);
    }

    protected LongMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseLongMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }
}
