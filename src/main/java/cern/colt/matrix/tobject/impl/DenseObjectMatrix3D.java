/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tobject.ObjectMatrix1D;
import cern.colt.matrix.tobject.ObjectMatrix2D;
import cern.colt.matrix.tobject.ObjectMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 3-d matrix holding <tt>Object</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array, addressed in (in
 * decreasing order of significance): slice major, row major, column major. Note
 * that this implementation is not synchronized.
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
 */
public class DenseObjectMatrix3D extends ObjectMatrix3D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /**
     * The elements of this matrix. elements are stored in slice major, then row
     * major, then column major, in order of significance, i.e.
     * index==slice*sliceStride+ row*rowStride + column*columnStride i.e.
     * {slice0 row0..m}, {slice1 row0..m}, ..., {sliceN row0..m} with each row
     * storead as {row0 column0..m}, {row1 column0..m}, ..., {rown column0..m}
     */
    protected Object[] elements;

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
    public DenseObjectMatrix3D(Object[][][] values) {
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
     *             if <tt>(Object)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public DenseObjectMatrix3D(int slices, int rows, int columns) {
        setUp(slices, rows, columns);
        this.elements = new Object[slices * rows * columns];
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
     * @param columnnStride
     *            the number of elements between two columns, i.e.
     *            <tt>index(k,i,j+1)-index(k,i,j)</tt>.
     * @param isView
     *            if true then a matrix view is constructed.
     * @throws IllegalArgumentException
     *             if <tt>(Object)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    protected DenseObjectMatrix3D(int slices, int rows, int columns, Object[] elements, int sliceZero, int rowZero,
            int columnZero, int sliceStride, int rowStride, int columnStride, boolean isView) {
        setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    public Object aggregate(final cern.colt.function.tobject.ObjectObjectFunction aggr,
            final cern.colt.function.tobject.ObjectFunction f) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        Object a = null;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Object>() {

                    public Object call() throws Exception {
                        Object a = f.apply(elements[zero + firstSlice * sliceStride]);
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

    public Object aggregate(final cern.colt.function.tobject.ObjectObjectFunction aggr,
            final cern.colt.function.tobject.ObjectFunction f, final cern.colt.function.tobject.ObjectProcedure cond) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        Object a = null;
        final int zero = (int) index(0, 0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Object>() {

                    public Object call() throws Exception {
                        Object elem = elements[zero + firstSlice * sliceStride];
                        Object a = 0;
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
            Object elem = elements[zero];
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

    public Object aggregate(final cern.colt.function.tobject.ObjectObjectFunction aggr,
            final cern.colt.function.tobject.ObjectFunction f, final IntArrayList sliceList,
            final IntArrayList rowList, final IntArrayList columnList) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        if (sliceList.size() == 0 || rowList.size() == 0 || columnList.size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int size = sliceList.size();
        final int[] sliceElements = sliceList.elements();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final int zero = (int) index(0, 0, 0);
        Object a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Object>() {

                    public Object call() throws Exception {
                        Object a = f.apply(elements[zero + sliceElements[firstIdx] * sliceStride
                                + rowElements[firstIdx] * rowStride + columnElements[firstIdx] * columnStride]);
                        Object elem;
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
            Object elem;
            for (int i = 1; i < size; i++) {
                elem = elements[zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i]
                        * columnStride];
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    public Object aggregate(final ObjectMatrix3D other, final cern.colt.function.tobject.ObjectObjectFunction aggr,
            final cern.colt.function.tobject.ObjectObjectFunction f) {
        if (!(other instanceof DenseObjectMatrix3D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        Object a = null;
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) other.index(0, 0, 0);
        final int sliceStrideOther = other.sliceStride();
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final Object[] elemsOther = (Object[]) other.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Object>() {
                    public Object call() throws Exception {
                        int idx = zero + firstSlice * sliceStride;
                        int idxOther = zeroOther + firstSlice * sliceStrideOther;
                        Object a = f.apply(elements[idx], elemsOther[idxOther]);
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

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form <tt>values[slice][row][column]</tt> and have
     * exactly the same number of slices, rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>values.length != slices() || for any 0 &lt;= slice &lt; slices(): values[slice].length != rows()</tt>
     *             .
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 0 &lt;= column &lt; columns(): values[slice][row].length != columns()</tt>
     *             .
     */

    public ObjectMatrix3D assign(final Object[][][] values) {
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
                                Object[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    Object[] currentRow = currentSlice[r];
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
                    Object[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        Object[] currentRow = currentSlice[r];
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
                                Object[][] currentSlice = values[s];
                                if (currentSlice.length != rows)
                                    throw new IllegalArgumentException(
                                            "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                    + "rows()=" + rows());
                                for (int r = 0; r < rows; r++) {
                                    idx = zero + s * sliceStride + r * rowStride;
                                    Object[] currentRow = currentSlice[r];
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
                    Object[][] currentSlice = values[s];
                    if (currentSlice.length != rows)
                        throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                                + currentSlice.length + "rows()=" + rows());
                    for (int r = 0; r < rows; r++) {
                        idx = zero + s * sliceStride + r * rowStride;
                        Object[] currentRow = currentSlice[r];
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
    
    public ObjectMatrix3D assign(final Object[] values) {
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
    
    public ObjectMatrix3D assign(final Object value) {
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

    public ObjectMatrix3D assign(final cern.colt.function.tobject.ObjectProcedure cond,
            final cern.colt.function.tobject.ObjectFunction f) {
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
                        Object elem;
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
            Object elem;
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

    public ObjectMatrix3D assign(final cern.colt.function.tobject.ObjectProcedure cond, final Object value) {
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
                        Object elem;
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
            Object elem;
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

    /**
     * Replaces all cell values of the receiver with the values of another
     * matrix. Both matrices must have the same number of slices, rows and
     * columns. If both matrices share the same cells (as is the case if they
     * are views derived from the same matrix) and intersect in an ambiguous
     * way, then replaces <i>as if</i> using an intermediate auxiliary deep copy
     * of <tt>other</tt>.
     * 
     * @param source
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>slices() != source.slices() || rows() != source.rows() || columns() != source.columns()</tt>
     */

    public ObjectMatrix3D assign(ObjectMatrix3D source) {
        // overriden for performance only
        if (!(source instanceof DenseObjectMatrix3D)) {
            return super.assign(source);
        }
        DenseObjectMatrix3D other = (DenseObjectMatrix3D) source;
        if (other == this)
            return this;
        checkShape(other);
        if (haveSharedCells(other)) {
            ObjectMatrix3D c = other.copy();
            if (!(c instanceof DenseObjectMatrix3D)) { // should not happen
                super.assign(source);
                return this;
            }
            other = (DenseObjectMatrix3D) c;
        }

        final DenseObjectMatrix3D other_final = other;
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
            final Object[] elemsOther = other_final.elements;
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
                nthreads = Math.min(nthreads, slices);
                Future<?>[] futures = new Future[nthreads];
                int k = slices / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstSlice = j * k;
                    final int lastSlice;
                    if (j == nthreads - 1) {
                        lastSlice = slices;
                    } else {
                        lastSlice = firstSlice + k;
                    }
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
    
    public ObjectMatrix3D assign(final cern.colt.function.tobject.ObjectFunction function) {
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

    
    public ObjectMatrix3D assign(final ObjectMatrix3D y, final cern.colt.function.tobject.ObjectObjectFunction function) {
        if (!(y instanceof DenseObjectMatrix3D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) y.index(0, 0, 0);
        final int sliceStrideOther = y.sliceStride();
        final int rowStrideOther = y.rowStride();
        final int columnStrideOther = y.columnStride();
        final Object[] elemsOther = (Object[]) y.elements();
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

    public ObjectMatrix3D assign(final ObjectMatrix3D y, final cern.colt.function.tobject.ObjectObjectFunction function,
            final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList) {
        if (!(y instanceof DenseObjectMatrix3D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final int zero = (int) index(0, 0, 0);
        final int zeroOther = (int) y.index(0, 0, 0);
        final int sliceStrideOther = y.sliceStride();
        final int rowStrideOther = y.rowStride();
        final int columnStrideOther = y.columnStride();
        final Object[] elemsOther = (Object[]) y.elements();
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

    public Object[] elements() {
        return elements;
    }

    /**
     * Returns the matrix cell value at coordinate <tt>[slice,row,column]</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @return the value at the specified coordinate.
     */

    public Object getQuick(int slice, int row, int column) {
        // if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows ||
        // column<0 || column>=columns) throw new
        // IndexOutOfBoundsException("slice:"+slice+", row:"+row+",
        // column:"+column);
        // return elements[index(slice,row,column)];
        // manually inlined:
        return elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column
                * columnStride];
    }

    /**
     * Returns <tt>true</tt> if both matrices share common cells. More formally,
     * returns <tt>true</tt> if <tt>other != null</tt> and at least one of the
     * following conditions is met
     * <ul>
     * <li>the receiver is a view of the other matrix
     * <li>the other matrix is a view of the receiver
     * <li><tt>this == other</tt>
     * </ul>
     */

    protected boolean haveSharedCellsRaw(ObjectMatrix3D other) {
        if (other instanceof SelectedDenseObjectMatrix3D) {
            SelectedDenseObjectMatrix3D otherMatrix = (SelectedDenseObjectMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseObjectMatrix3D) {
            DenseObjectMatrix3D otherMatrix = (DenseObjectMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Returns the position of the given coordinate within the (virtual or
     * non-virtual) internal 1-dimensional array.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the third-coordinate.
     */

    public long index(int slice, int row, int column) {
        // return _sliceOffset(_sliceRank(slice)) + _rowOffset(_rowRank(row)) +
        // _columnOffset(_columnRank(column));
        // manually inlined:
        return sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of slices, rows and columns.
     * For example, if the receiver is an instance of type
     * <tt>DenseObjectMatrix3D</tt> the new matrix must also be of type
     * <tt>DenseObjectMatrix3D</tt>, if the receiver is an instance of type
     * <tt>SparseObjectMatrix3D</tt> the new matrix must also be of type
     * <tt>SparseObjectMatrix3D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */

    public ObjectMatrix3D like(int slices, int rows, int columns) {
        return new DenseObjectMatrix3D(slices, rows, columns);
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseObjectMatrix3D</tt> the new matrix must also be
     * of type <tt>DenseObjectMatrix2D</tt>, if the receiver is an instance of
     * type <tt>SparseObjectMatrix3D</tt> the new matrix must also be of type
     * <tt>SparseObjectMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
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
     * @return a new matrix of the corresponding dynamic type.
     */

    protected ObjectMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
        return new DenseObjectMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride, true);
    }

    /**
     * Sets the matrix cell at coordinate <tt>[slice,row,column]</tt> to the
     * specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param value
     *            the value to be filled into the specified cell.
     */

    public void setQuick(int slice, int row, int column, Object value) {
        // if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows ||
        // column<0 || column>=columns) throw new
        // IndexOutOfBoundsException("slice:"+slice+", row:"+row+",
        // column:"+column);
        // elements[index(slice,row,column)] = value;
        // manually inlined:
        elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride] = value;
    }
    
    public ObjectMatrix1D vectorize() {
        ObjectMatrix1D v = new DenseObjectMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param sliceOffsets
     *            the offsets of the visible elements.
     * @param rowOffsets
     *            the offsets of the visible elements.
     * @param columnOffsets
     *            the offsets of the visible elements.
     * @return a new view.
     */

    protected ObjectMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseObjectMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
    }

    @Override
    public ObjectMatrix2D like2D(int rows, int columns) {
        return new DenseObjectMatrix2D(rows, columns);
    }
}
