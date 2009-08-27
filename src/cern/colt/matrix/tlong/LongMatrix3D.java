/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.matrix.AbstractMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Abstract base class for 3-d matrices holding <tt>int</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * A matrix has a number of slices, rows and columns, which are assigned upon
 * instance construction - The matrix's size is then
 * <tt>slices()*rows()*columns()</tt>. Elements are accessed via
 * <tt>[slice,row,column]</tt> coordinates. Legal coordinates range from
 * <tt>[0,0,0]</tt> to <tt>[slices()-1,rows()-1,columns()-1]</tt>. Any attempt
 * to access an element at a coordinate
 * <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>
 * will throw an <tt>IndexOutOfBoundsException</tt>.
 * <p>
 * <b>Note</b> that this implementation is not synchronized.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public abstract class LongMatrix3D extends AbstractMatrix3D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected LongMatrix3D() {
    }

    /**
     * Applies a function to each cell and aggregates the results. Returns a
     * value <tt>v</tt> such that <tt>v==a(size())</tt> where
     * <tt>a(i) == aggr( a(i-1), f(get(slice,row,column)) )</tt> and terminators
     * are <tt>a(1) == f(get(0,0,0)), a(0)==Long.NaN</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   cern.jet.math.Functions F = cern.jet.math.Functions.functions;
     *   2 x 2 x 2 matrix
     *   0 1
     *   2 3
     * 
     *   4 5
     *   6 7
     * 
     *   // Sum( x[slice,row,col]*x[slice,row,col] ) 
     *   matrix.aggregate(F.plus,F.square);
     *   --&gt; 140
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell value.
     * @param f
     *            a function transforming the current cell value.
     * @return the aggregated measure.
     * @see cern.jet.math.tlong.LongFunctions
     */
    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        long a = 0;
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
                        long a = f.apply(getQuick(firstSlice, 0, 0));
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    a = aggr.apply(a, f.apply(getQuick(s, r, c)));
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
            a = f.apply(getQuick(0, 0, 0));
            int d = 1; // first cell already done
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        a = aggr.apply(a, f.apply(getQuick(s, r, c)));
                    }
                    d = 0;
                }
            }
        }
        return a;
    }

    /**
     * Applies a function to each cell that satisfies a condition and aggregates
     * the results.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell value.
     * @param f
     *            a function transforming the current cell value.
     * @param cond
     *            a condition.
     * @return the aggregated measure.
     * @see cern.jet.math.tlong.LongFunctions
     */
    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f, final cern.colt.function.tlong.LongProcedure cond) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        long a = 0;
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
                        long elem = getQuick(firstSlice, 0, 0);
                        long a = 0;
                        if (cond.apply(elem) == true) {
                            a = aggr.apply(a, f.apply(elem));
                        }
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    elem = getQuick(s, r, c);
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
            long elem = getQuick(0, 0, 0);
            if (cond.apply(elem) == true) {
                a = aggr.apply(a, f.apply(elem));
            }
            int d = 1;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        elem = getQuick(s, r, c);
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

    /**
     * Applies a function to all cells with a given indexes and aggregates the
     * results.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell value.
     * @param f
     *            a function transforming the current cell value.
     * @param sliceList
     *            slice indexes.
     * @param rowList
     *            row indexes.
     * @param columnList
     *            column indexes.
     * @return the aggregated measure.
     * @see cern.jet.math.tlong.LongFunctions
     */
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
                        long a = f.apply(getQuick(sliceElements[firstIdx], rowElements[firstIdx],
                                columnElements[firstIdx]));
                        long elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            elem = getQuick(sliceElements[i], rowElements[i], columnElements[i]);
                            a = aggr.apply(a, f.apply(elem));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(getQuick(sliceElements[0], rowElements[0], columnElements[0]));
            long elem;
            for (int i = 1; i < size; i++) {
                elem = getQuick(sliceElements[i], rowElements[i], columnElements[i]);
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    /**
     * Applies a function to each corresponding cell of two matrices and
     * aggregates the results. Returns a value <tt>v</tt> such that
     * <tt>v==a(size())</tt> where
     * <tt>a(i) == aggr( a(i-1), f(get(slice,row,column),other.get(slice,row,column)) )</tt>
     * and terminators are
     * <tt>a(1) == f(get(0,0,0),other.get(0,0,0)), a(0)==Long.NaN</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   cern.jet.math.Functions F = cern.jet.math.Functions.functions;
     *   x = 2 x 2 x 2 matrix
     *   0 1
     *   2 3
     * 
     *   4 5
     *   6 7
     * 
     *   y = 2 x 2 x 2 matrix
     *   0 1
     *   2 3
     * 
     *   4 5
     *   6 7
     * 
     *   // Sum( x[slice,row,col] * y[slice,row,col] ) 
     *   x.aggregate(y, F.plus, F.mult);
     *   --&gt; 140
     * 
     *   // Sum( (x[slice,row,col] + y[slice,row,col])&circ;2 )
     *   x.aggregate(y, F.plus, F.chain(F.square,F.plus));
     *   --&gt; 560
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell values.
     * @param f
     *            a function transforming the current cell values.
     * @return the aggregated measure.
     * @throws IllegalArgumentException
     *             if
     *             <tt>slices() != other.slices() || rows() != other.rows() || columns() != other.columns()</tt>
     * @see cern.jet.math.tlong.LongFunctions
     */
    public long aggregate(final LongMatrix3D other, final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongLongFunction f) {
        checkShape(other);
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        long a = 0;
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
                        long a = f.apply(getQuick(firstSlice, 0, 0), other.getQuick(firstSlice, 0, 0));
                        int d = 1;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    a = aggr.apply(a, f.apply(getQuick(s, r, c), other.getQuick(s, r, c)));
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
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        a = aggr.apply(a, f.apply(getQuick(s, r, c), other.getQuick(s, r, c)));
                    }
                    d = 0;
                }
            }
        }
        return a;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[slice,row,col] = function(x[slice,row,col])</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   matrix = 1 x 2 x 2 matrix
     *   0.5 1.5      
     *   2.5 3.5
     * 
     *   // change each cell to its sine
     *   matrix.assign(cern.jet.math.Functions.sin);
     *   --&gt;
     *   1 x 2 x 2 matrix
     *   0.479426  0.997495 
     *   0.598472 -0.350783
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param function
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tlong.LongFunctions
     */
    public LongMatrix3D assign(final cern.colt.function.tlong.LongFunction function) {
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, function.apply(getQuick(s, r, c)));
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);

        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, function.apply(getQuick(s, r, c)));
                    }
                }
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>value</tt>.
     * 
     * @param value
     *            the value to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     */
    public LongMatrix3D assign(final long value) {
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, value);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, value);
                    }
                }
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form <tt>values[slice*row*column]</tt>.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != slices()*rows()*columns()</tt>
     */
    public LongMatrix3D assign(final long[] values) {
        if (values.length != slices * rows * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length
                    + "slices()*rows()*columns()=" + slices() * rows() * columns());
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
                        int idx = firstSlice * rows * columns;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, values[idx++]);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = 0;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, values[idx++]);
                    }
                }
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form <tt>values[slice*row*column]</tt>.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != slices()*rows()*columns()</tt>
     */
    public LongMatrix3D assign(final int[] values) {
        if (values.length != slices * rows * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length
                    + "slices()*rows()*columns()=" + slices() * rows() * columns());
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
                        int idx = firstSlice * rows * columns;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, values[idx++]);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = 0;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, values[idx++]);
                    }
                }
            }
        }
        return this;
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
    public LongMatrix3D assign(final long[][][] values) {
        if (values.length != slices)
            throw new IllegalArgumentException("Must have same number of slices: slices=" + values.length + "slices()="
                    + slices());
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
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, currentRow[c]);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);

        } else {
            for (int s = 0; s < slices; s++) {
                long[][] currentSlice = values[s];
                if (currentSlice.length != rows)
                    throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                            + currentSlice.length + "rows()=" + rows());
                for (int r = 0; r < rows; r++) {
                    long[] currentRow = currentSlice[r];
                    if (currentRow.length != columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "columns()=" + columns());
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, currentRow[c]);
                    }
                }
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to all cells that satisfy a condition.
     * 
     * @param cond
     *            a condition.
     * 
     * @param f
     *            a function object.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tlong.LongFunctions
     */
    public LongMatrix3D assign(final cern.colt.function.tlong.LongProcedure cond,
            final cern.colt.function.tlong.LongFunction f) {
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    elem = getQuick(s, r, c);
                                    if (cond.apply(elem) == true) {
                                        setQuick(s, r, c, f.apply(elem));
                                    }
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            long elem;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        elem = getQuick(s, r, c);
                        if (cond.apply(elem) == true) {
                            setQuick(s, r, c, f.apply(elem));
                        }
                    }
                }
            }
        }
        return this;
    }

    /**
     * Assigns a value to all cells that satisfy a condition.
     * 
     * @param cond
     *            a condition.
     * 
     * @param value
     *            a value.
     * @return <tt>this</tt> (for convenience only).
     * 
     */
    public LongMatrix3D assign(final cern.colt.function.tlong.LongProcedure cond, final long value) {
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    elem = getQuick(s, r, c);
                                    if (cond.apply(elem) == true) {
                                        setQuick(s, r, c, value);
                                    }
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            long elem;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        elem = getQuick(s, r, c);
                        if (cond.apply(elem) == true) {
                            setQuick(s, r, c, value);
                        }
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
     * @param other
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>slices() != other.slices() || rows() != other.rows() || columns() != other.columns()</tt>
     */
    public LongMatrix3D assign(LongMatrix3D other) {
        if (other == this)
            return this;
        checkShape(other);
        final LongMatrix3D otherLoc;
        if (haveSharedCells(other)) {
            otherLoc = other.copy();
        } else {
            otherLoc = other;
        }
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, otherLoc.getQuick(s, r, c));
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, otherLoc.getQuick(s, r, c));
                    }
                }
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[row,col] = function(x[row,col],y[row,col])</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   // assign x[row,col] = x[row,col]&lt;sup&gt;y[row,col]&lt;/sup&gt;
     *   m1 = 1 x 2 x 2 matrix 
     *   0 1 
     *   2 3
     * 
     *   m2 = 1 x 2 x 2 matrix 
     *   0 2 
     *   4 6
     * 
     *   m1.assign(m2, cern.jet.math.Functions.pow);
     *   --&gt;
     *   m1 == 1 x 2 x 2 matrix
     *   1   1 
     *   16 729
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param y
     *            the secondary matrix to operate on.
     * @param function
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>slices() != other.slices() || rows() != other.rows() || columns() != other.columns()</tt>
     * @see cern.jet.math.tlong.LongFunctions
     */
    public LongMatrix3D assign(final LongMatrix3D y, final cern.colt.function.tlong.LongLongFunction function) {
        checkShape(y);
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, function.apply(getQuick(s, r, c), y.getQuick(s, r, c)));
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, function.apply(getQuick(s, r, c), y.getQuick(s, r, c)));
                    }
                }
            }
        }

        return this;
    }

    /**
     * Assigns the result of a function to all cells with a given indexes
     * 
     * @param y
     *            the secondary matrix to operate on.
     * @param function
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>, *
     * @param sliceList
     *            slice indexes.
     * @param rowList
     *            row indexes.
     * @param columnList
     *            column indexes.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>slices() != other.slices() || rows() != other.rows() || columns() != other.columns()</tt>
     * @see cern.jet.math.tlong.LongFunctions
     */
    public LongMatrix3D assign(final LongMatrix3D y, final cern.colt.function.tlong.LongLongFunction function,
            final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList) {
        checkShape(y);
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
                            setQuick(sliceElements[i], rowElements[i], columnElements[i], function.apply(getQuick(
                                    sliceElements[i], rowElements[i], columnElements[i]), y.getQuick(sliceElements[i],
                                    rowElements[i], columnElements[i])));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(sliceElements[i], rowElements[i], columnElements[i], function.apply(getQuick(sliceElements[i],
                        rowElements[i], columnElements[i]), y.getQuick(sliceElements[i], rowElements[i],
                        columnElements[i])));
            }
        }
        return this;
    }

    /**
     * Returns the number of cells having non-zero values; ignores tolerance.
     * 
     * @return the number of cells having non-zero values.
     */
    public int cardinality() {
        int cardinality = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    if (getQuick(s, r, c) != 0)
                                        cardinality++;
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
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        if (getQuick(s, r, c) != 0)
                            cardinality++;
                    }
                }

            }
        }
        return cardinality;
    }

    /**
     * Constructs and returns a deep copy of the receiver.
     * <p>
     * <b>Note that the returned matrix is an independent deep copy.</b> The
     * returned matrix is not backed by this matrix, so changes in the returned
     * matrix are not reflected in this matrix, and vice-versa.
     * 
     * @return a deep copy of the receiver.
     */
    public LongMatrix3D copy() {
        return like().assign(this);
    }

    /**
     * Returns the elements of this matrix.
     * 
     * @return the elements
     */
    public abstract Object elements();

    /**
     * Returns whether all cells are equal to the given value.
     * 
     * @param value
     *            the value to test against.
     * @return <tt>true</tt> if all cells are equal to the given value,
     *         <tt>false</tt> otherwise.
     */
    public boolean equals(long value) {
        return cern.colt.matrix.tlong.algo.LongProperty.DEFAULT.equals(this, value);
    }

    /**
     * Compares this object against the specified object. The result is
     * <code>true</code> if and only if the argument is not <code>null</code>
     * and is at least a <code>LongMatrix3D</code> object that has the same
     * number of slices, rows and columns as the receiver and has exactly the
     * same values at the same coordinates.
     * 
     * @param obj
     *            the object to compare with.
     * @return <code>true</code> if the objects are the same; <code>false</code>
     *         otherwise.
     */

    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (!(obj instanceof LongMatrix3D))
            return false;

        return cern.colt.matrix.tlong.algo.LongProperty.DEFAULT.equals(this, (LongMatrix3D) obj);
    }

    /**
     * Returns the matrix cell value at coordinate <tt>[slice,row,column]</tt>.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @return the value of the specified cell.
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>
     *             .
     */
    public long get(int slice, int row, int column) {
        if (slice < 0 || slice >= slices || row < 0 || row >= rows || column < 0 || column >= columns)
            throw new IndexOutOfBoundsException("slice:" + slice + ", row:" + row + ", column:" + column);
        return getQuick(slice, row, column);
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     */
    protected LongMatrix3D getContent() {
        return this;
    }

    /**
     * Fills the coordinates and values of cells having negative values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * 
     * @param sliceList
     *            the list to be filled with slice indexes, can have any size.
     * @param rowList
     *            the list to be filled with row indexes, can have any size.
     * @param columnList
     *            the list to be filled with column indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getNegativeValues(final IntArrayList sliceList, final IntArrayList rowList,
            final IntArrayList columnList, final LongArrayList valueList) {
        sliceList.clear();
        rowList.clear();
        columnList.clear();
        valueList.clear();

        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    long value = getQuick(s, r, c);
                    if (value < 0) {
                        sliceList.add(s);
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(value);
                    }
                }
            }
        }

    }

    /**
     * Fills the coordinates and values of cells having non-zero values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * <p>
     * In general, fill order is <i>unspecified</i>. This implementation fill
     * like:
     * <tt>for (slice = 0..slices-1) for (row = 0..rows-1) for (column = 0..colums-1) do ... </tt>
     * . However, subclasses are free to us any other order, even an order that
     * may change over time as cell values are changed. (Of course, result lists
     * indexes are guaranteed to correspond to the same cell). For an example,
     * see
     * {@link LongMatrix3D#getNonZeros(IntArrayList,IntArrayList,IntArrayList,LongArrayList)}.
     * 
     * @param sliceList
     *            the list to be filled with slice indexes, can have any size.
     * @param rowList
     *            the list to be filled with row indexes, can have any size.
     * @param columnList
     *            the list to be filled with column indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getNonZeros(final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList,
            final LongArrayList valueList) {
        sliceList.clear();
        rowList.clear();
        columnList.clear();
        valueList.clear();

        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    long value = getQuick(s, r, c);
                    if (value != 0) {
                        sliceList.add(s);
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(value);
                    }
                }
            }

        }
    }

    /**
     * Fills the coordinates and values of cells having positive values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * 
     * @param sliceList
     *            the list to be filled with slice indexes, can have any size.
     * @param rowList
     *            the list to be filled with row indexes, can have any size.
     * @param columnList
     *            the list to be filled with column indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getPositiveValues(final IntArrayList sliceList, final IntArrayList rowList,
            final IntArrayList columnList, final LongArrayList valueList) {
        sliceList.clear();
        rowList.clear();
        columnList.clear();
        valueList.clear();

        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    long value = getQuick(s, r, c);
                    if (value > 0) {
                        sliceList.add(s);
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(value);
                    }
                }
            }
        }
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
    public abstract long getQuick(int slice, int row, int column);

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */
    protected boolean haveSharedCells(LongMatrix3D other) {
        if (other == null)
            return false;
        if (this == other)
            return true;
        return getContent().haveSharedCellsRaw(other.getContent());
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */
    protected boolean haveSharedCellsRaw(LongMatrix3D other) {
        return false;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the same number of slices, rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseLongMatrix3D</tt> the new matrix must also be of type
     * <tt>DenseLongMatrix3D</tt>, if the receiver is an instance of type
     * <tt>SparseLongMatrix3D</tt> the new matrix must also be of type
     * <tt>SparseLongMatrix3D</tt>, etc. In general, the new matrix should have
     * internal parametrization as similar as possible.
     * 
     * @return a new empty matrix of the same dynamic type.
     */
    public LongMatrix3D like() {
        return like(slices, rows, columns);
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of slices, rows and columns.
     * For example, if the receiver is an instance of type
     * <tt>DenseLongMatrix3D</tt> the new matrix must also be of type
     * <tt>DenseLongMatrix3D</tt>, if the receiver is an instance of type
     * <tt>SparseLongMatrix3D</tt> the new matrix must also be of type
     * <tt>SparseLongMatrix3D</tt>, etc. In general, the new matrix should have
     * internal parametrization as similar as possible.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public abstract LongMatrix3D like(int slices, int rows, int columns);

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseDoubleMatrix3D</tt> the new matrix must also be
     * of type <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of
     * type <tt>SparseDoubleMatrix3D</tt> the new matrix must also be of type
     * <tt>SparseDoubleMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public abstract LongMatrix2D like2D(int rows, int columns);

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseLongMatrix3D</tt> the new matrix must also be
     * of type <tt>DenseLongMatrix2D</tt>, if the receiver is an instance of
     * type <tt>SparseLongMatrix3D</tt> the new matrix must also be of type
     * <tt>SparseLongMatrix2D</tt>, etc.
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
    protected abstract LongMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
            int columnStride);

    /**
     * Return maximum value of this matrix together with its location
     * 
     * @return { maximum_value, slice_location, row_location, column_location };
     */
    public long[] getMaxLocation() {
        int sliceLocation = 0;
        int rowLocation = 0;
        int columnLocation = 0;
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
                        int sliceLocation = firstSlice;
                        int rowLocation = 0;
                        int columnLocation = 0;
                        long maxValue = getQuick(sliceLocation, 0, 0);
                        int d = 1;
                        long elem;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    elem = getQuick(s, r, c);
                                    if (maxValue < elem) {
                                        maxValue = elem;
                                        sliceLocation = s;
                                        rowLocation = r;
                                        columnLocation = c;
                                    }
                                }
                                d = 0;
                            }
                        }
                        return new long[] { maxValue, sliceLocation, rowLocation, columnLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                maxValue = results[0][0];
                sliceLocation = (int) results[0][1];
                rowLocation = (int) results[0][2];
                columnLocation = (int) results[0][3];
                for (int j = 1; j < nthreads; j++) {
                    if (maxValue < results[j][0]) {
                        maxValue = results[j][0];
                        sliceLocation = (int) results[j][1];
                        rowLocation = (int) results[j][2];
                        columnLocation = (int) results[j][3];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            maxValue = getQuick(0, 0, 0);
            long elem;
            int d = 1;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        elem = getQuick(s, r, c);
                        if (maxValue < elem) {
                            maxValue = elem;
                            sliceLocation = s;
                            rowLocation = r;
                            columnLocation = c;
                        }
                    }
                    d = 0;
                }
            }
        }
        return new long[] { maxValue, sliceLocation, rowLocation, columnLocation };
    }

    /**
     * Returns minimum value of this matrix together with its location
     * 
     * @return { minimum_value, slice_location, row_location, column_location };
     */
    public long[] getMinLocation() {
        int sliceLocation = 0;
        int rowLocation = 0;
        int columnLocation = 0;
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
                        int sliceLocation = firstSlice;
                        int rowLocation = 0;
                        int columnLocation = 0;
                        long minValue = getQuick(sliceLocation, 0, 0);
                        int d = 1;
                        long elem;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = d; c < columns; c++) {
                                    elem = getQuick(s, r, c);
                                    if (minValue > elem) {
                                        minValue = elem;
                                        sliceLocation = s;
                                        rowLocation = r;
                                        columnLocation = c;
                                    }
                                }
                                d = 0;
                            }
                        }
                        return new long[] { minValue, sliceLocation, rowLocation, columnLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                minValue = results[0][0];
                sliceLocation = (int) results[0][1];
                rowLocation = (int) results[0][2];
                columnLocation = (int) results[0][3];
                for (int j = 1; j < nthreads; j++) {
                    if (minValue > results[j][0]) {
                        minValue = results[j][0];
                        sliceLocation = (int) results[j][1];
                        rowLocation = (int) results[j][2];
                        columnLocation = (int) results[j][3];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            minValue = getQuick(0, 0, 0);
            long elem;
            int d = 1;
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = d; c < columns; c++) {
                        elem = getQuick(s, r, c);
                        if (minValue > elem) {
                            minValue = elem;
                            sliceLocation = s;
                            rowLocation = r;
                            columnLocation = c;
                        }
                    }
                    d = 0;
                }
            }
        }
        return new long[] { minValue, sliceLocation, rowLocation, columnLocation };
    }

    /**
     * Sets the matrix cell at coordinate <tt>[slice,row,column]</tt> to the
     * specified value.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param value
     *            the value to be filled into the specified cell.
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>row&lt;0 || row&gt;=rows() || slice&lt;0 || slice&gt;=slices() || column&lt;0 || column&gt;=column()</tt>
     *             .
     */
    public void set(int slice, int row, int column, long value) {
        if (slice < 0 || slice >= slices || row < 0 || row >= rows || column < 0 || column >= columns)
            throw new IndexOutOfBoundsException("slice:" + slice + ", row:" + row + ", column:" + column);
        setQuick(slice, row, column, value);
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
    public abstract void setQuick(int slice, int row, int column, long value);

    /**
     * Constructs and returns a 2-dimensional array containing the cell values.
     * The returned array <tt>values</tt> has the form
     * <tt>values[slice][row][column]</tt> and has the same number of slices,
     * rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @return an array filled with the values of the cells.
     */
    public long[][][] toArray() {
        final long[][][] values = new long[slices][rows][columns];
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
                            long[][] currentSlice = values[s];
                            for (int r = 0; r < rows; r++) {
                                long[] currentRow = currentSlice[r];
                                for (int c = 0; c < columns; c++) {
                                    currentRow[c] = getQuick(s, r, c);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                long[][] currentSlice = values[s];
                for (int r = 0; r < rows; r++) {
                    long[] currentRow = currentSlice[r];
                    for (int c = 0; c < columns; c++) {
                        currentRow[c] = getQuick(s, r, c);
                    }
                }
            }
        }
        return values;
    }

    /**
     * Returns a string representation using default formatting.
     * 
     * @see cern.colt.matrix.tlong.algo.LongFormatter
     */

    public String toString() {
        return new cern.colt.matrix.tlong.algo.LongFormatter().toString(this);
    }

    /**
     * Returns a vector obtained by stacking the columns of each slice of the
     * matrix on top of one another.
     * 
     * @return a vector obtained by stacking the columns of each slice of the
     *         matrix on top of one another.
     */
    public abstract LongMatrix1D vectorize();

    /**
     * Constructs and returns a new view equal to the receiver. The view is a
     * shallow clone. Calls <code>clone()</code> and casts the result.
     * <p>
     * <b>Note that the view is not a deep copy.</b> The returned matrix is
     * backed by this matrix, so changes in the returned matrix are reflected in
     * this matrix, and vice-versa.
     * <p>
     * Use {@link #copy()} if you want to construct an independent deep copy
     * rather than a new view.
     * 
     * @return a new view of the receiver.
     */
    protected LongMatrix3D view() {
        return (LongMatrix3D) clone();
    }

    /**
     * Constructs and returns a new 2-dimensional <i>slice view</i> representing
     * the slices and rows of the given column. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * <p>
     * To obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>view().part(...)</tt>), then apply this method to the sub-range view.
     * To obtain 1-dimensional views, apply this method, then apply another
     * slice view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the
     * intermediate 2-dimensional view. To obtain 1-dimensional views on
     * subranges, apply both steps.
     * 
     * @param column
     *            the index of the column to fix.
     * @return a new 2-dimensional slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>column < 0 || column >= columns()</tt>.
     * @see #viewSlice(int)
     * @see #viewRow(int)
     */
    public LongMatrix2D viewColumn(int column) {
        checkColumn(column);
        int sliceRows = this.slices;
        int sliceColumns = this.rows;

        // int sliceOffset = index(0,0,column);
        int sliceRowZero = sliceZero;
        int sliceColumnZero = rowZero + _columnOffset(_columnRank(column));

        int sliceRowStride = this.sliceStride;
        int sliceColumnStride = this.rowStride;
        return like2D(sliceRows, sliceColumns, sliceRowZero, sliceColumnZero, sliceRowStride, sliceColumnStride);
    }

    /**
     * Constructs and returns a new <i>flip view</i> along the column axis. What
     * used to be column <tt>0</tt> is now column <tt>columns()-1</tt>, ...,
     * what used to be column <tt>columns()-1</tt> is now column <tt>0</tt>. The
     * returned view is backed by this matrix, so changes in the returned view
     * are reflected in this matrix, and vice-versa.
     * 
     * @return a new flip view.
     * @see #viewSliceFlip()
     * @see #viewRowFlip()
     */
    public LongMatrix3D viewColumnFlip() {
        return (LongMatrix3D) (view().vColumnFlip());
    }

    /**
     * Constructs and returns a new <i>dice view</i>; Swaps dimensions (axes);
     * Example: 3 x 4 x 5 matrix --> 4 x 3 x 5 matrix. The view has dimensions
     * exchanged; what used to be one axis is now another, in all desired
     * permutations. The returned view is backed by this matrix, so changes in
     * the returned view are reflected in this matrix, and vice-versa.
     * 
     * @param axis0
     *            the axis that shall become axis 0 (legal values 0..2).
     * @param axis1
     *            the axis that shall become axis 1 (legal values 0..2).
     * @param axis2
     *            the axis that shall become axis 2 (legal values 0..2).
     * @return a new dice view.
     * @throws IllegalArgumentException
     *             if some of the parameters are equal or not in range 0..2.
     */
    public LongMatrix3D viewDice(int axis0, int axis1, int axis2) {
        return (LongMatrix3D) (view().vDice(axis0, axis1, axis2));
    }

    /**
     * Constructs and returns a new <i>sub-range view</i> that is a
     * <tt>depth x height x width</tt> sub matrix starting at
     * <tt>[slice,row,column]</tt>; Equivalent to
     * <tt>view().part(slice,row,column,depth,height,width)</tt>; Provided for
     * convenience only. The returned view is backed by this matrix, so changes
     * in the returned view are reflected in this matrix, and vice-versa.
     * 
     * @param slice
     *            The index of the slice-coordinate.
     * @param row
     *            The index of the row-coordinate.
     * @param column
     *            The index of the column-coordinate.
     * @param depth
     *            The depth of the box.
     * @param height
     *            The height of the box.
     * @param width
     *            The width of the box.
     * @throws IndexOutOfBoundsException
     *             if
     * 
     *             <tt>slice<0 || depth<0 || slice+depth>slices() || row<0 || height<0 || row+height>rows() || column<0 || width<0 || column+width>columns()</tt>
     * @return the new view.
     * 
     */
    public LongMatrix3D viewPart(int slice, int row, int column, int depth, int height, int width) {
        return (LongMatrix3D) (view().vPart(slice, row, column, depth, height, width));
    }

    /**
     * Constructs and returns a new 2-dimensional <i>slice view</i> representing
     * the slices and columns of the given row. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * <p>
     * To obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>view().part(...)</tt>), then apply this method to the sub-range view.
     * To obtain 1-dimensional views, apply this method, then apply another
     * slice view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the
     * intermediate 2-dimensional view. To obtain 1-dimensional views on
     * subranges, apply both steps.
     * 
     * @param row
     *            the index of the row to fix.
     * @return a new 2-dimensional slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>row < 0 || row >= row()</tt>.
     * @see #viewSlice(int)
     * @see #viewColumn(int)
     */
    public LongMatrix2D viewRow(int row) {
        checkRow(row);
        int sliceRows = this.slices;
        int sliceColumns = this.columns;

        // int sliceOffset = index(0,row,0);
        int sliceRowZero = sliceZero;
        int sliceColumnZero = columnZero + _rowOffset(_rowRank(row));

        int sliceRowStride = this.sliceStride;
        int sliceColumnStride = this.columnStride;
        return like2D(sliceRows, sliceColumns, sliceRowZero, sliceColumnZero, sliceRowStride, sliceColumnStride);
    }

    /**
     * Constructs and returns a new <i>flip view</i> along the row axis. What
     * used to be row <tt>0</tt> is now row <tt>rows()-1</tt>, ..., what used to
     * be row <tt>rows()-1</tt> is now row <tt>0</tt>. The returned view is
     * backed by this matrix, so changes in the returned view are reflected in
     * this matrix, and vice-versa.
     * 
     * @return a new flip view.
     * @see #viewSliceFlip()
     * @see #viewColumnFlip()
     */
    public LongMatrix3D viewRowFlip() {
        return (LongMatrix3D) (view().vRowFlip());
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding all <b>slices</b> matching the given condition. Applies the
     * condition to each slice and takes only those where
     * <tt>condition.apply(viewSlice(i))</tt> yields <tt>true</tt>. To match
     * rows or columns, use a dice view.
     * <p>
     * <b>Example:</b> <br>
     * 
     * <pre>
     * // extract and view all slices which have an aggregate sum &gt; 1000
     * matrix.viewSelection(new LongMatrix2DProcedure() {
     *     public final boolean apply(LongMatrix2D m) {
     *         return m.zSum &gt; 1000;
     *     }
     * });
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>. The returned
     * view is backed by this matrix, so changes in the returned view are
     * reflected in this matrix, and vice-versa.
     * 
     * @param condition
     *            The condition to be matched.
     * @return the new view.
     */
    public LongMatrix3D viewSelection(LongMatrix2DProcedure condition) {
        IntArrayList matches = new IntArrayList();
        for (int i = 0; i < slices; i++) {
            if (condition.apply(viewSlice(i)))
                matches.add(i);
        }

        matches.trimToSize();
        return viewSelection(matches.elements(), null, null); // take all rows
        // and columns
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding the indicated cells. There holds
     * 
     * <tt>view.slices() == sliceIndexes.length, view.rows() == rowIndexes.length, view.columns() == columnIndexes.length</tt>
     * and
     * <tt>view.get(k,i,j) == this.get(sliceIndexes[k],rowIndexes[i],columnIndexes[j])</tt>
     * . Indexes can occur multiple times and can be in arbitrary order. For an
     * example see {@link LongMatrix2D#viewSelection(int[],int[])}.
     * <p>
     * Note that modifying the index arguments after this call has returned has
     * no effect on the view. The returned view is backed by this matrix, so
     * changes in the returned view are reflected in this matrix, and
     * vice-versa.
     * 
     * @param sliceIndexes
     *            The slices of the cells that shall be visible in the new view.
     *            To indicate that <i>all</i> slices shall be visible, simply
     *            set this parameter to <tt>null</tt>.
     * @param rowIndexes
     *            The rows of the cells that shall be visible in the new view.
     *            To indicate that <i>all</i> rows shall be visible, simply set
     *            this parameter to <tt>null</tt>.
     * @param columnIndexes
     *            The columns of the cells that shall be visible in the new
     *            view. To indicate that <i>all</i> columns shall be visible,
     *            simply set this parameter to <tt>null</tt>.
     * @return the new view.
     * @throws IndexOutOfBoundsException
     *             if <tt>!(0 <= sliceIndexes[i] < slices())</tt> for any
     *             <tt>i=0..sliceIndexes.length()-1</tt>.
     * @throws IndexOutOfBoundsException
     *             if <tt>!(0 <= rowIndexes[i] < rows())</tt> for any
     *             <tt>i=0..rowIndexes.length()-1</tt>.
     * @throws IndexOutOfBoundsException
     *             if <tt>!(0 <= columnIndexes[i] < columns())</tt> for any
     *             <tt>i=0..columnIndexes.length()-1</tt>.
     */
    public LongMatrix3D viewSelection(int[] sliceIndexes, int[] rowIndexes, int[] columnIndexes) {
        // check for "all"
        if (sliceIndexes == null) {
            sliceIndexes = new int[slices];
            for (int i = 0; i < slices; i++)
                sliceIndexes[i] = i;
        }
        if (rowIndexes == null) {
            rowIndexes = new int[rows];
            for (int i = 0; i < rows; i++)
                rowIndexes[i] = i;
        }
        if (columnIndexes == null) {
            columnIndexes = new int[columns];
            for (int i = 0; i < columns; i++)
                columnIndexes[i] = i;
        }

        checkSliceIndexes(sliceIndexes);
        checkRowIndexes(rowIndexes);
        checkColumnIndexes(columnIndexes);

        int[] sliceOffsets = new int[sliceIndexes.length];
        int[] rowOffsets = new int[rowIndexes.length];
        int[] columnOffsets = new int[columnIndexes.length];

        for (int i = 0; i < sliceIndexes.length; i++) {
            sliceOffsets[i] = _sliceOffset(_sliceRank(sliceIndexes[i]));
        }
        for (int i = 0; i < rowIndexes.length; i++) {
            rowOffsets[i] = _rowOffset(_rowRank(rowIndexes[i]));
        }
        for (int i = 0; i < columnIndexes.length; i++) {
            columnOffsets[i] = _columnOffset(_columnRank(columnIndexes[i]));
        }

        return viewSelectionLike(sliceOffsets, rowOffsets, columnOffsets);
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
    protected abstract LongMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets);

    /**
     * Constructs and returns a new 2-dimensional <i>slice view</i> representing
     * the rows and columns of the given slice. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * <p>
     * To obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>view().part(...)</tt>), then apply this method to the sub-range view.
     * To obtain 1-dimensional views, apply this method, then apply another
     * slice view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the
     * intermediate 2-dimensional view. To obtain 1-dimensional views on
     * subranges, apply both steps.
     * 
     * @param slice
     *            the index of the slice to fix.
     * @return a new 2-dimensional slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>slice < 0 || slice >= slices()</tt>.
     * @see #viewRow(int)
     * @see #viewColumn(int)
     */
    public LongMatrix2D viewSlice(int slice) {
        checkSlice(slice);
        int sliceRows = this.rows;
        int sliceColumns = this.columns;

        // int sliceOffset = index(slice,0,0);
        int sliceRowZero = rowZero;
        int sliceColumnZero = columnZero + _sliceOffset(_sliceRank(slice));

        int sliceRowStride = this.rowStride;
        int sliceColumnStride = this.columnStride;
        return like2D(sliceRows, sliceColumns, sliceRowZero, sliceColumnZero, sliceRowStride, sliceColumnStride);
    }

    /**
     * Constructs and returns a new <i>flip view</i> along the slice axis. What
     * used to be slice <tt>0</tt> is now slice <tt>slices()-1</tt>, ..., what
     * used to be slice <tt>slices()-1</tt> is now slice <tt>0</tt>. The
     * returned view is backed by this matrix, so changes in the returned view
     * are reflected in this matrix, and vice-versa.
     * 
     * @return a new flip view.
     * @see #viewRowFlip()
     * @see #viewColumnFlip()
     */
    public LongMatrix3D viewSliceFlip() {
        return (LongMatrix3D) (view().vSliceFlip());
    }

    /**
     * Sorts the matrix slices into ascending order, according to the <i>natural
     * ordering</i> of the matrix values in the given <tt>[row,column]</tt>
     * position. This sort is guaranteed to be <i>stable</i>. For further
     * information, see
     * {@link cern.colt.matrix.tlong.algo.LongSorting#sort(LongMatrix3D,int,int)}
     * . For more advanced sorting functionality, see
     * {@link cern.colt.matrix.tlong.algo.LongSorting}.
     * 
     * @return a new sorted vector (matrix) view.
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>row < 0 || row >= rows() || column < 0 || column >= columns()</tt>
     *             .
     */
    public LongMatrix3D viewSorted(int row, int column) {
        return cern.colt.matrix.tlong.algo.LongSorting.mergeSort.sort(this, row, column);
    }

    /**
     * Constructs and returns a new <i>stride view</i> which is a sub matrix
     * consisting of every i-th cell. More specifically, the view has
     * <tt>this.slices()/sliceStride</tt> slices and
     * <tt>this.rows()/rowStride</tt> rows and
     * <tt>this.columns()/columnStride</tt> columns holding cells
     * <tt>this.get(k*sliceStride,i*rowStride,j*columnStride)</tt> for all
     * 
     * <tt>k = 0..slices()/sliceStride - 1, i = 0..rows()/rowStride - 1, j = 0..columns()/columnStride - 1</tt>
     * . The returned view is backed by this matrix, so changes in the returned
     * view are reflected in this matrix, and vice-versa.
     * 
     * @param sliceStride
     *            the slice step factor.
     * @param rowStride
     *            the row step factor.
     * @param columnStride
     *            the column step factor.
     * @return a new view.
     * @throws IndexOutOfBoundsException
     *             if <tt>sliceStride<=0 || rowStride<=0 || columnStride<=0</tt>
     *             .
     */
    public LongMatrix3D viewStrides(int sliceStride, int rowStride, int columnStride) {
        return (LongMatrix3D) (view().vStrides(sliceStride, rowStride, columnStride));
    }

    /**
     * Returns the sum of all cells; <tt>Sum( x[i,j,k] )</tt>.
     * 
     * @return the sum.
     */
    public long zSum() {
        if (size() == 0)
            return 0;
        return aggregate(cern.jet.math.tlong.LongFunctions.plus, cern.jet.math.tlong.LongFunctions.identity);
    }
}
