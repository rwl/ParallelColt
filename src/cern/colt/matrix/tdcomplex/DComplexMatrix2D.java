/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.AbstractMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplex;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Abstract base class for 2-d matrices holding <tt>complex</tt> elements.
 * 
 * A matrix has a number of rows and columns, which are assigned upon instance
 * construction - The matrix's size is then <tt>rows()*columns()</tt>. Elements
 * are accessed via <tt>[row,column]</tt> coordinates. Legal coordinates range
 * from <tt>[0,0]</tt> to <tt>[rows()-1,columns()-1]</tt>. Any attempt to access
 * an element at a coordinate
 * <tt>column&lt;0 || column&gt;=columns() || row&lt;0 || row&gt;=rows()</tt>
 * will throw an <tt>IndexOutOfBoundsException</tt>.
 * <p>
 * <b>Note</b> that this implementation is not synchronized.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public abstract class DComplexMatrix2D extends AbstractMatrix2D {
    private static final long serialVersionUID = 1L;

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected DComplexMatrix2D() {
    }

    /**
     * Applies a function to each cell and aggregates the results.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell value.
     * @param f
     *            a function transforming the current cell value.
     * @return the aggregated measure.
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public double[] aggregate(final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction aggr,
            final cern.colt.function.tdcomplex.DComplexDComplexFunction f) {
        double[] b = new double[2];
        if (size() == 0) {
            b[0] = Double.NaN;
            b[1] = Double.NaN;
            return b;
        }
        double[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        double[] a = f.apply(getQuick(firstRow, 0));
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                a = aggr.apply(a, f.apply(getQuick(r, c)));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(getQuick(0, 0));
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    a = aggr.apply(a, f.apply(getQuick(r, c)));
                }
                d = 0;
            }
        }
        return a;
    }

    /**
     * Applies a function to each corresponding cell of two matrices and
     * aggregates the results.
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
     *             <tt>columns() != other.columns() || rows() != other.rows()</tt>
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public double[] aggregate(final DComplexMatrix2D other,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction aggr,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction f) {
        checkShape(other);
        double[] b = new double[2];
        if (size() == 0) {
            b[0] = Double.NaN;
            b[1] = Double.NaN;
            return b;
        }
        double[] a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {

                    public double[] call() throws Exception {
                        double[] a = f.apply(getQuick(firstRow, 0), other.getQuick(firstRow, 0));
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                a = aggr.apply(a, f.apply(getQuick(r, c), other.getQuick(r, c)));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(getQuick(0, 0), other.getQuick(0, 0));
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    a = aggr.apply(a, f.apply(getQuick(r, c), other.getQuick(r, c)));
                }
                d = 0;
            }
        }
        return a;
    }

    /**
     * Assigns the result of a function to each cell;
     * 
     * @param f
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexDComplexFunction f) {
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                setQuick(r, c, f.apply(getQuick(r, c)));
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    setQuick(r, c, f.apply(getQuick(r, c)));
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
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond,
            final cern.colt.function.tdcomplex.DComplexDComplexFunction f) {
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
                        double[] elem;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                elem = getQuick(r, c);
                                if (cond.apply(elem) == true) {
                                    setQuick(r, c, f.apply(elem));
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] elem;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    elem = getQuick(r, c);
                    if (cond.apply(elem) == true) {
                        setQuick(r, c, f.apply(elem));
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
     *            a value (re=value[0], im=value[1]).
     * @return <tt>this</tt> (for convenience only).
     * 
     */
    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond, final double[] value) {
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
                        double[] elem;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                elem = getQuick(r, c);
                                if (cond.apply(elem) == true) {
                                    setQuick(r, c, value);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] elem;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    elem = getQuick(r, c);
                    if (cond.apply(elem) == true) {
                        setQuick(r, c, value);
                    }
                }
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to the real part of the receiver. The
     * imaginary part of the receiver is reset to zero.
     * 
     * @param f
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexRealFunction f) {
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                double re = f.apply(getQuick(r, c));
                                setQuick(r, c, re, 0);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double re = f.apply(getQuick(r, c));
                    setQuick(r, c, re, 0);
                }
            }
        }
        return this;
    }

    /**
     * Replaces all cell values of the receiver with the values of another
     * matrix. Both matrices must have the same number of rows and columns. If
     * both matrices share the same cells (as is the case if they are views
     * derived from the same matrix) and intersect in an ambiguous way, then
     * replaces <i>as if</i> using an intermediate auxiliary deep copy of
     * <tt>other</tt>.
     * 
     * @param other
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>columns() != other.columns() || rows() != other.rows()</tt>
     */
    public DComplexMatrix2D assign(DComplexMatrix2D other) {
        if (other == this)
            return this;
        checkShape(other);
        final DComplexMatrix2D otherLoc;
        if (haveSharedCells(other)) {
            otherLoc = other.copy();
        } else {
            otherLoc = other;
        }
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                setQuick(r, c, otherLoc.getQuick(r, c));
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    setQuick(r, c, otherLoc.getQuick(r, c));
                }
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to each cell.
     * 
     * @param y
     *            the secondary matrix to operate on.
     * @param f
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>columns() != other.columns() || rows() != other.rows()</tt>
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix2D assign(final DComplexMatrix2D y,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction f) {
        checkShape(y);
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                setQuick(r, c, f.apply(getQuick(r, c), y.getQuick(r, c)));
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    setQuick(r, c, f.apply(getQuick(r, c), y.getQuick(r, c)));
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
     *            cell's value of <tt>y</tt>,
     * @param rowList
     *            row indexes.
     * @param columnList
     *            column indexes.
     * 
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>columns() != other.columns() || rows() != other.rows()</tt>
     * @see cern.jet.math.tdouble.DoubleFunctions
     */
    public DComplexMatrix2D assign(final DComplexMatrix2D y,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction function, IntArrayList rowList,
            IntArrayList columnList) {
        checkShape(y);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(rowElements[i], columnElements[i], function.apply(getQuick(rowElements[i],
                                    columnElements[i]), y.getQuick(rowElements[i], columnElements[i])));
                        }
                    }

                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(rowElements[i], columnElements[i], function.apply(getQuick(rowElements[i], columnElements[i]),
                        y.getQuick(rowElements[i], columnElements[i])));
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>re</tt> and <tt>im</tt>.
     * 
     * @param re
     *            the real part of the value to be filled into the cells.
     * @param im
     *            the imaginary part of the value to be filled into the cells.
     * 
     * @return <tt>this</tt> (for convenience only).
     */
    public DComplexMatrix2D assign(final double re, final double im) {
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                setQuick(r, c, re, im);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    setQuick(r, c, re, im);
                }
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form
     * <tt>re = values[row*rowStride+column*columnStride]; im = values[row*rowStride+column*columnStride+1]</tt>
     * and have exactly the same number of rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != rows()*2*columns()</tt>.
     */
    public DComplexMatrix2D assign(final double[] values) {
        if (values.length != rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + "rows()*2*columns()="
                    + rows() * 2 * columns());
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
                        int idx = firstRow * columns * 2;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                setQuick(r, c, values[idx], values[idx + 1]);
                                idx += 2;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {

            int idx = 0;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    setQuick(r, c, values[idx], values[idx + 1]);
                    idx += 2;
                }
            }
        }

        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form
     * <tt>re = values[row][2*column]; im = values[row][2*column+1]</tt> and
     * have exactly the same number of rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>values.length != rows() || for any 0 &lt;= row &lt; rows(): values[row].length != 2*columns()</tt>
     *             .
     */
    public DComplexMatrix2D assign(final double[][] values) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (values.length != rows)
            throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "rows()="
                    + rows());
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            double[] currentRow = values[r];
                            if (currentRow.length != 2 * columns)
                                throw new IllegalArgumentException(
                                        "Must have same number of columns in every row: columns=" + currentRow.length
                                                + "2*columns()=" + 2 * columns());
                            for (int c = 0; c < columns; c++) {
                                setQuick(r, c, currentRow[2 * c], currentRow[2 * c + 1]);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                double[] currentRow = values[r];
                if (currentRow.length != 2 * columns)
                    throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                            + currentRow.length + "2*columns()=" + 2 * columns());
                for (int c = 0; c < columns; c++) {
                    setQuick(r, c, currentRow[2 * c], currentRow[2 * c + 1]);
                }
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form
     * <tt>re = values[row*rowStride+column*columnStride]; im = values[row*rowStride+column*columnStride+1]</tt>
     * and have exactly the same number of rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != rows()*2*columns()</tt>.
     */
    public DComplexMatrix2D assign(final float[] values) {
        if (values.length != rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + "rows()*2*columns()="
                    + rows() * 2 * columns());
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
                        int idx = firstRow * columns * 2;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                setQuick(r, c, values[idx], values[idx + 1]);
                                idx += 2;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {

            int idx = 0;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    setQuick(r, c, values[idx], values[idx + 1]);
                    idx += 2;
                }
            }
        }

        return this;
    }

    /**
     * Replaces imaginary part of the receiver with the values of another real
     * matrix. The real part of the receiver remains unchanged. Both matrices
     * must have the same size.
     * 
     * @param other
     *            the source matrix to copy from
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public DComplexMatrix2D assignImaginary(final DoubleMatrix2D other) {
        checkShape(other);
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                double re = getQuick(r, c)[0];
                                double im = other.getQuick(r, c);
                                setQuick(r, c, re, im);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double re = getQuick(r, c)[0];
                    double im = other.getQuick(r, c);
                    setQuick(r, c, re, im);
                }
            }
        }
        return this;
    }

    /**
     * Replaces real part of the receiver with the values of another real
     * matrix. The imaginary part of the receiver remains unchanged. Both
     * matrices must have the same size.
     * 
     * @param other
     *            the source matrix to copy from
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public DComplexMatrix2D assignReal(final DoubleMatrix2D other) {
        checkShape(other);
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                double re = other.getQuick(r, c);
                                double im = getQuick(r, c)[1];
                                setQuick(r, c, re, im);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double re = other.getQuick(r, c);
                    double im = getQuick(r, c)[1];
                    setQuick(r, c, re, im);
                }
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
                        double[] tmp = new double[2];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                tmp = getQuick(r, c);
                                if ((tmp[0] != 0.0) || (tmp[1] != 0.0))
                                    cardinality++;
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
            double[] tmp = new double[2];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    tmp = getQuick(r, c);
                    if (tmp[0] != 0 || tmp[1] != 0)
                        cardinality++;
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
    public DComplexMatrix2D copy() {
        return like().assign(this);
    }

    /**
     * Returns whether all cells are equal to the given value.
     * 
     * @param value
     *            the value to test against.
     * @return <tt>true</tt> if all cells are equal to the given value,
     *         <tt>false</tt> otherwise.
     */
    public boolean equals(double[] value) {
        return cern.colt.matrix.tdcomplex.algo.DComplexProperty.DEFAULT.equals(this, value);
    }

    /**
     * Compares this object against the specified object. The result is
     * <code>true</code> if and only if the argument is not <code>null</code>
     * and is at least a <code>DoubleMatrix2D</code> object that has the same
     * number of columns and rows as the receiver and has exactly the same
     * values at the same coordinates.
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
        if (!(obj instanceof DComplexMatrix2D))
            return false;

        return cern.colt.matrix.tdcomplex.algo.DComplexProperty.DEFAULT.equals(this, (DComplexMatrix2D) obj);
    }

    /**
     * Assigns the result of a function to each <i>non-zero</i> cell. Use this
     * method for fast special-purpose iteration. If you want to modify another
     * matrix instead of <tt>this</tt> (i.e. work in read-only mode), simply
     * return the input value unchanged.
     * 
     * Parameters to function are as follows: <tt>first==row</tt>,
     * <tt>second==column</tt>, <tt>third==nonZeroValue</tt>.
     * 
     * @param function
     *            a function object taking as argument the current non-zero
     *            cell's row, column and value.
     * @return <tt>this</tt> (for convenience only).
     */
    public DComplexMatrix2D forEachNonZero(final cern.colt.function.tdcomplex.IntIntDComplexFunction function) {
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                double[] value = getQuick(r, c);
                                if (value[0] != 0 || value[1] != 0) {
                                    double[] v = function.apply(r, c, value);
                                    setQuick(r, c, v);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double[] value = getQuick(r, c);
                    if (value[0] != 0 || value[1] != 0) {
                        double[] v = function.apply(r, c, value);
                        setQuick(r, c, v);
                    }
                }
            }
        }
        return this;
    }

    /**
     * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @return the value of the specified cell.
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>column&lt;0 || column&gt;=columns() || row&lt;0 || row&gt;=rows()</tt>
     */
    public double[] get(int row, int column) {
        if (column < 0 || column >= columns || row < 0 || row >= rows)
            throw new IndexOutOfBoundsException("row:" + row + ", column:" + column);
        return getQuick(row, column);
    }

    /**
     * Returns a new matrix that is a complex conjugate of this matrix. If
     * unconjugated complex transposition is needed, one should use viewDice()
     * method. This method creates a new object (not a view), so changes in the
     * returned matrix are NOT reflected in this matrix.
     * 
     * @return a complex conjugate matrix
     */
    public DComplexMatrix2D getConjugateTranspose() {
        final DComplexMatrix2D transpose = this.viewDice().copy();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? columns : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        double[] tmp = new double[2];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < rows; c++) {
                                tmp = transpose.getQuick(r, c);
                                tmp[1] = -tmp[1];
                                transpose.setQuick(r, c, tmp);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] tmp = new double[2];
            for (int r = 0; r < columns; r++) {
                for (int c = 0; c < rows; c++) {
                    tmp = transpose.getQuick(r, c);
                    tmp[1] = -tmp[1];
                    transpose.setQuick(r, c, tmp);
                }
            }
        }
        return transpose;
    }

    /**
     * Returns the elements of this matrix.
     * 
     * @return the elements
     */
    public abstract Object elements();

    /**
     * Returns the imaginary part of this matrix
     * 
     * @return the imaginary part
     */
    public abstract DoubleMatrix2D getImaginaryPart();

    /**
     * Fills the coordinates and values of cells having non-zero values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * <p>
     * In general, fill order is <i>unspecified</i>. This implementation fills
     * like <tt>for (row = 0..rows-1) for (column = 0..columns-1) do ... </tt>.
     * However, subclasses are free to us any other order, even an order that
     * may change over time as cell values are changed. (Of course, result lists
     * indexes are guaranteed to correspond to the same cell).
     * 
     * @param rowList
     *            the list to be filled with row indexes, can have any size.
     * @param columnList
     *            the list to be filled with column indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList,
            final ArrayList<double[]> valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double[] value = getQuick(r, c);
                if (value[0] != 0 || value[1] != 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
            }
        }

    }

    /**
     * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @return the value at the specified coordinate.
     */
    public abstract double[] getQuick(int row, int column);

    /**
     * Returns the real part of this matrix
     * 
     * @return the real part
     */
    public abstract DoubleMatrix2D getRealPart();

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the same number of rows and columns. For example,
     * if the receiver is an instance of type <tt>DenseComplexMatrix2D</tt> the
     * new matrix must also be of type <tt>DenseComplexMatrix2D</tt>. In
     * general, the new matrix should have internal parametrization as similar
     * as possible.
     * 
     * @return a new empty matrix of the same dynamic type.
     */
    public DComplexMatrix2D like() {
        return like(rows, columns);
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseComplexMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseComplexMatrix2D</tt>. In general, the new matrix should have
     * internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public abstract DComplexMatrix2D like(int rows, int columns);

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseComplexMatrix2D</tt> the new
     * matrix must be of type <tt>DenseComplexMatrix1D</tt>.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public abstract DComplexMatrix1D like1D(int size);

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified
     * value.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param value
     *            the value to be filled into the specified cell.
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>column&lt;0 || column&gt;=columns() || row&lt;0 || row&gt;=rows()</tt>
     */
    public void set(int row, int column, double[] value) {
        if (column < 0 || column >= columns || row < 0 || row >= rows)
            throw new IndexOutOfBoundsException("row:" + row + ", column:" + column);
        setQuick(row, column, value);
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified
     * value.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>column&lt;0 || column&gt;=columns() || row&lt;0 || row&gt;=rows()</tt>
     */
    public void set(int row, int column, double re, double im) {
        if (column < 0 || column >= columns || row < 0 || row >= rows)
            throw new IndexOutOfBoundsException("row:" + row + ", column:" + column);
        setQuick(row, column, re, im);
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified
     * value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     * 
     */
    public abstract void setQuick(int row, int column, double re, double im);

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified
     * value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param value
     *            the value to be filled into the specified cell.
     */
    public abstract void setQuick(int row, int column, double[] value);

    /**
     * Constructs and returns a 2-dimensional array containing the cell values.
     * The returned array <tt>values</tt> has the form
     * <tt>re = values[row][2*column]; im = values[row][2*column+1]</tt> and has
     * the same number of rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @return an array filled with the values of the cells.
     */
    public double[][] toArray() {
        final double[][] values = new double[rows][2 * columns];
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
                        double[] tmp;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                tmp = getQuick(r, c);
                                values[r][2 * c] = tmp[0];
                                values[r][2 * c + 1] = tmp[1];
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] tmp;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    tmp = getQuick(r, c);
                    values[r][2 * c] = tmp[0];
                    values[r][2 * c + 1] = tmp[1];
                }
            }
        }
        return values;
    }

    /**
     * Returns a string representation using default formatting ("%.4f").
     * 
     * @return a string representation of the matrix.
     */

    public String toString() {
        return toString("%.4f");
    }

    /**
     * Returns a string representation using using given <tt>format</tt>
     * 
     * @param format
     * @return a string representation of the matrix.
     * 
     */
    public String toString(String format) {
        StringBuffer s = new StringBuffer(String.format("ComplexMatrix2D: %d rows, %d columns\n\n", rows, columns));
        double[] elem = new double[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                elem = getQuick(r, c);
                if (elem[1] == 0) {
                    s.append(String.format(format + "\t", elem[0]));
                    continue;
                }
                if (elem[0] == 0) {
                    s.append(String.format(format + "i\t", elem[1]));
                    continue;
                }
                if (elem[1] < 0) {
                    s.append(String.format(format + " - " + format + "i\t", elem[0], -elem[1]));
                    continue;
                }
                s.append(String.format(format + " + " + format + "i\t", elem[0], elem[1]));
            }
            s.append("\n");
        }
        return s.toString();
    }

    /**
     * Returns a vector obtained by stacking the columns of this matrix on top
     * of one another.
     * 
     * @return a vector of columns of this matrix.
     */
    public abstract DComplexMatrix1D vectorize();

    /**
     * Constructs and returns a new <i>slice view</i> representing the rows of
     * the given column. The returned view is backed by this matrix, so changes
     * in the returned view are reflected in this matrix, and vice-versa. To
     * obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>viewPart(...)</tt>), then apply this method to the sub-range view.
     * 
     * @param column
     *            the column to fix.
     * @return a new slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>column < 0 || column >= columns()</tt>.
     * @see #viewRow(int)
     */
    public DComplexMatrix1D viewColumn(int column) {
        checkColumn(column);
        int viewSize = this.rows;
        int viewZero = (int) index(0, column);
        int viewStride = this.rowStride;
        return like1D(viewSize, viewZero, viewStride);
    }

    /**
     * Constructs and returns a new <i>flip view</i> along the column axis. What
     * used to be column <tt>0</tt> is now column <tt>columns()-1</tt>, ...,
     * what used to be column <tt>columns()-1</tt> is now column <tt>0</tt>. The
     * returned view is backed by this matrix, so changes in the returned view
     * are reflected in this matrix, and vice-versa.
     * 
     * @return a new flip view.
     * @see #viewRowFlip()
     */
    public DComplexMatrix2D viewColumnFlip() {
        return (DComplexMatrix2D) (view().vColumnFlip());
    }

    /**
     * Constructs and returns a new <i>dice (transposition) view</i>; Swaps
     * axes; example: 3 x 4 matrix --> 4 x 3 matrix. The view has both
     * dimensions exchanged; what used to be columns become rows, what used to
     * be rows become columns. This is a zero-copy transposition, taking O(1),
     * i.e. constant time. The returned view is backed by this matrix, so
     * changes in the returned view are reflected in this matrix, and
     * vice-versa. Use idioms like <tt>result = viewDice(A).copy()</tt> to
     * generate an independent transposed matrix.
     * 
     * @return a new dice view.
     */
    public DComplexMatrix2D viewDice() {
        return (DComplexMatrix2D) (view().vDice());
    }

    /**
     * Constructs and returns a new <i>sub-range view</i> that is a
     * <tt>height x width</tt> sub matrix starting at <tt>[row,column]</tt>.
     * 
     * Operations on the returned view can only be applied to the restricted
     * range. Any attempt to access coordinates not contained in the view will
     * throw an <tt>IndexOutOfBoundsException</tt>.
     * <p>
     * <b>Note that the view is really just a range restriction:</b> The
     * returned matrix is backed by this matrix, so changes in the returned
     * matrix are reflected in this matrix, and vice-versa.
     * <p>
     * The view contains the cells from <tt>[row,column]</tt> to
     * <tt>[row+height-1,column+width-1]</tt>, all inclusive. and has
     * <tt>view.rows() == height; view.columns() == width;</tt>. A view's legal
     * coordinates are again zero based, as usual. In other words, legal
     * coordinates of the view range from <tt>[0,0]</tt> to
     * <tt>[view.rows()-1==height-1,view.columns()-1==width-1]</tt>. As usual,
     * any attempt to access a cell at a coordinate
     * <tt>column&lt;0 || column&gt;=view.columns() || row&lt;0 || row&gt;=view.rows()</tt>
     * will throw an <tt>IndexOutOfBoundsException</tt>.
     * 
     * @param row
     *            The index of the row-coordinate.
     * @param column
     *            The index of the column-coordinate.
     * @param height
     *            The height of the box.
     * @param width
     *            The width of the box.
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>column<0 || width<0 || column+width>columns() || row<0 || height<0 || row+height>rows()</tt>
     * @return the new view.
     * 
     */
    public DComplexMatrix2D viewPart(int row, int column, int height, int width) {
        return (DComplexMatrix2D) (view().vPart(row, column, height, width));
    }

    /**
     * Constructs and returns a new <i>slice view</i> representing the columns
     * of the given row. The returned view is backed by this matrix, so changes
     * in the returned view are reflected in this matrix, and vice-versa. To
     * obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>viewPart(...)</tt>), then apply this method to the sub-range view.
     * 
     * @param row
     *            the row to fix.
     * @return a new slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>row < 0 || row >= rows()</tt>.
     * @see #viewColumn(int)
     */
    public DComplexMatrix1D viewRow(int row) {
        checkRow(row);
        int viewSize = this.columns;
        int viewZero = (int) index(row, 0);
        int viewStride = this.columnStride;
        return like1D(viewSize, viewZero, viewStride);
    }

    /**
     * Constructs and returns a new <i>flip view</i> along the row axis. What
     * used to be row <tt>0</tt> is now row <tt>rows()-1</tt>, ..., what used to
     * be row <tt>rows()-1</tt> is now row <tt>0</tt>. The returned view is
     * backed by this matrix, so changes in the returned view are reflected in
     * this matrix, and vice-versa.
     * 
     * @return a new flip view.
     * @see #viewColumnFlip()
     */
    public DComplexMatrix2D viewRowFlip() {
        return (DComplexMatrix2D) (view().vRowFlip());
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding all <b>rows</b> matching the given condition. Applies the
     * condition to each row and takes only those row where
     * <tt>condition.apply(viewRow(i))</tt> yields <tt>true</tt>. To match
     * columns, use a dice view.
     * 
     * @param condition
     *            The condition to be matched.
     * @return the new view.
     */
    public DComplexMatrix2D viewSelection(DComplexMatrix1DProcedure condition) {
        IntArrayList matches = new IntArrayList();
        for (int i = 0; i < rows; i++) {
            if (condition.apply(viewRow(i)))
                matches.add(i);
        }
        matches.trimToSize();
        return viewSelection(matches.elements(), null); // take all columns
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding the indicated cells. There holds
     * <tt>view.rows() == rowIndexes.length, view.columns() == columnIndexes.length</tt>
     * and <tt>view.get(i,j) == this.get(rowIndexes[i],columnIndexes[j])</tt>.
     * Indexes can occur multiple times and can be in arbitrary order.
     * 
     * Note that modifying the index arguments after this call has returned has
     * no effect on the view. The returned view is backed by this matrix, so
     * changes in the returned view are reflected in this matrix, and
     * vice-versa.
     * <p>
     * To indicate "all" rows or "all columns", simply set the respective
     * parameter
     * 
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
     *             if <tt>!(0 <= rowIndexes[i] < rows())</tt> for any
     *             <tt>i=0..rowIndexes.length()-1</tt>.
     * @throws IndexOutOfBoundsException
     *             if <tt>!(0 <= columnIndexes[i] < columns())</tt> for any
     *             <tt>i=0..columnIndexes.length()-1</tt>.
     */
    public DComplexMatrix2D viewSelection(int[] rowIndexes, int[] columnIndexes) {
        // check for "all"
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

        checkRowIndexes(rowIndexes);
        checkColumnIndexes(columnIndexes);
        int[] rowOffsets = new int[rowIndexes.length];
        int[] columnOffsets = new int[columnIndexes.length];
        for (int i = 0; i < rowIndexes.length; i++) {
            rowOffsets[i] = _rowOffset(_rowRank(rowIndexes[i]));
        }
        for (int i = 0; i < columnIndexes.length; i++) {
            columnOffsets[i] = _columnOffset(_columnRank(columnIndexes[i]));
        }
        return viewSelectionLike(rowOffsets, columnOffsets);
    }

    /**
     * Constructs and returns a new <i>stride view</i> which is a sub matrix
     * consisting of every i-th cell. More specifically, the view has
     * <tt>this.rows()/rowStride</tt> rows and
     * <tt>this.columns()/columnStride</tt> columns holding cells
     * <tt>this.get(i*rowStride,j*columnStride)</tt> for all
     * <tt>i = 0..rows()/rowStride - 1, j = 0..columns()/columnStride - 1</tt>.
     * The returned view is backed by this matrix, so changes in the returned
     * view are reflected in this matrix, and vice-versa.
     * 
     * @param rowStride
     *            the row step factor.
     * @param columnStride
     *            the column step factor.
     * @return a new view.
     * @throws IndexOutOfBoundsException
     *             if <tt>rowStride<=0 || columnStride<=0</tt>.
     */
    public DComplexMatrix2D viewStrides(int rowStride, int columnStride) {
        return (DComplexMatrix2D) (view().vStrides(rowStride, columnStride));
    }

    /**
     * Linear algebraic matrix-vector multiplication; <tt>z = A * y</tt>;
     * Equivalent to <tt>return A.zMult(y,z,1,0);</tt>
     * 
     * @param y
     *            the source vector.
     * @param z
     *            the vector where results are to be stored. Set this parameter
     *            to <tt>null</tt> to indicate that a new result vector shall be
     *            constructed.
     * @return z (for convenience only).
     */
    public DComplexMatrix1D zMult(DComplexMatrix1D y, DComplexMatrix1D z) {
        return zMult(y, z, new double[] { 1, 0 }, (z == null ? new double[] { 1, 0 } : new double[] { 0, 0 }), false);
    }

    /**
     * Linear algebraic matrix-vector multiplication;
     * <tt>z = alpha * A * y + beta*z</tt>. Where <tt>A == this</tt>. <br>
     * Note: Matrix shape conformance is checked <i>after</i> potential
     * transpositions.
     * 
     * @param y
     *            the source vector.
     * @param z
     *            the vector where results are to be stored. Set this parameter
     *            to <tt>null</tt> to indicate that a new result vector shall be
     *            constructed.
     * @return z (for convenience only).
     * 
     * @throws IllegalArgumentException
     *             if <tt>A.columns() != y.size() || A.rows() > z.size())</tt>.
     */
    public DComplexMatrix1D zMult(final DComplexMatrix1D y, DComplexMatrix1D z, final double[] alpha,
            final double[] beta, boolean transposeA) {
        if (transposeA)
            return getConjugateTranspose().zMult(y, z, alpha, beta, false);
        final DComplexMatrix1D zz;
        if (z == null) {
            zz = y.like(this.rows);
        } else {
            zz = z;
        }
        if (columns != y.size() || rows > zz.size())
            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort()
                    + ", " + zz.toStringShort());
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
                        double[] s = new double[2];
                        for (int r = firstRow; r < lastRow; r++) {
                            s[0] = 0;
                            s[1] = 0;
                            for (int c = 0; c < columns; c++) {
                                s = DComplex.plus(s, DComplex.mult(getQuick(r, c), y.getQuick(c)));
                            }
                            zz.setQuick(r, DComplex.plus(DComplex.mult(s, alpha), DComplex.mult(zz.getQuick(r), beta)));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] s = new double[2];
            for (int r = 0; r < rows; r++) {
                s[0] = 0;
                s[1] = 0;
                for (int c = 0; c < columns; c++) {
                    s = DComplex.plus(s, DComplex.mult(getQuick(r, c), y.getQuick(c)));
                }
                zz.setQuick(r, DComplex.plus(DComplex.mult(s, alpha), DComplex.mult(zz.getQuick(r), beta)));
            }
        }
        return zz;
    }

    /**
     * Linear algebraic matrix-matrix multiplication; <tt>C = A x B</tt>;
     * Equivalent to <tt>A.zMult(B,C,1,0,false,false)</tt>.
     * 
     * @param B
     *            the second source matrix.
     * @param C
     *            the matrix where results are to be stored. Set this parameter
     *            to <tt>null</tt> to indicate that a new result matrix shall be
     *            constructed.
     * @return C (for convenience only).
     */
    public DComplexMatrix2D zMult(DComplexMatrix2D B, DComplexMatrix2D C) {
        return zMult(B, C, new double[] { 1, 0 }, (C == null ? new double[] { 1, 0 } : new double[] { 0, 0 }), false,
                false);
    }

    /**
     * Linear algebraic matrix-matrix multiplication;
     * <tt>C = alpha * A x B + beta*C</tt>. Matrix shapes:
     * <tt>A(m x n), B(n x p), C(m x p)</tt>. <br>
     * Note: Matrix shape conformance is checked <i>after</i> potential
     * transpositions.
     * 
     * @param B
     *            the second source matrix.
     * @param C
     *            the matrix where results are to be stored. Set this parameter
     *            to <tt>null</tt> to indicate that a new result matrix shall be
     *            constructed.
     * @return C (for convenience only).
     * 
     * @throws IllegalArgumentException
     *             if <tt>B.rows() != A.columns()</tt>.
     * @throws IllegalArgumentException
     *             if
     *             <tt>C.rows() != A.rows() || C.columns() != B.columns()</tt>.
     * @throws IllegalArgumentException
     *             if <tt>A == C || B == C</tt>.
     */
    public DComplexMatrix2D zMult(final DComplexMatrix2D B, DComplexMatrix2D C, final double[] alpha,
            final double[] beta, boolean transposeA, boolean transposeB) {
        if (transposeA)
            return getConjugateTranspose().zMult(B, C, alpha, beta, false, transposeB);
        if (transposeB)
            return this.zMult(B.getConjugateTranspose(), C, alpha, beta, transposeA, false);
        final int m = rows;
        final int n = columns;
        final int p = B.columns;
        final DComplexMatrix2D CC;
        if (C == null) {
            CC = like(m, p);
        } else {
            CC = C;
        }
        if (B.rows != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + B.toStringShort());
        if (CC.rows != m || CC.columns != p)
            throw new IllegalArgumentException("Incompatibe result matrix: " + toStringShort() + ", "
                    + B.toStringShort() + ", " + CC.toStringShort());
        if (this == CC || B == CC)
            throw new IllegalArgumentException("Matrices must not be identical");
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, p);
            Future<?>[] futures = new Future[nthreads];
            int k = p / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? p : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        double[] s = new double[2];
                        for (int a = firstIdx; a < lastIdx; a++) {
                            for (int b = 0; b < m; b++) {
                                s[0] = 0;
                                s[1] = 0;
                                for (int c = 0; c < n; c++) {
                                    s = DComplex.plus(s, DComplex.mult(getQuick(b, c), B.getQuick(c, a)));
                                }
                                CC.setQuick(b, a, DComplex.plus(DComplex.mult(s, alpha), DComplex.mult(CC
                                        .getQuick(b, a), beta)));
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] s = new double[2];
            for (int a = 0; a < p; a++) {
                for (int b = 0; b < m; b++) {
                    s[0] = 0;
                    s[1] = 0;
                    for (int c = 0; c < n; c++) {
                        s = DComplex.plus(s, DComplex.mult(getQuick(b, c), B.getQuick(c, a)));
                    }
                    CC.setQuick(b, a, DComplex.plus(DComplex.mult(s, alpha), DComplex.mult(CC.getQuick(b, a), beta)));
                }
            }
        }
        return CC;
    }

    /**
     * Returns the sum of all cells.
     * 
     * @return the sum.
     */
    public double[] zSum() {
        if (size() == 0)
            return new double[] { 0, 0 };
        return aggregate(cern.jet.math.tdcomplex.DComplexFunctions.plus,
                cern.jet.math.tdcomplex.DComplexFunctions.identity);
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     * 
     * @return <tt>this</tt>
     */
    protected DComplexMatrix2D getContent() {
        return this;
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     * 
     * @param other
     *            matrix
     * @return <tt>true</tt> if both matrices share at least one identical cell.
     */
    protected boolean haveSharedCells(DComplexMatrix2D other) {
        if (other == null)
            return false;
        if (this == other)
            return true;
        return getContent().haveSharedCellsRaw(other.getContent());
    }

    /**
     * Always returns false
     * 
     * @param other
     *            matrix
     * @return false
     */
    protected boolean haveSharedCellsRaw(DComplexMatrix2D other) {
        return false;
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseComplexMatrix2D</tt> the new matrix must be of
     * type <tt>DenseComplexMatrix1D</tt>.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param zero
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @return a new matrix of the corresponding dynamic type.
     */
    protected abstract DComplexMatrix1D like1D(int size, int zero, int stride);

    /**
     * Constructs and returns a new view equal to the receiver. The view is a
     * shallow clone. Calls <code>clone()</code> and casts the result.
     * <p>
     * <b>Note that the view is not a deep copy.</b> The returned matrix is
     * backed by this matrix, so changes in the returned matrix are reflected in
     * this matrix, and vice-versa.
     * <p>
     * Use {@link #copy()} to construct an independent deep copy rather than a
     * new view.
     * 
     * @return a new view of the receiver.
     */
    protected DComplexMatrix2D view() {
        return (DComplexMatrix2D) clone();
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param rowOffsets
     *            the offsets of the visible elements.
     * @param columnOffsets
     *            the offsets of the visible elements.
     * @return a new view.
     */
    protected abstract DComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets);
}
