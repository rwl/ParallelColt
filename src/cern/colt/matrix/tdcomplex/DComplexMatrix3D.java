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
import cern.colt.matrix.AbstractMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Abstract base class for 3-d matrices holding <tt>complex</tt> elements.
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
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public abstract class DComplexMatrix3D extends AbstractMatrix3D {
    private static final long serialVersionUID = 1L;

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected DComplexMatrix3D() {
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
     * @see cern.jet.math.tdouble.DoubleFunctions
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
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {

                    public double[] call() throws Exception {
                        double[] a = f.apply(getQuick(firstSlice, 0, 0));
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
     *             <tt>slices() != other.slices() || rows() != other.rows() || columns() != other.columns()</tt>
     * @see cern.jet.math.tdouble.DoubleFunctions
     */
    public double[] aggregate(final DComplexMatrix3D other,
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
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {

                    public double[] call() throws Exception {
                        double[] a = f.apply(getQuick(firstSlice, 0, 0), other.getQuick(firstSlice, 0, 0));
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
     * Assigns the result of a function to each cell.
     * 
     * @param function
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix3D assign(final cern.colt.function.tdcomplex.DComplexDComplexFunction function) {
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
     * Assigns the result of a function to the real part of the receiver. The
     * imaginary part of the receiver is reset to zero.
     * 
     * @param function
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix3D assign(final cern.colt.function.tdcomplex.DComplexRealFunction function) {
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
                        double[] tmp = new double[2];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    tmp[0] = function.apply(getQuick(s, r, c));
                                    setQuick(s, r, c, tmp);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);

        } else {
            double[] tmp = new double[2];
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        tmp[0] = function.apply(getQuick(s, r, c));
                        setQuick(s, r, c, tmp);
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
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix3D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond,
            final cern.colt.function.tdcomplex.DComplexDComplexFunction f) {
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
                        double[] elem;
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
            double[] elem;
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
     *            a value (re=value[0], im=value[1]).
     * @return <tt>this</tt> (for convenience only).
     * 
     */
    public DComplexMatrix3D assign(final cern.colt.function.tdcomplex.DComplexProcedure cond, final double[] value) {
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
                        double[] elem;
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
            double[] elem;
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
    public DComplexMatrix3D assign(DComplexMatrix3D other) {
        if (other == this)
            return this;
        checkShape(other);
        final DComplexMatrix3D B;
        if (haveSharedCells(other)) {
            B = other.copy();
        } else {
            B = other;
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
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, B.getQuick(s, r, c));
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
                        setQuick(s, r, c, B.getQuick(s, r, c));
                    }
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
     * @param function
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>slices() != other.slices() || rows() != other.rows() || columns() != other.columns()</tt>
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix3D assign(final DComplexMatrix3D y,
            final cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction function) {
        checkShape(y);
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
     * Sets all cells to the state specified by <tt>re</tt> and <tt>im</tt>.
     * 
     * @param re
     *            the real part of the value to be filled into the cells.
     * @param im
     *            the imagiary part of the value to be filled into the cells.
     * 
     * @return <tt>this</tt> (for convenience only).
     */
    public DComplexMatrix3D assign(final double re, final double im) {
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
                                    setQuick(s, r, c, re, im);
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
                        setQuick(s, r, c, re, im);
                    }
                }
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form
     * <tt>re = values[slice*silceStride+row*rowStride+2*column], 
     * im = values[slice*silceStride+row*rowStride+2*column+1]</tt> and have
     * exactly the same number of slices, rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != slices()*rows()*2*columns()
     */
    public DComplexMatrix3D assign(final double[] values) {
        if (values.length != slices * rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length
                    + "slices()*rows()*2*columns()=" + slices() * rows() * 2 * columns());
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
                        int idx = firstSlice * rows * columns * 2;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, values[idx], values[idx + 1]);
                                    idx += 2;
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
                        setQuick(s, r, c, values[idx], values[idx + 1]);
                        idx += 2;
                    }
                }
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the form
     * <tt>re = values[slice][row][2*column], im = values[slice][row][2*column+1]</tt>
     * and have exactly the same number of slices, rows and columns as the
     * receiver.
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
     *             <tt>for any 0 &lt;= column &lt; columns(): values[slice][row].length != 2*columns()</tt>
     *             .
     */
    public DComplexMatrix3D assign(final double[][][] values) {
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
                            double[][] currentSlice = values[s];
                            if (currentSlice.length != rows)
                                throw new IllegalArgumentException(
                                        "Must have same number of rows in every slice: rows=" + currentSlice.length
                                                + "rows()=" + rows());
                            for (int r = 0; r < rows; r++) {
                                double[] currentRow = currentSlice[r];
                                if (currentRow.length != 2 * columns)
                                    throw new IllegalArgumentException(
                                            "Must have same number of columns in every row: columns="
                                                    + currentRow.length + "2*columns()=" + 2 * columns());
                                for (int c = 0; c < columns; c++) {
                                    setQuick(s, r, c, currentRow[2 * c], currentRow[2 * c + 1]);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);

        } else {
            for (int s = 0; s < slices; s++) {
                double[][] currentSlice = values[s];
                if (currentSlice.length != rows)
                    throw new IllegalArgumentException("Must have same number of rows in every slice: rows="
                            + currentSlice.length + "rows()=" + rows());
                for (int r = 0; r < rows; r++) {
                    double[] currentRow = currentSlice[r];
                    if (currentRow.length != 2 * columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "2*columns()=" + 2 * columns());
                    for (int c = 0; c < columns; c++) {
                        setQuick(s, r, c, currentRow[2 * c], currentRow[2 * c + 1]);
                    }
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
    public DComplexMatrix3D assignImaginary(final DoubleMatrix3D other) {
        checkShape(other);
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
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    double re = getQuick(s, r, c)[0];
                                    double im = other.getQuick(s, r, c);
                                    setQuick(s, r, c, re, im);
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
                        double re = getQuick(s, r, c)[0];
                        double im = other.getQuick(s, r, c);
                        setQuick(s, r, c, re, im);
                    }
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
    public DComplexMatrix3D assignReal(final DoubleMatrix3D other) {
        checkShape(other);
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
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    double re = other.getQuick(s, r, c);
                                    double im = getQuick(s, r, c)[1];
                                    setQuick(s, r, c, re, im);
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
                        double re = other.getQuick(s, r, c);
                        double im = getQuick(s, r, c)[1];
                        setQuick(s, r, c, re, im);
                    }
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
                        double[] tmp = new double[2];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    tmp = getQuick(s, r, c);
                                    if ((tmp[0] != 0.0) || (tmp[1] != 0.0))
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
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        tmp = getQuick(s, r, c);
                        if (tmp[0] != 0 || tmp[1] != 0)
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
    public DComplexMatrix3D copy() {
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
     * and is at least a <code>DoubleMatrix3D</code> object that has the same
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
        if (!(obj instanceof DComplexMatrix3D))
            return false;

        return cern.colt.matrix.tdcomplex.algo.DComplexProperty.DEFAULT.equals(this, (DComplexMatrix3D) obj);
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
    public double[] get(int slice, int row, int column) {
        if (slice < 0 || slice >= slices || row < 0 || row >= rows || column < 0 || column >= columns)
            throw new IndexOutOfBoundsException("slice:" + slice + ", row:" + row + ", column:" + column);
        return getQuick(slice, row, column);
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
    public abstract DoubleMatrix3D getImaginaryPart();

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
     * indexes are guaranteed to correspond to the same cell).
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
            final ArrayList<double[]> valueList) {
        sliceList.clear();
        rowList.clear();
        columnList.clear();
        valueList.clear();

        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double[] value = getQuick(s, r, c);
                    if (value[0] != 0 || value[1] != 0) {
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
    public abstract double[] getQuick(int slice, int row, int column);

    /**
     * Returns the real part of this matrix
     * 
     * @return the real part
     */
    public abstract DoubleMatrix3D getRealPart();

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the same number of slices, rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseComplexMatrix3D</tt> the new matrix must also be of type
     * <tt>DenseComplexMatrix3D</tt>. In general, the new matrix should have
     * internal parametrization as similar as possible.
     * 
     * @return a new empty matrix of the same dynamic type.
     */
    public DComplexMatrix3D like() {
        return like(slices, rows, columns);
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of slices, rows and columns.
     * For example, if the receiver is an instance of type
     * <tt>DenseComplexMatrix3D</tt> the new matrix must also be of type
     * <tt>DenseComplexMatrix3D</tt>. In general, the new matrix should have
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
    public abstract DComplexMatrix3D like(int slices, int rows, int columns);

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
    public void set(int slice, int row, int column, double[] value) {
        if (slice < 0 || slice >= slices || row < 0 || row >= rows || column < 0 || column >= columns)
            throw new IndexOutOfBoundsException("slice:" + slice + ", row:" + row + ", column:" + column);
        setQuick(slice, row, column, value);
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
     *            the index of the column-coordinate. the index of the
     *            column-coordinate.
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     * 
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>row&lt;0 || row&gt;=rows() || slice&lt;0 || slice&gt;=slices() || column&lt;0 || column&gt;=column()</tt>
     *             .
     */
    public void set(int slice, int row, int column, double re, double im) {
        if (slice < 0 || slice >= slices || row < 0 || row >= rows || column < 0 || column >= columns)
            throw new IndexOutOfBoundsException("slice:" + slice + ", row:" + row + ", column:" + column);
        setQuick(slice, row, column, re, im);
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
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     * 
     */
    public abstract void setQuick(int slice, int row, int column, double re, double im);

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
    public abstract void setQuick(int slice, int row, int column, double[] value);

    /**
     * Constructs and returns a 3-dimensional array containing the cell values.
     * The returned array <tt>values</tt> has the form
     * <tt>values[slice][row][column]</tt> and has the same number of slices,
     * rows and columns as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @return an array filled with the values of the cells.
     */
    public double[][][] toArray() {
        final double[][][] values = new double[slices][rows][2 * columns];
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
                        double[] tmp = new double[2];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            double[][] currentSlice = values[s];
                            for (int r = 0; r < rows; r++) {
                                double[] currentRow = currentSlice[r];
                                for (int c = 0; c < columns; c++) {
                                    tmp = getQuick(s, r, c);
                                    currentRow[2 * c] = tmp[0];
                                    currentRow[2 * c + 1] = tmp[1];
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            double[] tmp = new double[2];
            for (int s = 0; s < slices; s++) {
                double[][] currentSlice = values[s];
                for (int r = 0; r < rows; r++) {
                    double[] currentRow = currentSlice[r];
                    for (int c = 0; c < columns; c++) {
                        tmp = getQuick(s, r, c);
                        currentRow[2 * c] = tmp[0];
                        currentRow[2 * c + 1] = tmp[1];
                    }
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
        StringBuffer sb = new StringBuffer(String.format("ComplexMatrix3D: %d slices, %d rows, %d columns\n\n", slices,
                rows, columns));
        double[] elem = new double[2];
        for (int s = 0; s < slices; s++) {
            sb.append(String.format("(:,:,%d)=\n", s));
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    elem = getQuick(s, r, c);
                    if (elem[1] == 0) {
                        sb.append(String.format(format + "\t", elem[0]));
                        continue;
                    }
                    if (elem[0] == 0) {
                        sb.append(String.format(format + "i\t", elem[1]));
                        continue;
                    }
                    if (elem[1] < 0) {
                        sb.append(String.format(format + " - " + format + "i\t", elem[0], -elem[1]));
                        continue;
                    }
                    sb.append(String.format(format + " + " + format + "i\t", elem[0], elem[1]));
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Returns a vector obtained by stacking the columns of each slice of the
     * matrix on top of one another.
     * 
     * @return a vector obtained by stacking the columns of each slice of the
     *         matrix on top of one another.
     */
    public abstract DComplexMatrix1D vectorize();

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
    public DComplexMatrix2D viewColumn(int column) {
        checkColumn(column);
        int sliceRows = this.slices;
        int sliceColumns = this.rows;

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
    public DComplexMatrix3D viewColumnFlip() {
        return (DComplexMatrix3D) (view().vColumnFlip());
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
    public DComplexMatrix3D viewDice(int axis0, int axis1, int axis2) {
        return (DComplexMatrix3D) (view().vDice(axis0, axis1, axis2));
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
    public DComplexMatrix3D viewPart(int slice, int row, int column, int depth, int height, int width) {
        return (DComplexMatrix3D) (view().vPart(slice, row, column, depth, height, width));
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
    public DComplexMatrix2D viewRow(int row) {
        checkRow(row);
        int sliceRows = this.slices;
        int sliceColumns = this.columns;

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
    public DComplexMatrix3D viewRowFlip() {
        return (DComplexMatrix3D) (view().vRowFlip());
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding all <b>slices</b> matching the given condition. Applies the
     * condition to each slice and takes only those where
     * <tt>condition.apply(viewSlice(i))</tt> yields <tt>true</tt>. To match
     * rows or columns, use a dice view. The returned view is backed by this
     * matrix, so changes in the returned view are reflected in this matrix, and
     * vice-versa.
     * 
     * @param condition
     *            The condition to be matched.
     * @return the new view.
     */
    public DComplexMatrix3D viewSelection(DComplexMatrix2DProcedure condition) {
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
     * . Indexes can occur multiple times and can be in arbitrary order.
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
    public DComplexMatrix3D viewSelection(int[] sliceIndexes, int[] rowIndexes, int[] columnIndexes) {
        // check for "all"
        if (sliceIndexes == null) {
            sliceIndexes = new int[slices];
            for (int i = slices; --i >= 0;)
                sliceIndexes[i] = i;
        }
        if (rowIndexes == null) {
            rowIndexes = new int[rows];
            for (int i = rows; --i >= 0;)
                rowIndexes[i] = i;
        }
        if (columnIndexes == null) {
            columnIndexes = new int[columns];
            for (int i = columns; --i >= 0;)
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
    public DComplexMatrix2D viewSlice(int slice) {
        checkSlice(slice);
        int sliceRows = this.rows;
        int sliceColumns = this.columns;

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
    public DComplexMatrix3D viewSliceFlip() {
        return (DComplexMatrix3D) (view().vSliceFlip());
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
    public DComplexMatrix3D viewStrides(int sliceStride, int rowStride, int columnStride) {
        return (DComplexMatrix3D) (view().vStrides(sliceStride, rowStride, columnStride));
    }

    /**
     * Returns the sum of all cells; <tt>Sum( x[i,j,k] )</tt>.
     * 
     * @return the sum.
     */
    public double[] zSum() {
        if (size() == 0)
            return new double[2];
        return aggregate(cern.jet.math.tdcomplex.DComplexFunctions.plus,
                cern.jet.math.tdcomplex.DComplexFunctions.identity);
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     * 
     * @return this
     */
    protected DComplexMatrix3D getContent() {
        return this;
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     * 
     * @param other
     *            matrix
     * @return <tt>true</tt> if both matrices share at least one identical cell.
     */
    protected boolean haveSharedCells(DComplexMatrix3D other) {
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
    protected boolean haveSharedCellsRaw(DComplexMatrix3D other) {
        return false;
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseComplexMatrix3D</tt> the new matrix must also
     * be of type <tt>DenseComplexMatrix2D</tt>.
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
    protected abstract DComplexMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
            int columnStride);

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
    protected DComplexMatrix3D view() {
        return (DComplexMatrix3D) clone();
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
    protected abstract DComplexMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets);

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseDComplexMatrix3D</tt> the new matrix must also
     * be of type <tt>DenseDComplexMatrix2D</tt>, if the receiver is an instance
     * of type <tt>SparseDComplexMatrix3D</tt> the new matrix must also be of
     * type <tt>SparseDComplexMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public abstract DComplexMatrix2D like2D(int rows, int columns);
}
