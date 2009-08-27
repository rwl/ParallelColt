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
import cern.colt.matrix.AbstractMatrix1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Abstract base class for 1-d matrices (aka <i>vectors</i>) holding
 * <tt>int</tt> elements. First see the <a href="package-summary.html">package
 * summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the
 * broad picture.
 * <p>
 * A matrix has a number of cells (its <i>size</i>), which are assigned upon
 * instance construction. Elements are accessed via zero based indexes. Legal
 * indexes are of the form <tt>[0..size()-1]</tt>. Any attempt to access an
 * element at a coordinate <tt>index&lt;0 || index&gt;=size()</tt> will throw an
 * <tt>IndexOutOfBoundsException</tt>.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public abstract class LongMatrix1D extends AbstractMatrix1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected LongMatrix1D() {
    }

    /**
     * Applies a function to each cell and aggregates the results. Returns a
     * value <tt>v</tt> such that <tt>v==a(size())</tt> where
     * <tt>a(i) == aggr( a(i-1), f(get(i)) )</tt> and terminators are
     * <tt>a(1) == f(get(0)), a(0)==Long.NaN</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   cern.jet.math.Functions F = cern.jet.math.Functions.functions;
     *   matrix = 0 1 2 3 
     * 
     *   // Sum( x[i]*x[i] ) 
     *   matrix.aggregate(F.plus,F.square);
     *   --&gt; 14
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
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        long a = f.apply(getQuick(0));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        long a = f.apply(getQuick(firstIdx));
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            a = aggr.apply(a, f.apply(getQuick(i)));
                        }
                        return Long.valueOf(a);
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            for (int i = 1; i < size; i++) {
                a = aggr.apply(a, f.apply(getQuick(i)));
            }
        }
        return a;
    }

    /**
     * 
     * Applies a function to all cells with a given indexes and aggregates the
     * results.
     * 
     * @param aggr
     *            an aggregation function taking as first argument the current
     *            aggregation and as second argument the transformed current
     *            cell value.
     * @param f
     *            a function transforming the current cell value.
     * @param indexList
     *            indexes.
     * 
     * @return the aggregated measure.
     * @see cern.jet.math.tlong.LongFunctions
     */
    public long aggregate(final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongFunction f, final IntArrayList indexList) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int size = indexList.size();
        final int[] indexElements = indexList.elements();
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(getQuick(indexElements[firstIdx]));
                        long elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            elem = getQuick(indexElements[i]);
                            a = aggr.apply(a, f.apply(elem));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            long elem;
            a = f.apply(getQuick(indexElements[0]));
            for (int i = 1; i < size; i++) {
                elem = getQuick(indexElements[i]);
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    /**
     * Applies a function to each corresponding cell of two matrices and
     * aggregates the results. Returns a value <tt>v</tt> such that
     * <tt>v==a(size())</tt> where
     * <tt>a(i) == aggr( a(i-1), f(get(i),other.get(i)) )</tt> and terminators
     * are <tt>a(1) == f(get(0),other.get(0)), a(0)==Long.NaN</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   cern.jet.math.Functions F = cern.jet.math.Functions.functions;
     *   x = 0 1 2 3 
     *   y = 0 1 2 3 
     * 
     *   // Sum( x[i]*y[i] )
     *   x.aggregate(y, F.plus, F.mult);
     *   --&gt; 14
     * 
     *   // Sum( (x[i]+y[i])&circ;2 )
     *   x.aggregate(y, F.plus, F.chain(F.square,F.plus));
     *   --&gt; 56
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
     *             if <tt>size() != other.size()</tt>.
     * @see cern.jet.math.tlong.LongFunctions
     */
    public long aggregate(final LongMatrix1D other, final cern.colt.function.tlong.LongLongFunction aggr,
            final cern.colt.function.tlong.LongLongFunction f) {
        checkSize(other);
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        long a = f.apply(getQuick(0), other.getQuick(0));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        long a = f.apply(getQuick(firstIdx), other.getQuick(firstIdx));
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            a = aggr.apply(a, f.apply(getQuick(i), other.getQuick(i)));
                        }
                        return Long.valueOf(a);
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            for (int i = 1; i < size; i++) {
                a = aggr.apply(a, f.apply(getQuick(i), other.getQuick(i)));
            }
        }
        return a;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[i] = function(x[i])</tt>. (Iterates downwards from
     * <tt>[size()-1]</tt> to <tt>[0]</tt>).
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   // change each cell to its sine
     *   matrix =   0.5      1.5      2.5       3.5 
     *   matrix.assign(cern.jet.math.Functions.sin);
     *   --&gt;
     *   matrix ==  0.479426 0.997495 0.598472 -0.350783
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param f
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tlong.LongFunctions
     */
    public LongMatrix1D assign(final cern.colt.function.tlong.LongFunction f) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, f.apply(getQuick(i)));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, f.apply(getQuick(i)));
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
    public LongMatrix1D assign(final cern.colt.function.tlong.LongProcedure cond,
            final cern.colt.function.tlong.LongFunction f) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        long elem;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elem = getQuick(i);
                            if (cond.apply(elem) == true) {
                                setQuick(i, f.apply(elem));
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            long elem;
            for (int i = 0; i < size; i++) {
                elem = getQuick(i);
                if (cond.apply(elem) == true) {
                    setQuick(i, f.apply(elem));
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
    public LongMatrix1D assign(final cern.colt.function.tlong.LongProcedure cond, final long value) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        long elem;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elem = getQuick(i);
                            if (cond.apply(elem) == true) {
                                setQuick(i, value);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            long elem;
            for (int i = 0; i < size; i++) {
                elem = getQuick(i);
                if (cond.apply(elem) == true) {
                    setQuick(i, value);
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
    public LongMatrix1D assign(final long value) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, value);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, value);
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the same number of cells as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != size()</tt>.
     */
    public LongMatrix1D assign(final long[] values) {
        if (values.length != size)
            throw new IllegalArgumentException("Must have same number of cells: length=" + values.length + "size()="
                    + size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, values[i]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, values[i]);
            }
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the same number of cells as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != size()</tt>.
     */
    public LongMatrix1D assign(final int[] values) {
        if (values.length != size)
            throw new IllegalArgumentException("Must have same number of cells: length=" + values.length + "size()="
                    + size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, values[i]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, values[i]);
            }
        }
        return this;
    }

    /**
     * Replaces all cell values of the receiver with the values of another
     * matrix. Both matrices must have the same size. If both matrices share the
     * same cells (as is the case if they are views derived from the same
     * matrix) and intersect in an ambiguous way, then replaces <i>as if</i>
     * using an intermediate auxiliary deep copy of <tt>other</tt>.
     * 
     * @param other
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public LongMatrix1D assign(LongMatrix1D other) {
        if (other == this)
            return this;
        checkSize(other);
        final LongMatrix1D other_loc;
        if (haveSharedCells(other)) {
            other_loc = other.copy();
        } else {
            other_loc = other;
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx;
                if (j == nthreads - 1) {
                    lastIdx = size;
                } else {
                    lastIdx = firstIdx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, other_loc.getQuick(i));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, other_loc.getQuick(i));
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[i] = function(x[i],y[i])</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   // assign x[i] = x[i]&lt;sup&gt;y[i]&lt;/sup&gt;
     *   m1 = 0 1 2 3;
     *   m2 = 0 2 4 6;
     *   m1.assign(m2, cern.jet.math.Functions.pow);
     *   --&gt;
     *   m1 == 1 1 16 729
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
     *             if <tt>size() != y.size()</tt>.
     * @see cern.jet.math.tlong.LongFunctions
     */
    public LongMatrix1D assign(final LongMatrix1D y, final cern.colt.function.tlong.LongLongFunction function) {
        checkSize(y);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            setQuick(i, function.apply(getQuick(i), y.getQuick(i)));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                setQuick(i, function.apply(getQuick(i), y.getQuick(i)));
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[i] = function(x[i],y[i])</tt>. (Iterates downwards from
     * <tt>[size()-1]</tt> to <tt>[0]</tt>).
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     *   // assign x[i] = x[i]&lt;sup&gt;y[i]&lt;/sup&gt;
     *   m1 = 0 1 2 3;
     *   m2 = 0 2 4 6;
     *   m1.assign(m2, cern.jet.math.Functions.pow);
     *   --&gt;
     *   m1 == 1 1 16 729
     * 
     *   // for non-standard functions there is no shortcut: 
     *   m1.assign(m2,
     *      new LongLongFunction() {
     *         public int apply(int x, int y) { return Math.pow(x,y); }
     *      }
     *   );
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
     *            cell's value of <tt>y</tt>.
     * @param nonZeroIndexes
     *            list of indexes of non-zero values
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != y.size()</tt>.
     * @see cern.jet.math.tlong.LongFunctions
     */
    public LongMatrix1D assign(LongMatrix1D y, cern.colt.function.tlong.LongLongFunction function,
            cern.colt.list.tint.IntArrayList nonZeroIndexes) {
        checkSize(y);
        int[] nonZeroElements = nonZeroIndexes.elements();

        // specialized for speed
        if (function == cern.jet.math.tlong.LongFunctions.mult) { // x[i] = x[i] *
            // y[i]
            int j = 0;
            for (int index = nonZeroIndexes.size(); --index >= 0;) {
                int i = nonZeroElements[index];
                for (; j < i; j++)
                    setQuick(j, 0); // x[i] = 0 for all zeros
                setQuick(i, getQuick(i) * y.getQuick(i)); // x[i] * y[i] for
                // all nonZeros
                j++;
            }
        } else if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
            long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
            if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                return this;
            } else if (multiplicator == 1) { // x[i] = x[i] + y[i]
                for (int index = nonZeroIndexes.size(); --index >= 0;) {
                    int i = nonZeroElements[index];
                    setQuick(i, getQuick(i) + y.getQuick(i));
                }
            } else if (multiplicator == -1) { // x[i] = x[i] - y[i]
                for (int index = nonZeroIndexes.size(); --index >= 0;) {
                    int i = nonZeroElements[index];
                    setQuick(i, getQuick(i) - y.getQuick(i));
                }
            } else { // the general case x[i] = x[i] + mult*y[i]
                for (int index = nonZeroIndexes.size(); --index >= 0;) {
                    int i = nonZeroElements[index];
                    setQuick(i, getQuick(i) + multiplicator * y.getQuick(i));
                }
            }
        } else { // the general case x[i] = f(x[i],y[i])
            return assign(y, function);
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
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            if (getQuick(i) != 0)
                                cardinality++;
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
            for (int i = 0; i < size; i++) {
                if (getQuick(i) != 0)
                    cardinality++;
            }
        }
        return cardinality;
    }

    /**
     * Returns the number of cells having non-zero values, but at most
     * maxCardinality; ignores tolerance.
     */
    protected int cardinality(int maxCardinality) {
        int cardinality = 0;
        int i = size;
        while (--i >= 0 && cardinality < maxCardinality) {
            if (getQuick(i) != 0)
                cardinality++;
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
    public LongMatrix1D copy() {
        LongMatrix1D copy = like();
        copy.assign(this);
        return copy;
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
     * and is at least a <code>LongMatrix1D</code> object that has the same
     * sizes as the receiver and has exactly the same values at the same
     * indexes.
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
        if (!(obj instanceof LongMatrix1D))
            return false;

        return cern.colt.matrix.tlong.algo.LongProperty.DEFAULT.equals(this, (LongMatrix1D) obj);
    }

    /**
     * Returns the matrix cell value at coordinate <tt>index</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @return the value of the specified cell.
     * @throws IndexOutOfBoundsException
     *             if <tt>index&lt;0 || index&gt;=size()</tt>.
     */
    public long get(int index) {
        if (index < 0 || index >= size)
            checkIndex(index);
        return getQuick(index);
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     */
    protected LongMatrix1D getContent() {
        return this;
    }

    /**
     * Fills the coordinates and values of cells having negative values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * 
     * @param indexList
     *            the list to be filled with indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getNegativeValues(final IntArrayList indexList, final LongArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int rem = size % 2;
        if (rem == 1) {
            long value = getQuick(0);
            if (value < 0) {
                indexList.add(0);
                valueList.add(value);
            }
        }

        for (int i = rem; i < size; i += 2) {
            long value = getQuick(i);
            if (value < 0) {
                indexList.add(i);
                valueList.add(value);
            }
            value = getQuick(i + 1);
            if (value < 0) {
                indexList.add(i + 1);
                valueList.add(value);
            }
        }
    }

    /**
     * Fills the coordinates and values of cells having non-zero values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * <p>
     * In general, fill order is <i>unspecified</i>. This implementation fills
     * like: <tt>for (index = 0..size()-1)  do ... </tt>. However, subclasses
     * are free to us any other order, even an order that may change over time
     * as cell values are changed. (Of course, result lists indexes are
     * guaranteed to correspond to the same cell).
     * <p>
     * <b>Example:</b> <br>
     * 
     * <pre>
     *   0, 0, 8, 0, 7
     *   --&gt;
     *   indexList  = (2,4)
     *   valueList  = (8,7)
     * 
     * </pre>
     * 
     * In other words, <tt>get(2)==8, get(4)==7</tt>.
     * 
     * @param indexList
     *            the list to be filled with indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getNonZeros(final IntArrayList indexList, final LongArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int rem = size % 2;
        if (rem == 1) {
            long value = getQuick(0);
            if (value != 0) {
                indexList.add(0);
                valueList.add(value);
            }
        }

        for (int i = rem; i < size; i += 2) {
            long value = getQuick(i);
            if (value != 0) {
                indexList.add(i);
                valueList.add(value);
            }
            value = getQuick(i + 1);
            if (value != 0) {
                indexList.add(i + 1);
                valueList.add(value);
            }
        }
    }

    /**
     * Fills the coordinates and values of the first <tt>maxCardinality</tt>
     * cells having non-zero values into the specified lists. Fills into the
     * lists, starting at index 0. After this call returns the specified lists
     * all have a new size, the number of non-zero values.
     * <p>
     * In general, fill order is <i>unspecified</i>. This implementation fills
     * like: <tt>for (index = 0..size()-1)  do ... </tt>. However, subclasses
     * are free to us any other order, even an order that may change over time
     * as cell values are changed. (Of course, result lists indexes are
     * guaranteed to correspond to the same cell).
     * <p>
     * <b>Example:</b> <br>
     * 
     * <pre>
     *   0, 0, 8, 0, 7
     *   --&gt;
     *   indexList  = (2,4)
     *   valueList  = (8,7)
     * 
     * </pre>
     * 
     * In other words, <tt>get(2)==8, get(4)==7</tt>.
     * 
     * @param indexList
     *            the list to be filled with indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     * @param maxCardinality
     *            maximal cardinality
     */
    public void getNonZeros(IntArrayList indexList, LongArrayList valueList, int maxCardinality) {
        boolean fillIndexList = indexList != null;
        boolean fillValueList = valueList != null;
        if (fillIndexList)
            indexList.clear();
        if (fillValueList)
            valueList.clear();
        int s = size;
        int currentSize = 0;
        for (int i = 0; i < s; i++) {
            long value = getQuick(i);
            if (value != 0) {
                if (fillIndexList)
                    indexList.add(i);
                if (fillValueList)
                    valueList.add(value);
                currentSize++;
            }
            if (currentSize >= maxCardinality) {
                break;
            }
        }
    }

    /**
     * Fills the coordinates and values of cells having positive values into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists all have a new size, the number of
     * non-zero values.
     * 
     * @param indexList
     *            the list to be filled with indexes, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void getPositiveValues(final IntArrayList indexList, final LongArrayList valueList) {
        indexList.clear();
        valueList.clear();
        int rem = size % 2;
        if (rem == 1) {
            long value = getQuick(0);
            if (value > 0) {
                indexList.add(0);
                valueList.add(value);
            }
        }

        for (int i = rem; i < size; i += 2) {
            long value = getQuick(i);
            if (value > 0) {
                indexList.add(i);
                valueList.add(value);
            }
            value = getQuick(i + 1);
            if (value > 0) {
                indexList.add(i + 1);
                valueList.add(value);
            }
        }
    }

    /**
     * Returns the matrix cell value at coordinate <tt>index</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @return the value of the specified cell.
     */
    public abstract long getQuick(int index);

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */
    protected boolean haveSharedCells(LongMatrix1D other) {
        if (other == null)
            return false;
        if (this == other)
            return true;
        return getContent().haveSharedCellsRaw(other.getContent());
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */
    protected boolean haveSharedCellsRaw(LongMatrix1D other) {
        return false;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the same size. For example, if the receiver is an
     * instance of type <tt>DenseLongMatrix1D</tt> the new matrix must also be
     * of type <tt>DenseLongMatrix1D</tt>, if the receiver is an instance of
     * type <tt>SparseLongMatrix1D</tt> the new matrix must also be of type
     * <tt>SparseLongMatrix1D</tt>, etc. In general, the new matrix should have
     * internal parametrization as similar as possible.
     * 
     * @return a new empty matrix of the same dynamic type.
     */
    public LongMatrix1D like() {
        return like(size);
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified size. For example, if the receiver
     * is an instance of type <tt>DenseLongMatrix1D</tt> the new matrix must
     * also be of type <tt>DenseLongMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseLongMatrix1D</tt> the new matrix must also be
     * of type <tt>SparseLongMatrix1D</tt>, etc. In general, the new matrix
     * should have internal parametrization as similar as possible.
     * 
     * @param size
     *            the number of cell the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public abstract LongMatrix1D like(int size);

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseLongMatrix1D</tt> the new matrix
     * must be of type <tt>DenseLongMatrix2D</tt>, if the receiver is an
     * instance of type <tt>SparseLongMatrix1D</tt> the new matrix must be of
     * type <tt>SparseLongMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public abstract LongMatrix2D like2D(int rows, int columns);

    /**
     * Return the maximum value of this matrix together with its location
     * 
     * @return { maximum_value, location };
     */
    public long[] getMaxLocation() {
        int location = 0;
        long maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        int location = firstIdx;
                        long maxValue = getQuick(location);
                        long elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            elem = getQuick(i);
                            if (maxValue < elem) {
                                maxValue = elem;
                                location = i;
                            }
                        }
                        return new long[] { maxValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                maxValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < nthreads; j++) {
                    if (maxValue < results[j][0]) {
                        maxValue = results[j][0];
                        location = (int) results[j][1];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            maxValue = getQuick(location);
            long elem;
            for (int i = 1; i < size(); i++) {
                elem = getQuick(i);
                if (maxValue < elem) {
                    maxValue = elem;
                    location = i;
                }
            }
        }
        return new long[] { maxValue, location };
    }

    /**
     * Return the minimum value of this matrix together with its location
     * 
     * @return { minimum_value, location };
     */
    public long[] getMinLocation() {
        int location = 0;
        long minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][2];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        int location = firstIdx;
                        long minValue = getQuick(location);
                        long elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            elem = getQuick(i);
                            if (minValue > elem) {
                                minValue = elem;
                                location = i;
                            }
                        }
                        return new long[] { minValue, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
                }
                minValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < nthreads; j++) {
                    if (minValue > results[j][0]) {
                        minValue = results[j][0];
                        location = (int) results[j][1];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            minValue = getQuick(location);
            long elem;
            for (int i = 1; i < size(); i++) {
                elem = getQuick(i);
                if (minValue > elem) {
                    minValue = elem;
                    location = i;
                }
            }
        }
        return new long[] { minValue, location };
    }

    /**
     * Returns new LongMatrix2D of size rows x columns whose elements are taken
     * column-wise from this matrix.
     * 
     * @param rows
     *            number of rows
     * @param columns
     *            number of columns
     * @return new 2D matrix with columns being the elements of this matrix.
     */
    public abstract LongMatrix2D reshape(int rows, int columns);

    /**
     * Returns new LongMatrix3D of size slices x rows x columns, whose elements
     * are taken column-wise from this matrix.
     * 
     * @param rows
     *            number of rows
     * @param columns
     *            number of columns
     * @return new 2D matrix with columns being the elements of this matrix.
     */
    public abstract LongMatrix3D reshape(int slices, int rows, int columns);

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * @param index
     *            the index of the cell.
     * @param value
     *            the value to be filled into the specified cell.
     * @throws IndexOutOfBoundsException
     *             if <tt>index&lt;0 || index&gt;=size()</tt>.
     */
    public void set(int index, long value) {
        if (index < 0 || index >= size)
            checkIndex(index);
        setQuick(index, value);
    }

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @param value
     *            the value to be filled into the specified cell.
     */
    public abstract void setQuick(int index, long value);

    /**
     * Sets the size of this matrix.
     * 
     * @param size
     */
    public void setSize(int size) {
        this.size = size;
    }

    /**
     * Swaps each element <tt>this[i]</tt> with <tt>other[i]</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */
    public void swap(final LongMatrix1D other) {
        checkSize(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            long tmp = getQuick(i);
                            setQuick(i, other.getQuick(i));
                            other.setQuick(i, tmp);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                long tmp = getQuick(i);
                setQuick(i, other.getQuick(i));
                other.setQuick(i, tmp);
            }
        }
    }

    /**
     * Constructs and returns a 1-dimensional array containing the cell values.
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa. The returned array
     * <tt>values</tt> has the form <br>
     * <tt>for (int i=0; i < size(); i++) values[i] = get(i);</tt>
     * 
     * @return an array filled with the values of the cells.
     */
    public long[] toArray() {
        long[] values = new long[size];
        toArray(values);
        return values;
    }

    /**
     * Fills the cell values into the specified 1-dimensional array. The values
     * are copied. So subsequent changes in <tt>values</tt> are not reflected in
     * the matrix, and vice-versa. After this call returns the array
     * <tt>values</tt> has the form <br>
     * <tt>for (int i=0; i < size(); i++) values[i] = get(i);</tt>
     * 
     * @throws IllegalArgumentException
     *             if <tt>values.length < size()</tt>.
     */
    public void toArray(final long[] values) {
        if (values.length < size)
            throw new IllegalArgumentException("values too small");
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            values[i] = getQuick(i);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                values[i] = getQuick(i);
            }
        }
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
    protected LongMatrix1D view() {
        return (LongMatrix1D) clone();
    }

    /**
     * Constructs and returns a new <i>flip view</i>. What used to be index
     * <tt>0</tt> is now index <tt>size()-1</tt>, ..., what used to be index
     * <tt>size()-1</tt> is now index <tt>0</tt>. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * 
     * @return a new flip view.
     */
    public LongMatrix1D viewFlip() {
        return (LongMatrix1D) (view().vFlip());
    }

    /**
     * Constructs and returns a new <i>sub-range view</i> that is a
     * <tt>width</tt> sub matrix starting at <tt>index</tt>.
     * 
     * Operations on the returned view can only be applied to the restricted
     * range. Any attempt to access coordinates not contained in the view will
     * throw an <tt>IndexOutOfBoundsException</tt>.
     * <p>
     * <b>Note that the view is really just a range restriction:</b> The
     * returned matrix is backed by this matrix, so changes in the returned
     * matrix are reflected in this matrix, and vice-versa.
     * <p>
     * The view contains the cells from <tt>index..index+width-1</tt>. and has
     * <tt>view.size() == width</tt>. A view's legal coordinates are again zero
     * based, as usual. In other words, legal coordinates of the view are
     * <tt>0 .. view.size()-1==width-1</tt>. As usual, any attempt to access a
     * cell at other coordinates will throw an
     * <tt>IndexOutOfBoundsException</tt>.
     * 
     * @param index
     *            The index of the first cell.
     * @param width
     *            The width of the range.
     * @throws IndexOutOfBoundsException
     *             if <tt>index<0 || width<0 || index+width>size()</tt>.
     * @return the new view.
     * 
     */
    public LongMatrix1D viewPart(int index, int width) {
        return (LongMatrix1D) (view().vPart(index, width));
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding the cells matching the given condition. Applies the condition to
     * each cell and takes only those cells where
     * <tt>condition.apply(get(i))</tt> yields <tt>true</tt>.
     * <p>
     * <b>Example:</b> <br>
     * 
     * <pre>
     *   // extract and view all cells with even value
     *   matrix = 0 1 2 3 
     *   matrix.viewSelection( 
     *      new LongProcedure() {
     *         public final boolean apply(int a) { return a % 2 == 0; }
     *      }
     *   );
     *   --&gt;
     *   matrix ==  0 2
     * 
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
    public LongMatrix1D viewSelection(cern.colt.function.tlong.LongProcedure condition) {
        IntArrayList matches = new IntArrayList();
        for (int i = 0; i < size; i++) {
            if (condition.apply(getQuick(i)))
                matches.add(i);
        }
        matches.trimToSize();
        return viewSelection(matches.elements());
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding the indicated cells. There holds
     * <tt>view.size() == indexes.length</tt> and
     * <tt>view.get(i) == this.get(indexes[i])</tt>. Indexes can occur multiple
     * times and can be in arbitrary order.
     * <p>
     * <b>Example:</b> <br>
     * 
     * <pre>
     *   this     = (0,0,8,0,7)
     *   indexes  = (0,2,4,2)
     *   --&gt;
     *   view     = (0,8,7,8)
     * 
     * </pre>
     * 
     * Note that modifying <tt>indexes</tt> after this call has returned has no
     * effect on the view. The returned view is backed by this matrix, so
     * changes in the returned view are reflected in this matrix, and
     * vice-versa.
     * 
     * @param indexes
     *            The indexes of the cells that shall be visible in the new
     *            view. To indicate that <i>all</i> cells shall be visible,
     *            simply set this parameter to <tt>null</tt>.
     * @return the new view.
     * @throws IndexOutOfBoundsException
     *             if <tt>!(0 <= indexes[i] < size())</tt> for any
     *             <tt>i=0..indexes.length()-1</tt>.
     */
    public LongMatrix1D viewSelection(int[] indexes) {
        // check for "all"
        if (indexes == null) {
            indexes = new int[size];
            for (int i = 0; i < size; i++)
                indexes[i] = i;
        }

        checkIndexes(indexes);
        int[] offsets = new int[indexes.length];
        for (int i = 0; i < indexes.length; i++) {
            offsets[i] = (int) index(indexes[i]);
        }
        return viewSelectionLike(offsets);
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param offsets
     *            the offsets of the visible elements.
     * @return a new view.
     */
    protected abstract LongMatrix1D viewSelectionLike(int[] offsets);

    /**
     * Sorts the vector into ascending order, according to the <i>natural
     * ordering</i>. This sort is guaranteed to be <i>stable</i>. For further
     * information, see
     * {@link cern.colt.matrix.tlong.algo.LongSorting#sort(LongMatrix1D)}. For
     * more advanced sorting functionality, see
     * {@link cern.colt.matrix.tlong.algo.LongSorting}.
     * 
     * @return a new sorted vector (matrix) view.
     */
    public LongMatrix1D viewSorted() {
        return cern.colt.matrix.tlong.algo.LongSorting.mergeSort.sort(this);
    }

    /**
     * Constructs and returns a new <i>stride view</i> which is a sub matrix
     * consisting of every i-th cell. More specifically, the view has size
     * <tt>this.size()/stride</tt> holding cells <tt>this.get(i*stride)</tt> for
     * all <tt>i = 0..size()/stride - 1</tt>.
     * 
     * @param stride
     *            the step factor.
     * @throws IndexOutOfBoundsException
     *             if <tt>stride <= 0</tt>.
     * @return the new view.
     * 
     */
    public LongMatrix1D viewStrides(int stride) {
        return (LongMatrix1D) (view().vStrides(stride));
    }

    /**
     * Returns the dot product of two vectors x and y, which is
     * <tt>Sum(x[i]*y[i])</tt>. Where <tt>x == this</tt>. Operates on cells at
     * indexes <tt>0 .. Math.min(size(),y.size())</tt>.
     * 
     * @param y
     *            the second vector.
     * @return the sum of products.
     */
    public long zDotProduct(LongMatrix1D y) {
        return zDotProduct(y, 0, size);
    }

    /**
     * Returns the dot product of two vectors x and y, which is
     * <tt>Sum(x[i]*y[i])</tt>. Where <tt>x == this</tt>. Operates on cells at
     * indexes <tt>from .. Min(size(),y.size(),from+length)-1</tt>.
     * 
     * @param y
     *            the second vector.
     * @param from
     *            the first index to be considered.
     * @param length
     *            the number of cells to be considered.
     * @return the sum of products; zero if <tt>from<0 || length<0</tt>.
     */
    public long zDotProduct(final LongMatrix1D y, final int from, int length) {
        if (from < 0 || length <= 0)
            return 0;

        int tail = from + length;
        if (size < tail)
            tail = size;
        if (y.size < tail)
            tail = y.size;
        length = tail - from;

        long sum = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, length);
            Future<?>[] futures = new Future[nthreads];
            Long[] results = new Long[nthreads];
            int k = length / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? length : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {
                    public Long call() throws Exception {
                        long sum = 0;
                        int idx;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            idx = k + from;
                            sum += getQuick(idx) * y.getQuick(idx);
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Long) futures[j].get();
                }
                sum = results[0];
                for (int j = 1; j < nthreads; j++) {
                    sum += results[j];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int i = tail - 1;
            for (int k = length; --k >= 0; i--) {
                sum += getQuick(i) * y.getQuick(i);
            }
        }
        return sum;
    }

    /**
     * Returns the dot product of two vectors x and y, which is
     * <tt>Sum(x[i]*y[i])</tt>. Where <tt>x == this</tt>.
     * 
     * @param y
     *            the second vector.
     * @param nonZeroIndexes
     *            the indexes of cells in <tt>y</tt>having a non-zero value.
     * @return the sum of products.
     */
    public long zDotProduct(LongMatrix1D y, int from, int length, IntArrayList nonZeroIndexes) {
        // determine minimum length
        if (from < 0 || length <= 0)
            return 0;

        int tail = from + length;
        if (size < tail)
            tail = size;
        if (y.size < tail)
            tail = y.size;
        length = tail - from;
        if (length <= 0)
            return 0;
        IntArrayList indexesCopy = nonZeroIndexes.copy();
        indexesCopy.trimToSize();
        indexesCopy.quickSort();
        int[] nonZeroIndexElements = indexesCopy.elements();
        int index = 0;
        int s = indexesCopy.size();
        // skip to start
        while ((index < s) && nonZeroIndexElements[index] < from)
            index++;
        // now the sparse dot product
        int i;
        long sum = 0;
        while ((--length >= 0) && (index < s) && ((i = nonZeroIndexElements[index]) < tail)) {
            sum += getQuick(i) * y.getQuick(i);
            index++;
        }
        return sum;
    }

    /**
     * Returns the dot product of two vectors x and y, which is
     * <tt>Sum(x[i]*y[i])</tt>. Where <tt>x == this</tt>.
     * 
     * @param y
     *            the second vector.
     * @param nonZeroIndexes
     *            the indexes of cells in <tt>y</tt>having a non-zero value.
     * @return the sum of products.
     */
    protected long zDotProduct(LongMatrix1D y, IntArrayList nonZeroIndexes) {
        return zDotProduct(y, 0, size, nonZeroIndexes);
    }

    /**
     * Returns the sum of all cells; <tt>Sum( x[i] )</tt>.
     * 
     * @return the sum.
     */
    public long zSum() {
        if (size() == 0)
            return 0;
        return aggregate(cern.jet.math.tlong.LongFunctions.plus, cern.jet.math.tlong.LongFunctions.identity);
    }
}
