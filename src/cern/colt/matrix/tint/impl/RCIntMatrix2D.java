/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse row-compressed 2-d matrix holding <tt>int</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally uses the standard sparse row-compressed format, with two important
 * differences that broaden the applicability of this storage format:
 * <ul>
 * <li>We use a {@link cern.colt.list.tint.IntArrayList} and
 * {@link cern.colt.list.tint.IntArrayList} to hold the column indexes and
 * nonzero values, respectively. This improves set(...) performance, because the
 * standard way of using non-resizable primitive arrays causes excessive memory
 * allocation, garbage collection and array copying. The small downside of this
 * is that set(...,0) does not free memory (The capacity of an arraylist does
 * not shrink upon element removal).
 * <li>Column indexes are kept sorted within a row. This both improves get and
 * set performance on rows with many non-zeros, because we can use a binary
 * search. (Experiments show that this hurts < 10% on rows with < 4 nonZeros.)
 * </ul>
 * <br>
 * Note that this implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * Cells that
 * <ul>
 * <li>are never set to non-zero values do not use any memory.
 * <li>switch from zero to non-zero state do use memory.
 * <li>switch back from non-zero to zero state also do use memory. Their memory
 * is <i>not</i> automatically reclaimed (because of the lists vs. arrays).
 * Reclamation can be triggered via {@link #trimToSize()}.
 * </ul>
 * <p>
 * <tt>memory [bytes] = 4*rows + 12 * nonZeros</tt>. <br>
 * Where <tt>nonZeros = cardinality()</tt> is the number of non-zero cells.
 * Thus, a 1000 x 1000 matrix with 1000000 non-zero cells consumes 11.5 MB. The
 * same 1000 x 1000 matrix with 1000 non-zero cells consumes 15 KB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * Getting a cell value takes time<tt> O(log nzr)</tt> where <tt>nzr</tt> is the
 * number of non-zeros of the touched row. This is usually quick, because
 * typically there are only few nonzeros per row. So, in practice, get has
 * <i>expected</i> constant time. Setting a cell value takes <i> </i>worst-case
 * time <tt>O(nz)</tt> where <tt>nzr</tt> is the total number of non-zeros in
 * the matrix. This can be extremely slow, but if you traverse coordinates
 * properly (i.e. upwards), each write is done much quicker:
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 * // rather quick
 * matrix.assign(0);
 * for (int row = 0; row &lt; rows; row++) {
 *     for (int column = 0; column &lt; columns; column++) {
 *         if (someCondition)
 *             matrix.setQuick(row, column, someValue);
 *     }
 * }
 * 
 * // poor
 * matrix.assign(0);
 * for (int row = rows; --row &gt;= 0;) {
 *     for (int column = columns; --column &gt;= 0;) {
 *         if (someCondition)
 *             matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * </td>
 * </table>
 * If for whatever reasons you can't iterate properly, consider to create an
 * empty dense matrix, store your non-zeros in it, then call
 * <tt>sparse.assign(dense)</tt>. Under the circumstances, this is still rather
 * quick.
 * <p>
 * Fast iteration over non-zeros can be done via {@link #forEachNonZero}, which
 * supplies your function with row, column and value of each nonzero. Although
 * the internally implemented version is a bit more sophisticated, here is how a
 * quite efficient user-level matrix-vector multiplication could look like:
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 * // Linear algebraic y = A * x
 * A.forEachNonZero(new cern.colt.function.IntIntIntFunction() {
 *     public int apply(int row, int column, int value) {
 *         y.setQuick(row, y.getQuick(row) + value * x.getQuick(column));
 *         return value;
 *     }
 * });
 * </pre>
 * 
 * </td>
 * </table>
 * <p>
 * Here is how a a quite efficient user-level combined scaling operation could
 * look like:
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 * // Elementwise A = A + alpha*B
 * B.forEachNonZero(new cern.colt.function.IntIntIntFunction() {
 *     public int apply(int row, int column, int value) {
 *         A.setQuick(row, column, A.getQuick(row, column) + alpha * value);
 *         return value;
 *     }
 * });
 * </pre>
 * 
 * </td>
 * </table>
 * Method
 * {@link #assign(IntMatrix2D,cern.colt.function.tint.IntIntFunction)}
 * does just that if you supply
 * {@link cern.jet.math.tint.IntFunctions#plusMultSecond} as argument.
 * 
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 0.9, 04/14/2000
 */
public class RCIntMatrix2D extends WrapperIntMatrix2D {
    /*
     * The elements of the matrix.
     */
    protected IntArrayList columnindexes;

    protected IntArrayList values;

    protected int[] rowPointers;


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
    public RCIntMatrix2D(int[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length);
        assign(values);
    }

    public RCIntMatrix2D(int rows, int columns, int[] rowPointers, IntArrayList columnIndexes, IntArrayList values) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.rowPointers = rowPointers;
        this.columnindexes = columnIndexes;
        this.values = values;
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
     *             <tt>rows<0 || columns<0 || (int)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public RCIntMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        columnindexes = new IntArrayList();
        values = new IntArrayList();
        rowPointers = new int[rows + 1];
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
     *             <tt>rows<0 || columns<0 || (int)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public RCIntMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        columnindexes = new IntArrayList(nzmax);
        values = new IntArrayList(nzmax);
        rowPointers = new int[rows + 1];
    }

    /**
     * Sets all nonzero cells to the state specified by <tt>value</tt>.
     * 
     * @param value
     *            the value to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     */
    public IntMatrix2D assign(int value) {
        // overriden for performance only
        if (value == 0) {
            columnindexes.clear();
            values.clear();
            rowPointers = new int[rows + 1];
            //            for (int i = starts.length; --i >= 0;)
            //                starts[i] = 0;
        } else {
            //            super.assign(value);
            int nnz = cardinality();
            for (int i = 0; i < nnz; i++) {
                values.setQuick(i, value);
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to each nonzero cell;
     * 
     * @param function
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tint.IntFunctions
     */
    public IntMatrix2D assign(final cern.colt.function.tint.IntFunction function) {
        if (function instanceof cern.jet.math.tint.IntMult) { // x[i] = mult*x[i]
            final int alpha = ((cern.jet.math.tint.IntMult) function).multiplicator;
            if (alpha == 1)
                return this;
            if (alpha == 0)
                return assign(0);
            if (alpha != alpha)
                return assign(alpha); // the funny definition of isNaN(). This should better not happen.

            final int[] valuesE = values.elements();
            int np = ConcurrencyUtils.getNumberOfThreads();
            if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future<?>[] futures = new Future[np];
                int k = valuesE.length / np;
                for (int j = 0; j < np; j++) {
                    final int startidx = j * k;
                    final int stopidx;
                    if (j == np - 1) {
                        stopidx = valuesE.length;
                    } else {
                        stopidx = startidx + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            for (int i = startidx; i < stopidx; i++) {
                                valuesE[i] *= alpha;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int j = values.size(); --j >= 0;) {
                    valuesE[j] *= alpha;
                }
            }
        } else {
            forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    return function.apply(value);
                }
            });
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
     * @param source
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>columns() != source.columns() || rows() != source.rows()</tt>
     */
    public IntMatrix2D assign(IntMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);
        // overriden for performance only
        if (!(source instanceof RCIntMatrix2D)) {
            assign(0);
            source.forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
            /*
             * indexes.clear(); values.clear(); int nonZeros=0; for (int row=0;
             * row<rows; row++) { starts[row]=nonZeros; for (int column=0;
             * column<columns; column++) { int v =
             * source.getQuick(row,column); if (v!=0) { values.add(v);
             * indexes.add(column); nonZeros++; } } } starts[rows]=nonZeros;
             */
            return this;
        }

        // even quicker
        RCIntMatrix2D other = (RCIntMatrix2D) source;

        System.arraycopy(other.rowPointers, 0, this.rowPointers, 0, this.rowPointers.length);
        int s = other.columnindexes.size();
        this.columnindexes.setSize(s);
        this.values.setSize(s);
        this.columnindexes.replaceFromToWithFrom(0, s - 1, other.columnindexes, 0);
        this.values.replaceFromToWithFrom(0, s - 1, other.values, 0);

        return this;
    }

    public IntMatrix2D assign(final IntMatrix2D y, cern.colt.function.tint.IntIntFunction function) {
        checkShape(y);

        if (function instanceof cern.jet.math.tint.IntPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final int alpha = ((cern.jet.math.tint.IntPlusMultSecond) function).multiplicator;
            if (alpha == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    setQuick(i, j, getQuick(i, j) + alpha * value);
                    return value;
                }
            });
            return this;
        }

        if (function instanceof cern.jet.math.tint.IntPlusMultFirst) { // x[i] = alpha*x[i] + y[i]
            final int alpha = ((cern.jet.math.tint.IntPlusMultFirst) function).multiplicator;
            if (alpha == 0)
                return assign(y);
            y.forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    setQuick(i, j, alpha * getQuick(i, j) + value);
                    return value;
                }
            });
            return this;
        }

        int np = ConcurrencyUtils.getNumberOfThreads();

        if (function == cern.jet.math.tint.IntFunctions.mult) { // x[i] = x[i] * y[i]
            final int[] indexesE = columnindexes.elements();
            final int[] valuesE = values.elements();

            if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future<?>[] futures = new Future[np];
                int k = (rowPointers.length - 1) / np;
                for (int j = 0; j < np; j++) {
                    final int startidx = j * k;
                    final int stopidx;
                    if (j == np - 1) {
                        stopidx = rowPointers.length - 1;
                    } else {
                        stopidx = startidx + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            for (int i = startidx; i < stopidx; i++) {
                                int high = rowPointers[i + 1];
                                for (int k = rowPointers[i]; k < high; k++) {
                                    int j = indexesE[k];
                                    valuesE[k] *= y.getQuick(i, j);
                                    if (valuesE[k] == 0)
                                        remove(i, j);
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int i = rowPointers.length - 1; --i >= 0;) {
                    int low = rowPointers[i];
                    for (int k = rowPointers[i + 1]; --k >= low;) {
                        int j = indexesE[k];
                        valuesE[k] *= y.getQuick(i, j);
                        if (valuesE[k] == 0)
                            remove(i, j);
                    }
                }
            }
            return this;
        }

        if (function == cern.jet.math.tint.IntFunctions.div) { // x[i] = x[i] / y[i]
            final int[] indexesE = columnindexes.elements();
            final int[] valuesE = values.elements();

            if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future<?>[] futures = new Future[np];
                int k = (rowPointers.length - 1) / np;
                for (int j = 0; j < np; j++) {
                    final int startidx = j * k;
                    final int stopidx;
                    if (j == np - 1) {
                        stopidx = rowPointers.length - 1;
                    } else {
                        stopidx = startidx + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            for (int i = startidx; i < stopidx; i++) {
                                int high = rowPointers[i + 1];
                                for (int k = rowPointers[i]; k < high; k++) {
                                    int j = indexesE[k];
                                    valuesE[k] /= y.getQuick(i, j);
                                    if (valuesE[k] == 0)
                                        remove(i, j);
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int i = rowPointers.length - 1; --i >= 0;) {
                    int low = rowPointers[i];
                    for (int k = rowPointers[i + 1]; --k >= low;) {
                        int j = indexesE[k];
                        valuesE[k] /= y.getQuick(i, j);
                        if (valuesE[k] == 0)
                            remove(i, j);
                    }
                }
            }
            return this;
        }

        return super.assign(y, function);
    }

    public IntMatrix2D forEachNonZero(final cern.colt.function.tint.IntIntIntFunction function) {
        final int[] indexesE = columnindexes.elements();
        final int[] valuesE = values.elements();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future<?>[] futures = new Future[np];
            int k = (rowPointers.length - 1) / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = rowPointers.length - 1;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = startidx; i < stopidx; i++) {
                            int high = rowPointers[i + 1];
                            for (int k = rowPointers[i]; k < high; k++) {
                                int j = indexesE[k];
                                int value = valuesE[k];
                                int r = function.apply(i, j, value);
                                if (r != value)
                                    valuesE[k] = r;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = rowPointers.length - 1; --i >= 0;) {
                int low = rowPointers[i];
                for (int k = rowPointers[i + 1]; --k >= low;) {
                    int j = indexesE[k];
                    int value = valuesE[k];
                    int r = function.apply(i, j, value);
                    if (r != value)
                        valuesE[k] = r;
                }
            }
        }
        return this;
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     */
    protected IntMatrix2D getContent() {
        return this;
    }

    public IntArrayList getColumnindexes() {
        return columnindexes;
    }

    public int[] getRowPointers() {
        return rowPointers;
    }

    public IntArrayList getValues() {
        return values;
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
    public int getQuick(int row, int column) {
        int k = columnindexes.binarySearchFromTo(column, rowPointers[row], rowPointers[row + 1] - 1);
        int v = 0;
        if (k >= 0)
            v = values.getQuick(k);
        return v;
    }

    protected synchronized void insert(int row, int column, int index, int value) {
        columnindexes.beforeInsert(index, column);
        values.beforeInsert(index, value);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]++;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseIntMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseIntMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseIntMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseIntMatrix2D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public IntMatrix2D like(int rows, int columns) {
        return new RCIntMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirely independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseIntMatrix2D</tt> the new
     * matrix must be of type <tt>DenseIntMatrix1D</tt>, if the receiver is
     * an instance of type <tt>SparseIntMatrix2D</tt> the new matrix must be
     * of type <tt>SparseIntMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public IntMatrix1D like1D(int size) {
        return new SparseIntMatrix1D(size);
    }

    protected void remove(int row, int index) {
        columnindexes.remove(index);
        values.remove(index);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]--;
    }

    public int cardinality() {
        return columnindexes.size();
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
     * @param value
     *            the value to be filled into the specified cell.
     */
    public synchronized void setQuick(int row, int column, int value) {
        int k = columnindexes.binarySearchFromTo(column, rowPointers[row], rowPointers[row + 1] - 1);
        if (k >= 0) { // found
            if (value == 0)
                remove(row, k);
            else
                values.setQuick(k, value);
            return;
        }

        if (value != 0) {
            k = -k - 1;
            insert(row, column, k, value);
        }
    }

    public DenseIntMatrix2D getFull() {
        final DenseIntMatrix2D full = new DenseIntMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
            public int apply(int i, int j, int value) {
                full.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return full;
    }

    public void trimToSize() {
        columnindexes.trimToSize();
        values.trimToSize();
    }

    public IntMatrix1D zMult(IntMatrix1D y, IntMatrix1D z) {
        int m = rows;
        int n = columns;

        if (z == null)
            z = new DenseIntMatrix1D(m);

        if (!(y instanceof DenseIntMatrix1D && z instanceof DenseIntMatrix1D)) {
            return super.zMult(y, z);
        }

        if (n != y.size() || m > z.size())
            throw new IllegalArgumentException("Incompatible args: " + this.toStringShort() + ", " + y.toStringShort() + ", " + z.toStringShort());

        DenseIntMatrix1D zz = (DenseIntMatrix1D) z;
        final int[] zElements = zz.elements;
        final int zStride = zz.stride();
        final int zi = (int)z.index(0);

        DenseIntMatrix1D yy = (DenseIntMatrix1D) y;
        final int[] yElements = yy.elements;
        final int yStride = yy.stride();
        final int yi = (int)y.index(0);

        final int[] columnindexesElements = columnindexes.elements();
        final int[] valuesElements = values.elements();

        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future<?>[] futures = new Future[np];
            int k = rowPointers.length / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rowPointers.length - 1;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int zidx = zi + startrow * zStride;
                        for (int i = startrow; i < stoprow; i++) {
                            int high = rowPointers[i + 1];
                            int sum = 0;
                            for (int k = rowPointers[i]; k < high; k++) {
                                int j = columnindexesElements[k];
                                sum += valuesElements[k] * yElements[yi + yStride * j];
                            }
                            zElements[zidx] = sum;
                            zidx += zStride;
                        }

                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int s = rowPointers.length - 1;
            int zidx = zi;
            for (int i = 0; i < s; i++) {
                int high = rowPointers[i + 1];
                int sum = 0;
                for (int k = rowPointers[i]; k < high; k++) {
                    int j = columnindexesElements[k];
                    sum += valuesElements[k] * yElements[yi + yStride * j];
                }
                zElements[zidx] = sum;
                zidx += zStride;
            }
        }
        return z;
    }

    public IntMatrix1D zMult(IntMatrix1D y, IntMatrix1D z, final int alpha, final int beta, final boolean transposeA) {
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }

        boolean ignore = (z == null || !transposeA);
        if (z == null)
            z = new DenseIntMatrix1D(m);

        if (!(y instanceof DenseIntMatrix1D && z instanceof DenseIntMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (n != y.size() || m > z.size())
            throw new IllegalArgumentException("Incompatible args: " + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", " + z.toStringShort());

        DenseIntMatrix1D zz = (DenseIntMatrix1D) z;
        final int[] zElements = zz.elements;
        final int zStride = zz.stride();
        final int zi = (int)z.index(0);

        DenseIntMatrix1D yy = (DenseIntMatrix1D) y;
        final int[] yElements = yy.elements;
        final int yStride = yy.stride();
        final int yi = (int)y.index(0);

        if (transposeA) {
            if ((!ignore) && (beta != 1.0))
                z.assign(cern.jet.math.tint.IntFunctions.mult(beta));
        }

        final int[] idx = columnindexes.elements();
        final int[] vals = values.elements();
        int s = rowPointers.length - 1;
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future<?>[] futures = new Future[np];
            int k = rowPointers.length / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rowPointers.length - 1;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int zidx = zi + startrow * zStride;
                        if (!transposeA) {
                            if (beta == 0.0) {
                                for (int i = startrow; i < stoprow; i++) {
                                    int high = rowPointers[i + 1];
                                    int sum = 0;
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = idx[k];
                                        sum += vals[k] * yElements[yi + yStride * j];
                                    }
                                    zElements[zidx] = alpha * sum;
                                    zidx += zStride;
                                }
                            } else {
                                for (int i = startrow; i < stoprow; i++) {
                                    int high = rowPointers[i + 1];
                                    int sum = 0;
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = idx[k];
                                        sum += vals[k] * yElements[yi + yStride * j];
                                    }
                                    zElements[zidx] = alpha * sum + beta * zElements[zidx];
                                    zidx += zStride;
                                }
                            }
                        } else {
                            for (int i = startrow; i < stoprow; i++) {
                                int high = rowPointers[i + 1];
                                int yElem = alpha * yElements[yi + yStride * i];
                                for (int k = rowPointers[i]; k < high; k++) {
                                    int j = idx[k];
                                    zElements[zi + zStride * j] += vals[k] * yElem;
                                }
                            }
                        }

                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int zidx = zi;
            if (!transposeA) {
                if (beta == 0.0) {
                    for (int i = 0; i < s; i++) {
                        int high = rowPointers[i + 1];
                        int sum = 0;
                        for (int k = rowPointers[i]; k < high; k++) {
                            int j = idx[k];
                            sum += vals[k] * yElements[yi + yStride * j];
                        }
                        zElements[zidx] = alpha * sum;
                        zidx += zStride;
                    }
                } else {
                    for (int i = 0; i < s; i++) {
                        int high = rowPointers[i + 1];
                        int sum = 0;
                        for (int k = rowPointers[i]; k < high; k++) {
                            int j = idx[k];
                            sum += vals[k] * yElements[yi + yStride * j];
                        }
                        zElements[zidx] = alpha * sum + beta * zElements[zidx];
                        zidx += zStride;
                    }
                }
            } else {
                for (int i = 0; i < s; i++) {
                    int high = rowPointers[i + 1];
                    int yElem = alpha * yElements[yi + yStride * i];
                    for (int k = rowPointers[i]; k < high; k++) {
                        int j = idx[k];
                        zElements[zi + zStride * j] += vals[k] * yElem;
                    }
                }
            }
        }

        return z;
    }

    public IntMatrix2D zMult(IntMatrix2D B, IntMatrix2D C) {
        int m = rows;
        int n = columns;
        int p = B.columns();
        if (C == null)
            C = new DenseIntMatrix2D(m, p);

        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + B.toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatible result matrix: " + toStringShort() + ", " + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        // cache views
        final IntMatrix1D[] Brows = new IntMatrix1D[n];
        for (int i = n; --i >= 0;)
            Brows[i] = B.viewRow(i);
        final IntMatrix1D[] Crows = new IntMatrix1D[m];
        for (int i = m; --i >= 0;)
            Crows[i] = C.viewRow(i);

        final cern.jet.math.tint.IntPlusMultSecond fun = cern.jet.math.tint.IntPlusMultSecond.plusMult(0);

        final int[] indexesE = columnindexes.elements();
        final int[] valuesE = values.elements();
        for (int i = rowPointers.length - 1; --i >= 0;) {
            int low = rowPointers[i];
            for (int k = rowPointers[i + 1]; --k >= low;) {
                int j = indexesE[k];
                fun.multiplicator = valuesE[k];
                Crows[i].assign(Brows[j], fun);
            }
        }
        return C;
    }

    public IntMatrix2D zMult(IntMatrix2D B, IntMatrix2D C, final int alpha, int beta, final boolean transposeA, boolean transposeB) {
        if (transposeB)
            B = B.viewDice();
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }
        int p = B.columns();
        boolean ignore = (C == null);
        if (C == null)
            C = new DenseIntMatrix2D(m, p);

        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatible result matrix: " + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!ignore)
            C.assign(cern.jet.math.tint.IntFunctions.mult(beta));

        // cache views
        final IntMatrix1D[] Brows = new IntMatrix1D[n];
        for (int i = n; --i >= 0;)
            Brows[i] = B.viewRow(i);
        final IntMatrix1D[] Crows = new IntMatrix1D[m];
        for (int i = m; --i >= 0;)
            Crows[i] = C.viewRow(i);

        final cern.jet.math.tint.IntPlusMultSecond fun = cern.jet.math.tint.IntPlusMultSecond.plusMult(0);

        final int[] indexesE = columnindexes.elements();
        final int[] valuesE = values.elements();
        for (int i = rowPointers.length - 1; --i >= 0;) {
            int low = rowPointers[i];
            for (int k = rowPointers[i + 1]; --k >= low;) {
                int j = indexesE[k];
                fun.multiplicator = valuesE[k] * alpha;
                if (!transposeA)
                    Crows[i].assign(Brows[j], fun);
                else
                    Crows[j].assign(Brows[i], fun);
            }
        }
        return C;
    }
}
