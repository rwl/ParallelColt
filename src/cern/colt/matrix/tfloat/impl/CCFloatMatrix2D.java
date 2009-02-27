/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import java.util.concurrent.Future;

import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse row-compressed 2-d matrix holding <tt>float</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally uses the standard sparse row-compressed format, with two important
 * differences that broaden the applicability of this storage format:
 * <ul>
 * <li>We use a {@link cern.colt.list.tint.IntArrayList} and
 * {@link cern.colt.list.tfloat.FloatArrayList} to hold the column indexes and
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
 * A.forEachNonZero(new cern.colt.function.IntIntFloatFunction() {
 *     public float apply(int row, int column, float value) {
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
 * B.forEachNonZero(new cern.colt.function.IntIntFloatFunction() {
 *     public float apply(int row, int column, float value) {
 *         A.setQuick(row, column, A.getQuick(row, column) + alpha * value);
 *         return value;
 *     }
 * });
 * </pre>
 * 
 * </td>
 * </table>
 * Method
 * {@link #assign(FloatMatrix2D,cern.colt.function.tfloat.FloatFloatFunction)}
 * does just that if you supply
 * {@link cern.jet.math.tfloat.FloatFunctions#plusMultSecond} as argument.
 * 
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 0.9, 04/14/2000
 */
public class CCFloatMatrix2D extends WrapperFloatMatrix2D {
    /*
     * The elements of the matrix.
     */
    protected IntArrayList rowIndexes;

    protected FloatArrayList values;

    protected int[] columnPointers;

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
    public CCFloatMatrix2D(float[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length);
        assign(values);
    }

    /**
     * Constructs a matrix with indexes and values given in compressed-column
     * format.
     * 
     * @param rows
     * @param columns
     * @param columnPointers
     * @param rowIndexes
     * @param values
     */
    public CCFloatMatrix2D(int rows, int columns, int[] columnPointers, IntArrayList rowIndexes, FloatArrayList values) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.columnPointers = columnPointers;
        this.rowIndexes = rowIndexes;
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public CCFloatMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        rowIndexes = new IntArrayList();
        values = new FloatArrayList();
        columnPointers = new int[columns + 1];
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public CCFloatMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        rowIndexes = new IntArrayList(nzmax);
        values = new FloatArrayList(nzmax);
        columnPointers = new int[columns + 1];
    }

    /**
     * Constructs a matrix with indexes and values given in coordinate format.
     * 
     * @param rows
     * @param columns
     * @param rowIndexes
     * @param columnIndexes
     * @param values
     */
    public CCFloatMatrix2D(int rows, int columns, IntArrayList rowIndexes, IntArrayList columnIndexes, FloatArrayList values) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        int nnz = rowIndexes.size();
        int[] rowIndexesElements = rowIndexes.elements();
        int[] columnIndexesElements = columnIndexes.elements();
        float[] valuesElements = values.elements();

        int[] idxs = new int[nnz];
        float[] vals = new float[nnz];
        int[] starts = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nnz; k++) {
            w[columnIndexesElements[k]]++;
        }
        cumsum(starts, w, columns);
        for (int k = 0; k < nnz; k++) {
            idxs[r = w[columnIndexesElements[k]]++] = rowIndexesElements[k];
            vals[r] = valuesElements[k];
        }
        this.columnPointers = starts;
        this.rowIndexes = new IntArrayList(idxs);
        this.values = new FloatArrayList(vals);
    }

    /**
     * Constructs a matrix with indexes given in coordinate format and single
     * value.
     * 
     * @param rows
     * @param columns
     * @param rowIndexes
     * @param columnIndexes
     * @param value
     */
    public CCFloatMatrix2D(int rows, int columns, IntArrayList rowIndexes, IntArrayList columnIndexes, float value) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        int nnz = rowIndexes.size();
        int[] rowIndexesElements = rowIndexes.elements();
        int[] columnIndexesElements = columnIndexes.elements();

        int[] idxs = new int[nnz];
        float[] vals = new float[nnz];
        int[] starts = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nnz; k++) {
            w[columnIndexesElements[k]]++;
        }
        cumsum(starts, w, columns);
        for (int k = 0; k < nnz; k++) {
            idxs[r = w[columnIndexesElements[k]]++] = rowIndexesElements[k];
            vals[r] = value;
        }
        this.columnPointers = starts;
        this.rowIndexes = new IntArrayList(idxs);
        this.values = new FloatArrayList(vals);
    }

    private float cumsum(int[] p, int[] c, int n) {
        int nz = 0;
        float nz2 = 0;
        for (int k = 0; k < n; k++) {
            p[k] = nz;
            nz += c[k];
            nz2 += c[k];
            c[k] = p[k];
        }
        p[n] = nz;
        return (nz2);
    }

    /**
     * Sets all nonzero cells to the state specified by <tt>value</tt>.
     * 
     * @param value
     *            the value to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     */
    public FloatMatrix2D assign(float value) {
        // overriden for performance only
        if (value == 0) {
            rowIndexes.clear();
            values.clear();
            columnPointers = new int[columns + 1];
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
     * @see cern.jet.math.tfloat.FloatFunctions
     */
    public FloatMatrix2D assign(final cern.colt.function.tfloat.FloatFunction function) {
        if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] = mult*x[i]
            final float alpha = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
            if (alpha == 1)
                return this;
            if (alpha == 0)
                return assign(0);
            if (alpha != alpha)
                return assign(alpha); // the funny definition of isNaN(). This should better not happen.

            final float[] valuesE = values.elements();
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
            forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
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
    public FloatMatrix2D assign(FloatMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);
        // overriden for performance only
        if (!(source instanceof CCFloatMatrix2D)) {
            assign(0);
            source.forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
            /*
             * indexes.clear(); values.clear(); int nonZeros=0; for (int row=0;
             * row<rows; row++) { starts[row]=nonZeros; for (int column=0;
             * column<columns; column++) { float v =
             * source.getQuick(row,column); if (v!=0) { values.add(v);
             * indexes.add(column); nonZeros++; } } } starts[rows]=nonZeros;
             */
            return this;
        }

        // even quicker
        CCFloatMatrix2D other = (CCFloatMatrix2D) source;

        System.arraycopy(other.columnPointers, 0, this.columnPointers, 0, this.columnPointers.length);
        int s = other.rowIndexes.size();
        this.rowIndexes.setSize(s);
        this.values.setSize(s);
        this.rowIndexes.replaceFromToWithFrom(0, s - 1, other.rowIndexes, 0);
        this.values.replaceFromToWithFrom(0, s - 1, other.values, 0);

        return this;
    }

    public FloatMatrix2D assign(final FloatMatrix2D y, cern.colt.function.tfloat.FloatFloatFunction function) {
        checkShape(y);

        if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final float alpha = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
            if (alpha == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    setQuick(i, j, getQuick(i, j) + alpha * value);
                    return value;
                }
            });
            return this;
        }

        int np = ConcurrencyUtils.getNumberOfThreads();

        if (function == cern.jet.math.tfloat.FloatFunctions.mult) { // x[i] = x[i] * y[i]
            final int[] indexesE = rowIndexes.elements();
            final float[] valuesE = values.elements();

            if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future<?>[] futures = new Future[np];
                int k = (columnPointers.length - 1) / np;
                for (int j = 0; j < np; j++) {
                    final int startidx = j * k;
                    final int stopidx;
                    if (j == np - 1) {
                        stopidx = columnPointers.length - 1;
                    } else {
                        stopidx = startidx + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            for (int j = startidx; j < stopidx; j++) {
                                int high = columnPointers[j + 1];
                                for (int k = columnPointers[j]; k < high; k++) {
                                    int i = indexesE[k];
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
                for (int j = columnPointers.length - 1; --j >= 0;) {
                    int low = columnPointers[j];
                    for (int k = columnPointers[j + 1]; --k >= low;) {
                        int i = indexesE[k];
                        valuesE[k] *= y.getQuick(i, j);
                        if (valuesE[k] == 0)
                            remove(i, j);
                    }
                }
            }
            return this;
        }

        if (function == cern.jet.math.tfloat.FloatFunctions.div) { // x[i] = x[i] / y[i]
            final int[] indexesE = rowIndexes.elements();
            final float[] valuesE = values.elements();

            if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future<?>[] futures = new Future[np];
                int k = (columnPointers.length - 1) / np;
                for (int j = 0; j < np; j++) {
                    final int startidx = j * k;
                    final int stopidx;
                    if (j == np - 1) {
                        stopidx = columnPointers.length - 1;
                    } else {
                        stopidx = startidx + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            for (int j = startidx; j < stopidx; j++) {
                                int high = columnPointers[j + 1];
                                for (int k = columnPointers[j]; k < high; k++) {
                                    int i = indexesE[k];
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
                for (int j = columnPointers.length - 1; --j >= 0;) {
                    int low = columnPointers[j];
                    for (int k = columnPointers[j + 1]; --k >= low;) {
                        int i = indexesE[k];
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

    public FloatMatrix2D forEachNonZero(final cern.colt.function.tfloat.IntIntFloatFunction function) {
        final int[] indexesE = rowIndexes.elements();
        final float[] valuesE = values.elements();
        int np = ConcurrencyUtils.getNumberOfThreads();
        if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future<?>[] futures = new Future[np];
            int k = (columnPointers.length - 1) / np;
            for (int j = 0; j < np; j++) {
                final int startidx = j * k;
                final int stopidx;
                if (j == np - 1) {
                    stopidx = columnPointers.length - 1;
                } else {
                    stopidx = startidx + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int j = startidx; j < stopidx; j++) {
                            int high = columnPointers[j + 1];
                            for (int k = columnPointers[j]; k < high; k++) {
                                int i = indexesE[k];
                                float value = valuesE[k];
                                float r = function.apply(i, j, value);
                                if (r != value)
                                    valuesE[k] = r;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int j = columnPointers.length - 1; --j >= 0;) {
                int low = columnPointers[j];
                for (int k = columnPointers[j + 1]; --k >= low;) {
                    int i = indexesE[k];
                    float value = valuesE[k];
                    float r = function.apply(i, j, value);
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
    protected FloatMatrix2D getContent() {
        return this;
    }

    public IntArrayList getRowindexes() {
        return rowIndexes;
    }

    public int[] getColumnPointers() {
        return columnPointers;
    }

    public FloatArrayList getValues() {
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
    public float getQuick(int row, int column) {
        int k = rowIndexes.binarySearchFromTo(row, columnPointers[column], columnPointers[column + 1] - 1);
        //        int k = searchFromTo(rowIndexes.elements(), row, columnPointers[column], columnPointers[column + 1] - 1);

        float v = 0;
        if (k >= 0)
            v = values.getQuick(k);
        return v;
    }

    protected synchronized void insert(int row, int column, int index, float value) {
        rowIndexes.beforeInsert(index, row);
        values.beforeInsert(index, value);
        for (int i = columnPointers.length; --i > column;)
            columnPointers[i]++;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseFloatMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseFloatMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseFloatMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseFloatMatrix2D</tt>, etc. In general, the new matrix should have
     * internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public FloatMatrix2D like(int rows, int columns) {
        return new CCFloatMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirely independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseFloatMatrix2D</tt> the new
     * matrix must be of type <tt>DenseFloatMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseFloatMatrix2D</tt> the new matrix must be of
     * type <tt>SparseFloatMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public FloatMatrix1D like1D(int size) {
        return new SparseFloatMatrix1D(size);
    }

    protected void remove(int column, int index) {
        rowIndexes.remove(index);
        values.remove(index);
        for (int i = columnPointers.length; --i > column;)
            columnPointers[i]--;
    }

    public int cardinality() {
        return rowIndexes.size();
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
    public synchronized void setQuick(int row, int column, float value) {
        int k = rowIndexes.binarySearchFromTo(row, columnPointers[column], columnPointers[column + 1] - 1);
        //        int k = searchFromTo(rowIndexes.elements(), row, columnPointers[column], columnPointers[column + 1] - 1);

        if (k >= 0) { // found
            if (value == 0)
                remove(column, k);
            else
                values.setQuick(k, value);
            return;
        }

        if (value != 0) {
            k = -k - 1;
            insert(row, column, k, value);
        }
    }

    public DenseFloatMatrix2D getFull() {
        final DenseFloatMatrix2D full = new DenseFloatMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
            public float apply(int i, int j, float value) {
                full.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return full;
    }

    public void trimToSize() {
        rowIndexes.trimToSize();
        values.trimToSize();
    }

    public FloatMatrix1D zMult(FloatMatrix1D y, FloatMatrix1D z, final float alpha, final float beta, final boolean transposeA) {
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }

        boolean ignore = (z == null || transposeA);
        if (z == null)
            z = new DenseFloatMatrix1D(m);

        if (!(y instanceof DenseFloatMatrix1D && z instanceof DenseFloatMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (n != y.size() || m > z.size())
            throw new IllegalArgumentException("Incompatible args: " + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", " + z.toStringShort());

        DenseFloatMatrix1D zz = (DenseFloatMatrix1D) z;
        final float[] zElements = zz.elements;
        final int zStride = zz.stride();
        final int zi = (int) zz.index(0);

        DenseFloatMatrix1D yy = (DenseFloatMatrix1D) y;
        final float[] yElements = yy.elements;
        final int yStride = yy.stride();
        final int yi = (int) yy.index(0);

        final int[] idx = rowIndexes.elements();
        final float[] vals = values.elements();

        int zidx = zi;

        if (!transposeA) {
            if ((!ignore) && (beta / alpha != 1.0)) {
                z.assign(cern.jet.math.tfloat.FloatFunctions.mult(beta / alpha));
            }
            for (int i = 0; i < columns; i++) {
                int high = columnPointers[i + 1];
                float yElem = yElements[yi + yStride * i];
                for (int k = columnPointers[i]; k < high; k++) {
                    int j = idx[k];
                    zElements[zi + zStride * j] += vals[k] * yElem;
                }
            }
            if (alpha != 1.0) {
                z.assign(cern.jet.math.tfloat.FloatFunctions.mult(alpha));
            }
        } else {
            int np = ConcurrencyUtils.getNumberOfThreads();
            if ((np > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future<?>[] futures = new Future[np];
                int k = columns / np;
                for (int j = 0; j < np; j++) {
                    final int startcolumn = j * k;
                    final int stopcolumn;
                    if (j == np - 1) {
                        stopcolumn = columns;
                    } else {
                        stopcolumn = startcolumn + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int zidx = zi + startcolumn * zStride;
                            for (int i = startcolumn; i < stopcolumn; i++) {
                                int high = columnPointers[i + 1];
                                float sum = 0;
                                for (int k = columnPointers[i]; k < high; k++) {
                                    int j = idx[k];
                                    sum += vals[k] * yElements[yi + yStride * j];
                                }
                                zElements[zidx] = alpha * sum + zElements[zidx] * beta;
                                zidx += zStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int i = 0; i < columns; i++) {
                    int high = columnPointers[i + 1];
                    float sum = 0;
                    for (int k = columnPointers[i]; k < high; k++) {
                        int j = idx[k];
                        sum += vals[k] * yElements[yi + yStride * j];
                    }
                    zElements[zidx] = alpha * sum + zElements[zidx] * beta;
                    zidx += zStride;
                }
            }
        }
        return z;
    }

    public FloatMatrix2D zMult(FloatMatrix2D B, FloatMatrix2D C, final float alpha, float beta, final boolean transposeA, boolean transposeB) {
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
            C = new DenseFloatMatrix2D(m, p);

        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatible result matrix: " + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!ignore)
            C.assign(cern.jet.math.tfloat.FloatFunctions.mult(beta));

        // cache views
        final FloatMatrix1D[] Brows = new FloatMatrix1D[n];
        for (int i = n; --i >= 0;)
            Brows[i] = B.viewRow(i);
        final FloatMatrix1D[] Crows = new FloatMatrix1D[m];
        for (int i = m; --i >= 0;)
            Crows[i] = C.viewRow(i);

        final cern.jet.math.tfloat.FloatPlusMultSecond fun = cern.jet.math.tfloat.FloatPlusMultSecond.plusMult(0);

        final int[] indexesE = rowIndexes.elements();
        final float[] valuesE = values.elements();
        for (int i = columns; --i >= 0;) {
            int low = columnPointers[i];
            for (int k = columnPointers[i + 1]; --k >= low;) {
                int j = indexesE[k];
                fun.multiplicator = valuesE[k] * alpha;
                if (!transposeA)
                    Crows[j].assign(Brows[i], fun);
                else
                    Crows[i].assign(Brows[j], fun);
            }
        }
        return C;
    }

    //    private static int searchFromTo(int[] list, int key, int from, int to) {
    //        while (from <= to) {
    //            if (list[from] == key) {
    //                return from;
    //            } else {
    //                from++;
    //                continue;
    //            }
    //        }
    //        return -(from + 1); // key not found.
    //    }

}
