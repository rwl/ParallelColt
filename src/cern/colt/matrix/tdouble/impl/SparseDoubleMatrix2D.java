/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import java.io.IOException;

import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.map.tdouble.AbstractLongDoubleMap;
import cern.colt.map.tdouble.OpenLongDoubleHashMap;
import cern.colt.matrix.io.MatrixInfo;
import cern.colt.matrix.io.MatrixSize;
import cern.colt.matrix.io.MatrixVectorReader;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;

/**
 * Sparse hashed 2-d matrix holding <tt>double</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Note that this implementation is not synchronized. Uses a
 * {@link cern.colt.map.tdouble.OpenIntDoubleHashMap}, which is a compact and
 * performant hashing technique.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * Cells that
 * <ul>
 * <li>are never set to non-zero values do not use any memory.
 * <li>switch from zero to non-zero state do use memory.
 * <li>switch back from non-zero to zero state also do use memory. However,
 * their memory is automatically reclaimed from time to time. It can also
 * manually be reclaimed by calling {@link #trimToSize()}.
 * </ul>
 * <p>
 * worst case: <tt>memory [bytes] = (1/minLoadFactor) * nonZeros * 13</tt>. <br>
 * best case: <tt>memory [bytes] = (1/maxLoadFactor) * nonZeros * 13</tt>. <br>
 * Where <tt>nonZeros = cardinality()</tt> is the number of non-zero cells.
 * Thus, a 1000 x 1000 matrix with minLoadFactor=0.25 and maxLoadFactor=0.5 and
 * 1000000 non-zero cells consumes between 25 MB and 50 MB. The same 1000 x 1000
 * matrix with 1000 non-zero cells consumes between 25 and 50 KB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * This class offers <i>expected</i> time complexity <tt>O(1)</tt> (i.e.
 * constant time) for the basic operations <tt>get</tt>, <tt>getQuick</tt>,
 * <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt> assuming the hash function
 * disperses the elements properly among the buckets. Otherwise, pathological
 * cases, although highly improbable, can occur, degrading performance to
 * <tt>O(N)</tt> in the worst case. As such this sparse class is expected to
 * have no worse time complexity than its dense counterpart
 * {@link DenseDoubleMatrix2D}. However, constant factors are considerably
 * larger.
 * <p>
 * Cells are internally addressed in row-major. Performance sensitive
 * applications can exploit this fact. Setting values in a loop row-by-row is
 * quicker than column-by-column, because fewer hash collisions occur. Thus
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
 * @see cern.colt.map
 * @see cern.colt.map.tdouble.OpenIntDoubleHashMap
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.1, 08/22/2007
 */
public class SparseDoubleMatrix2D extends DoubleMatrix2D {
    /*
     * The elements of the matrix.
     */
    protected AbstractLongDoubleMap elements;

    protected int dummy;

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
    public SparseDoubleMatrix2D(double[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of rows and columns and default
     * memory usage. All entries are initially <tt>0</tt>.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseDoubleMatrix2D(int rows, int columns) {
        this(rows, columns, rows * (columns / 1000), 0.2, 0.5);
    }

    /**
     * Constructs a matrix with a given number of rows and columns using memory
     * as specified. All entries are initially <tt>0</tt>. For details related
     * to memory usage see {@link cern.colt.map.tdouble.OpenIntDoubleHashMap}.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param initialCapacity
     *            the initial capacity of the hash map. If not known, set
     *            <tt>initialCapacity=0</tt> or small.
     * @param minLoadFactor
     *            the minimum load factor of the hash map.
     * @param maxLoadFactor
     *            the maximum load factor of the hash map.
     * @throws IllegalArgumentException
     *             if
     * 
     *             <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) || (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >= maxLoadFactor)</tt>
     *             .
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseDoubleMatrix2D(int rows, int columns, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = new OpenLongDoubleHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
    }

    /**
     * Constructs a matrix with a copy of the given indexes and values.
     * 
     * @param rows
     * @param columns
     * @param rowIndexes
     * @param columnIndexes
     * @param values
     */
    public SparseDoubleMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double[] values) {
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = new OpenLongDoubleHashMap(rowIndexes.length);
        insert(rowIndexes, columnIndexes, values);
    }

    /**
     * Constructs a matrix with a copy of the given indexes and value.
     * 
     * @param rows
     * @param columns
     * @param rowIndexes
     * @param columnIndexes
     * @param value
     */
    public SparseDoubleMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double value) {
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = new OpenLongDoubleHashMap(rowIndexes.length);
        insert(rowIndexes, columnIndexes, value);
    }

    /**
     * Constructs a matrix from MatrixVectorReader.
     * 
     * @param r
     *            matrix reader
     * @throws IOException
     */
    public SparseDoubleMatrix2D(MatrixVectorReader r) throws IOException {
        MatrixInfo info;
        if (r.hasInfo())
            info = r.readMatrixInfo();
        else
            info = new MatrixInfo(true, MatrixInfo.MatrixField.Real, MatrixInfo.MatrixSymmetry.General);

        if (info.isPattern())
            throw new UnsupportedOperationException("Pattern matrices are not supported");
        if (info.isDense())
            throw new UnsupportedOperationException("Dense matrices are not supported");
        if (info.isComplex())
            throw new UnsupportedOperationException("Complex matrices are not supported");

        MatrixSize size = r.readMatrixSize(info);
        try {
            setUp(size.numRows(), size.numColumns());
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        int numEntries = size.numEntries();
        int[] columnIndexes = new int[numEntries];
        int[] rowIndexes = new int[numEntries];
        double[] values = new double[numEntries];
        r.readCoordinate(rowIndexes, columnIndexes, values);
        if (info.isSymmetric() || info.isSkewSymmetric()) {
            this.elements = new OpenLongDoubleHashMap(2 * rowIndexes.length);
        } else {
            this.elements = new OpenLongDoubleHashMap(rowIndexes.length);
        }
        insert(rowIndexes, columnIndexes, values);

        if (info.isSymmetric()) {
            for (int i = 0; i < numEntries; i++) {
                if (rowIndexes[i] != columnIndexes[i]) {
                    set(columnIndexes[i], rowIndexes[i], values[i]);
                }
            }
        } else if (info.isSkewSymmetric()) {
            for (int i = 0; i < numEntries; i++) {
                if (rowIndexes[i] != columnIndexes[i]) {
                    set(columnIndexes[i], rowIndexes[i], -values[i]);
                }
            }
        }
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
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    protected SparseDoubleMatrix2D(int rows, int columns, AbstractLongDoubleMap elements, int rowZero, int columnZero, int rowStride, int columnStride) {
        try {
            setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = elements;
        this.isNoView = false;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[row,col] = function(x[row,col])</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * 	 matrix = 2 x 2 matrix
     * 	 0.5 1.5      
     * 	 2.5 3.5
     * 
     * 	 // change each cell to its sine
     * 	 matrix.assign(cern.jet.math.Functions.sin);
     * 	 --&gt;
     * 	 2 x 2 matrix
     * 	 0.479426  0.997495 
     * 	 0.598472 -0.350783
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param function
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tdouble.DoubleFunctions
     */
    public DoubleMatrix2D assign(cern.colt.function.tdouble.DoubleFunction function) {
        if (this.isNoView && function instanceof cern.jet.math.tdouble.DoubleMult) { // x[i]
            // =
            // mult*x[i]
            this.elements.assign(function);
        } else {
            super.assign(function);
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
    public DoubleMatrix2D assign(double value) {
        // overriden for performance only
        if (this.isNoView && value == 0)
            this.elements.clear();
        else
            super.assign(value);
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
    public DoubleMatrix2D assign(DoubleMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof SparseDoubleMatrix2D)) {
            return super.assign(source);
        }
        SparseDoubleMatrix2D other = (SparseDoubleMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);

        if (this.isNoView && other.isNoView) { // quickest
            this.elements.assign(other.elements);
            return this;
        }
        return super.assign(source);
    }

    public DoubleMatrix2D assign(final DoubleMatrix2D y, cern.colt.function.tdouble.DoubleDoubleFunction function) {
        if (!this.isNoView)
            return super.assign(y, function);

        checkShape(y);

        if (function instanceof cern.jet.math.tdouble.DoublePlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final double alpha = ((cern.jet.math.tdouble.DoublePlusMultSecond) function).multiplicator;
            if (alpha == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tdouble.IntIntDoubleFunction() {
                public double apply(int i, int j, double value) {
                    setQuick(i, j, getQuick(i, j) + alpha * value);
                    return value;
                }
            });
        }

        if (function == cern.jet.math.tdouble.DoubleFunctions.mult) { // x[i] = x[i] * y[i]
            this.elements.forEachPair(new cern.colt.function.tdouble.LongDoubleProcedure() {
                public boolean apply(long key, double value) {
                    int i = (int) (key / columns);
                    int j = (int) (key % columns);
                    double r = value * y.getQuick(i, j);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        }

        if (function == cern.jet.math.tdouble.DoubleFunctions.div) { // x[i] = x[i] /
            // y[i]
            this.elements.forEachPair(new cern.colt.function.tdouble.LongDoubleProcedure() {
                public boolean apply(long key, double value) {
                    int i = (int) (key / columns);
                    int j = (int) (key % columns);
                    double r = value / y.getQuick(i, j);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        }
        return this;

    }

    public SparseDoubleMatrix2D assign(final int[] rowIndexes, final int[] columnIndexes, final double[] values, final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        int size = rowIndexes.length;
        if (function == cern.jet.math.tdouble.DoubleFunctions.plus) { // x[i] = x[i] + y[i]
            for (int i = 0; i < size; i++) {
                double value = values[i];
                long row = (long) rowIndexes[i];
                long column = (long) columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                double elem = elements.get(index);
                value += elem;
                if (value != 0) {
                    elements.put(index, value);
                } else {
                    elements.removeKey(index);
                }
            }
        } else {
            for (int i = 0; i < size; i++) {
                double value = values[i];
                long row = (long) rowIndexes[i];
                long column = (long) columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                double elem = elements.get(index);
                value = function.apply(elem, value);
                if (value != 0) {
                    elements.put(index, value);
                } else {
                    elements.removeKey(index);
                }
            }
        }
        return this;
    }

    public SparseDoubleMatrix2D assign(final int[] rowIndexes, final int[] columnIndexes, final double value, final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        int size = rowIndexes.length;
        if (function == cern.jet.math.tdouble.DoubleFunctions.plus) { // x[i] = x[i] + y[i]
            for (int i = 0; i < size; i++) {
                long row = (long) rowIndexes[i];
                long column = (long) columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                double elem = elements.get(index);
                double sum = elem + value;
                if (sum != 0) {
                    elements.put(index, sum);
                } else {
                    elements.removeKey(index);
                }
            }
        } else {
            for (int i = 0; i < size; i++) {
                long row = (long) rowIndexes[i];
                long column = (long) columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                double elem = elements.get(index);
                double result = function.apply(elem, value);
                if (result != 0) {
                    elements.put(index, result);
                } else {
                    elements.removeKey(index);
                }
            }
        }
        return this;
    }

    /**
     * Returns the number of cells having non-zero values.
     */
    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
    }

    public RCDoubleMatrix2D convertToRCDoubleMatrix2D() {
        int nnz = cardinality();
        LongArrayList keysList = new LongArrayList(elements.size());
        DoubleArrayList valuesList = new DoubleArrayList(elements.size());
        elements.pairsSortedByKey(keysList, valuesList);
        final long[] keys = keysList.elements();
        double[] values = valuesList.elements();
        final int[] rowIndexes = new int[nnz];
        final int[] columnIndexes = new int[nnz];

        int[] idxs = new int[nnz];
        double[] vals = new double[nnz];
        int[] starts = new int[rows + 1];
        final int[] w = new int[rows];
        int r;
        for (int k = 0; k < nnz; k++) {
            long key = keys[k];
            rowIndexes[k] = (int) (key / columns);
            columnIndexes[k] = (int) (key % columns);
            w[rowIndexes[k]]++;
        }
        cumsum(starts, w, rows);
        for (int k = 0; k < nnz; k++) {
            idxs[r = w[rowIndexes[k]]++] = columnIndexes[k];
            vals[r] = values[k];
        }
        RCDoubleMatrix2D M = new RCDoubleMatrix2D(rows, columns, starts, new IntArrayList(idxs), new DoubleArrayList(vals));
        return M;
    }

    public CCDoubleMatrix2D convertToCCDoubleMatrix2D() {
        int nnz = cardinality();
        LongArrayList keysList = new LongArrayList();
        DoubleArrayList valuesList = new DoubleArrayList();
        elements.pairsSortedByKey(keysList, valuesList);
        long[] keys = keysList.elements();
        double[] values = valuesList.elements();
        int[] rowIndexes = new int[nnz];
        int[] columnIndexes = new int[nnz];

        int[] idxs = new int[nnz];
        double[] vals = new double[nnz];
        int[] starts = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nnz; k++) {
            rowIndexes[k] = (int) (keys[k] / columns);
            columnIndexes[k] = (int) (keys[k] % columns);
            w[columnIndexes[k]]++;
        }
        cumsum(starts, w, columns);
        for (int k = 0; k < nnz; k++) {
            idxs[r = w[columnIndexes[k]]++] = rowIndexes[k];
            vals[r] = values[k];
        }
        CCDoubleMatrix2D M = new CCDoubleMatrix2D(rows, columns, starts, new IntArrayList(idxs), new DoubleArrayList(vals));
        return M;
    }

    public RCMDoubleMatrix2D convertToRCMDoubleMatrix2D() {
        RCMDoubleMatrix2D A = new RCMDoubleMatrix2D(rows, columns);
        int nnz = cardinality();
        long[] keys = elements.keys().elements();
        double[] values = elements.values().elements();
        for (int i = 0; i < nnz; i++) {
            int row = (int) (keys[i] / columns);
            int column = (int) (keys[i] % columns);
            A.setQuick(row, column, values[i]);
        }
        return A;
    }

    public CCMDoubleMatrix2D convertToCCMDoubleMatrix2D() {
        CCMDoubleMatrix2D A = new CCMDoubleMatrix2D(rows, columns);
        int nnz = cardinality();
        long[] keys = elements.keys().elements();
        double[] values = elements.values().elements();
        for (int i = 0; i < nnz; i++) {
            int row = (int) (keys[i] / columns);
            int column = (int) (keys[i] % columns);
            A.setQuick(row, column, values[i]);
        }
        return A;
    }

    /**
     * Returns the elements of this matrix.
     * 
     * @return the elements
     */
    public AbstractLongDoubleMap elements() {
        return elements;
    }

    /**
     * Ensures that the receiver can hold at least the specified number of
     * non-zero cells without needing to allocate new internal memory. If
     * necessary, allocates new internal memory and increases the capacity of
     * the receiver.
     * <p>
     * This method never need be called; it is for performance tuning only.
     * Calling this method before tt>set()</tt>ing a large number of non-zero
     * values boosts performance, because the receiver will grow only once
     * instead of potentially many times and hash collisions get less probable.
     * 
     * @param minCapacity
     *            the desired minimum number of non-zero cells.
     */
    public void ensureCapacity(int minCapacity) {
        this.elements.ensureCapacity(minCapacity);
    }

    public DoubleMatrix2D forEachNonZero(final cern.colt.function.tdouble.IntIntDoubleFunction function) {
        if (this.isNoView) {
            this.elements.forEachPair(new cern.colt.function.tdouble.LongDoubleProcedure() {
                public boolean apply(long key, double value) {
                    int i = (int) (key / columns);
                    int j = (int) (key % columns);
                    double r = function.apply(i, j, value);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        } else {
            super.forEachNonZero(function);
        }
        return this;
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
    public double getQuick(int row, int column) {
        // if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
        // throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
        // return this.elements.get(index(row,column));
        // manually inlined:
        return this.elements.get((long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column * (long) columnStride);
    }

    /**
     * Returns the position of the given coordinate within the (virtual or
     * non-virtual) internal 1-dimensional array.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     */
    public long index(int row, int column) {
        // return super.index(row,column);
        // manually inlined for speed:
        return (long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseDoubleMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseDoubleMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseDoubleMatrix2D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public DoubleMatrix2D like(int rows, int columns) {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the new
     * matrix must be of type <tt>DenseDoubleMatrix1D</tt>, if the receiver is
     * an instance of type <tt>SparseDoubleMatrix2D</tt> the new matrix must be
     * of type <tt>SparseDoubleMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public DoubleMatrix1D like1D(int size) {
        return new SparseDoubleMatrix1D(size);
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
    public synchronized void setQuick(int row, int column, double value) {
        // if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
        // throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
        // int index = index(row,column);
        // manually inlined:
        long index = (long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;

        // if (value == 0 || Math.abs(value) < TOLERANCE)
        if (value == 0)
            this.elements.removeKey(index);
        else
            this.elements.put(index, value);
    }

    /**
     * Releases any superfluous memory created by explicitly putting zero values
     * into cells formerly having non-zero values; An application can use this
     * operation to minimize the storage of the receiver.
     * <p>
     * <b>Background:</b>
     * <p>
     * Cells that
     * <ul>
     * <li>are never set to non-zero values do not use any memory.
     * <li>switch from zero to non-zero state do use memory.
     * <li>switch back from non-zero to zero state also do use memory. However,
     * their memory can be reclaimed by calling <tt>trimToSize()</tt>.
     * </ul>
     * A sequence like <tt>set(r,c,5); set(r,c,0);</tt> sets a cell to non-zero
     * state and later back to zero state. Such as sequence generates obsolete
     * memory that is automatically reclaimed from time to time or can manually
     * be reclaimed by calling <tt>trimToSize()</tt>. Putting zeros into cells
     * already containing zeros does not generate obsolete memory since no
     * memory was allocated to them in the first place.
     */
    public void trimToSize() {
        this.elements.trimToSize();
    }

    /**
     * Returns a vector obtained by stacking the columns of the matrix on top of
     * one another.
     * 
     * @return a vector obtained by stacking the columns of the matrix on top of
     *         one another
     */
    public DoubleMatrix1D vectorize() {
        SparseDoubleMatrix1D v = new SparseDoubleMatrix1D(size());
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                double elem = getQuick(r, c);
                if (elem != 0) {
                    v.setQuick(idx++, elem);
                }
            }
        }
        return v;
    }

    public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, double alpha, double beta, final boolean transposeA) {
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }

        boolean ignore = (z == null);
        if (z == null)
            z = new DenseDoubleMatrix1D(m);

        if (!(this.isNoView && y instanceof DenseDoubleMatrix1D && z instanceof DenseDoubleMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (n != y.size() || m > z.size())
            throw new IllegalArgumentException("Incompatible args: " + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", " + z.toStringShort());

        if (!ignore)
            z.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta / alpha));

        DenseDoubleMatrix1D zz = (DenseDoubleMatrix1D) z;
        final double[] zElements = zz.elements;
        final int zStride = zz.stride();
        final int zi = (int) z.index(0);

        DenseDoubleMatrix1D yy = (DenseDoubleMatrix1D) y;
        final double[] yElements = yy.elements;
        final int yStride = yy.stride();
        final int yi = (int) y.index(0);

        if (yElements == null || zElements == null)
            throw new InternalError();

        this.elements.forEachPair(new cern.colt.function.tdouble.LongDoubleProcedure() {
            public boolean apply(long key, double value) {
                int i = (int) (key / columns);
                int j = (int) (key % columns);
                if (transposeA) {
                    int tmp = i;
                    i = j;
                    j = tmp;
                }
                zElements[zi + zStride * i] += value * yElements[yi + yStride * j];
                // System.out.println("["+i+","+j+"]-->"+value);
                return true;
            }
        });

        /*
         * forEachNonZero( new cern.colt.function.IntIntDoubleFunction() {
         * public double apply(int i, int j, double value) { if (transposeA) {
         * int tmp=i; i=j; j=tmp; } zElements[zi + zStride*i] += value *
         * yElements[yi + yStride*j]; //z.setQuick(row,z.getQuick(row) + value *
         * y.getQuick(column)); //System.out.println("["+i+","+j+"]-->"+value);
         * return value; } } );
         */

        if (alpha != 1)
            z.assign(cern.jet.math.tdouble.DoubleFunctions.mult(alpha));
        return z;
    }

    public DoubleMatrix2D zMult(DoubleMatrix2D B, DoubleMatrix2D C, final double alpha, double beta, final boolean transposeA, boolean transposeB) {
        if (!(this.isNoView)) {
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        }
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
            C = new DenseDoubleMatrix2D(m, p);

        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!ignore)
            C.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta));

        // cache views
        final DoubleMatrix1D[] Brows = new DoubleMatrix1D[n];
        for (int i = n; --i >= 0;)
            Brows[i] = B.viewRow(i);
        final DoubleMatrix1D[] Crows = new DoubleMatrix1D[m];
        for (int i = m; --i >= 0;)
            Crows[i] = C.viewRow(i);

        final cern.jet.math.tdouble.DoublePlusMultSecond fun = cern.jet.math.tdouble.DoublePlusMultSecond.plusMult(0);

        this.elements.forEachPair(new cern.colt.function.tdouble.LongDoubleProcedure() {
            public boolean apply(long key, double value) {
                int i = (int) (key / columns);
                int j = (int) (key % columns);
                fun.multiplicator = value * alpha;
                if (!transposeA)
                    Crows[i].assign(Brows[j], fun);
                else
                    Crows[j].assign(Brows[i], fun);
                return true;
            }
        });

        return C;
    }

    /**
     * Returns <tt>true</tt> if both matrices share common cells. More formally,
     * returns <tt>true</tt> if at least one of the following conditions is met
     * <ul>
     * <li>the receiver is a view of the other matrix
     * <li>the other matrix is a view of the receiver
     * <li><tt>this == other</tt>
     * </ul>
     */
    protected boolean haveSharedCellsRaw(DoubleMatrix2D other) {
        if (other instanceof SelectedSparseDoubleMatrix2D) {
            SelectedSparseDoubleMatrix2D otherMatrix = (SelectedSparseDoubleMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseDoubleMatrix2D) {
            SparseDoubleMatrix2D otherMatrix = (SparseDoubleMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseDoubleMatrix2D</tt> the new matrix must be of
     * type <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type
     * <tt>SparseDoubleMatrix2D</tt> the new matrix must be of type
     * <tt>SparseDoubleMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param offset
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @return a new matrix of the corresponding dynamic type.
     */
    protected DoubleMatrix1D like1D(int size, int offset, int stride) {
        return new SparseDoubleMatrix1D(size, this.elements, (int) offset, stride);
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
    protected DoubleMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseDoubleMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }

    private void insert(int[] rowIndexes, int[] columnIndexes, double value) {
        int size = rowIndexes.length;
        for (int i = 0; i < size; i++) {
            long row = (long) rowIndexes[i];
            long column = (long) columnIndexes[i];
            if (row >= rows || column >= columns) {
                throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
            }
            if (value != 0) {
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                double elem = elements.get(index);
                if (elem == 0) {
                    elements.put(index, value);
                } else {
                    double sum = elem + value;
                    if (sum == 0) {
                        elements.removeKey(index);
                    } else {
                        elements.put(index, sum);
                    }
                }
            }
        }
    }

    private void insert(int[] rowIndexes, int[] columnIndexes, double[] values) {
        int size = rowIndexes.length;
        for (int i = 0; i < size; i++) {
            double value = values[i];
            long row = (long) rowIndexes[i];
            long column = (long) columnIndexes[i];
            if (row >= rows || column >= columns) {
                throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
            }
            if (value != 0) {
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                double elem = elements.get(index);
                if (elem == 0) {
                    elements.put(index, value);
                } else {
                    double sum = elem + value;
                    if (sum == 0) {
                        elements.removeKey(index);
                    } else {
                        elements.put(index, sum);
                    }
                }
            }
        }
    }

    private double cumsum(int[] p, int[] c, int n) {
        int nz = 0;
        double nz2 = 0;
        for (int k = 0; k < n; k++) {
            p[k] = nz;
            nz += c[k];
            nz2 += c[k];
            c[k] = p[k];
        }
        p[n] = nz;
        return (nz2);
    }

}
