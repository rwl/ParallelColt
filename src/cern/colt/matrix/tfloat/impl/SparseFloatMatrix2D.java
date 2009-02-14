/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import java.io.IOException;

import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.map.tfloat.AbstractLongFloatMap;
import cern.colt.map.tfloat.OpenLongFloatHashMap;
import cern.colt.matrix.io.MatrixInfo;
import cern.colt.matrix.io.MatrixSize;
import cern.colt.matrix.io.MatrixVectorReader;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;

/**
 * Sparse hashed 2-d matrix holding <tt>float</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Note that this implementation is not synchronized. Uses a
 * {@link cern.colt.map.tfloat.OpenIntFloatHashMap}, which is a compact and
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
 * {@link DenseFloatMatrix2D}. However, constant factors are considerably
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
 * @see cern.colt.map.tfloat.OpenIntFloatHashMap
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.1, 08/22/2007
 */
public class SparseFloatMatrix2D extends FloatMatrix2D {
    /*
     * The elements of the matrix.
     */
    protected AbstractLongFloatMap elements;

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
    public SparseFloatMatrix2D(float[][] values) {
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseFloatMatrix2D(int rows, int columns) {
        this(rows, columns, rows * (columns / 1000), 0.2f, 0.5f);
    }

    /**
     * Constructs a matrix with a given number of rows and columns using memory
     * as specified. All entries are initially <tt>0</tt>. For details related
     * to memory usage see {@link cern.colt.map.tfloat.OpenIntFloatHashMap}.
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseFloatMatrix2D(int rows, int columns, int initialCapacity, float minLoadFactor, float maxLoadFactor) {
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = new OpenLongFloatHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
    }

    /**
     * Constructs a matrix with a copy of the given indexes and values.
     * 
     * @param rows
     * @param columns
     * @param rowindexes
     * @param columnindexes
     * @param values
     */
    public SparseFloatMatrix2D(int rows, int columns, int[] rowindexes, int[] columnindexes, float[] values) {
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = new OpenLongFloatHashMap(rowindexes.length);
        insert(rowindexes, columnindexes, values);
    }

    /**
     * Constructs a matrix from MatrixVectorReader.
     * 
     * @param r
     *            matrix reader
     * @throws IOException
     */
    public SparseFloatMatrix2D(MatrixVectorReader r) throws IOException {
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
        int[] columnindexes = new int[numEntries];
        int[] rowindexes = new int[numEntries];
        float[] values = new float[numEntries];
        r.readCoordinate(rowindexes, columnindexes, values);
        if (info.isSymmetric() || info.isSkewSymmetric()) {
            this.elements = new OpenLongFloatHashMap(2 * rowindexes.length);
        } else {
            this.elements = new OpenLongFloatHashMap(rowindexes.length);
        }
        insert(rowindexes, columnindexes, values);

        if (info.isSymmetric()) {
            for (int i = 0; i < numEntries; i++) {
                if (rowindexes[i] != columnindexes[i]) {
                    set(columnindexes[i], rowindexes[i], values[i]);
                }
            }
        } else if (info.isSkewSymmetric()) {
            for (int i = 0; i < numEntries; i++) {
                if (rowindexes[i] != columnindexes[i]) {
                    set(columnindexes[i], rowindexes[i], -values[i]);
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    protected SparseFloatMatrix2D(int rows, int columns, AbstractLongFloatMap elements, int rowZero, int columnZero, int rowStride, int columnStride) {
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
     *   matrix = 2 x 2 matrix
     *   0.5 1.5      
     *   2.5 3.5
     * 
     *   // change each cell to its sine
     *   matrix.assign(cern.jet.math.Functions.sin);
     *   --&gt;
     *   2 x 2 matrix
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
     * @see cern.jet.math.tfloat.FloatFunctions
     */
    public FloatMatrix2D assign(cern.colt.function.tfloat.FloatFunction function) {
        if (this.isNoView && function instanceof cern.jet.math.tfloat.FloatMult) { // x[i]
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
    public FloatMatrix2D assign(float value) {
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
    public FloatMatrix2D assign(FloatMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof SparseFloatMatrix2D)) {
            return super.assign(source);
        }
        SparseFloatMatrix2D other = (SparseFloatMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);

        if (this.isNoView && other.isNoView) { // quickest
            this.elements.assign(other.elements);
            return this;
        }
        return super.assign(source);
    }

    public FloatMatrix2D assign(final FloatMatrix2D y, cern.colt.function.tfloat.FloatFloatFunction function) {
        if (!this.isNoView)
            return super.assign(y, function);

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
        }

        if (function == cern.jet.math.tfloat.FloatFunctions.mult) { // x[i] = x[i] * y[i]
            this.elements.forEachPair(new cern.colt.function.tfloat.LongFloatProcedure() {
                public boolean apply(long key, float value) {
                    int i = (int) (key / columns);
                    int j = (int) (key % columns);
                    float r = value * y.getQuick(i, j);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        }

        if (function == cern.jet.math.tfloat.FloatFunctions.div) { // x[i] = x[i] /
            // y[i]
            this.elements.forEachPair(new cern.colt.function.tfloat.LongFloatProcedure() {
                public boolean apply(long key, float value) {
                    int i = (int) (key / columns);
                    int j = (int) (key % columns);
                    float r = value / y.getQuick(i, j);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        }
        return this;

    }

    public SparseFloatMatrix2D assign(final int[] rowindexes, final int[] columnindexes, final float[] values, final cern.colt.function.tfloat.FloatFloatFunction function) {
        int size = rowindexes.length;
        if (function == cern.jet.math.tfloat.FloatFunctions.plus) { // x[i] = x[i] + y[i]
            for (int i = 0; i < size; i++) {
                float value = values[i];
                long row = (long) rowindexes[i];
                long column = (long) columnindexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                float elem = elements.get(index);
                value += elem;
                if (value != 0) {
                    elements.put(index, value);
                } else {
                    elements.removeKey(index);
                }
            }
        } else {
            for (int i = 0; i < size; i++) {
                float value = values[i];
                long row = (long) rowindexes[i];
                long column = (long) columnindexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                float elem = elements.get(index);
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

    /**
     * Returns the number of cells having non-zero values.
     */
    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
    }

    public RCFloatMatrix2D convertToRCFloatMatrix2D() {
        int nnz = cardinality();
        LongArrayList keysList = new LongArrayList();
        FloatArrayList valuesList = new FloatArrayList();
        elements.pairsSortedByKey(keysList, valuesList);
        long[] keys = keysList.elements();
        float[] values = valuesList.elements();
        int[] rowindexes = new int[nnz];
        int[] columnindexes = new int[nnz];

        int[] idxs = new int[nnz];
        float[] vals = new float[nnz];
        int[] starts = new int[rows + 1];
        int[] w = new int[rows];
        int r;
        for (int k = 0; k < nnz; k++) {
            rowindexes[k] = (int) (keys[k] / columns);
            columnindexes[k] = (int) (keys[k] % columns);
            w[rowindexes[k]]++;
        }
        cumsum(starts, w, rows);
        for (int k = 0; k < nnz; k++) {
            idxs[r = w[rowindexes[k]]++] = columnindexes[k];
            vals[r] = values[k];
        }
        RCFloatMatrix2D M = new RCFloatMatrix2D(rows, columns, starts, new IntArrayList(idxs), new FloatArrayList(vals));
        return M;
    }

    public CCFloatMatrix2D convertToCCFloatMatrix2D() {
        int nnz = cardinality();
        LongArrayList keysList = new LongArrayList();
        FloatArrayList valuesList = new FloatArrayList();
        elements.pairsSortedByKey(keysList, valuesList);
        long[] keys = keysList.elements();
        float[] values = valuesList.elements();
        int[] rowindexes = new int[nnz];
        int[] columnindexes = new int[nnz];

        int[] idxs = new int[nnz];
        float[] vals = new float[nnz];
        int[] starts = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nnz; k++) {
            rowindexes[k] = (int) (keys[k] / columns);
            columnindexes[k] = (int) (keys[k] % columns);
            w[columnindexes[k]]++;
        }
        cumsum(starts, w, columns);
        for (int k = 0; k < nnz; k++) {
            idxs[r = w[columnindexes[k]]++] = rowindexes[k];
            vals[r] = values[k];
        }
        CCFloatMatrix2D M = new CCFloatMatrix2D(rows, columns, starts, new IntArrayList(idxs), new FloatArrayList(vals));
        return M;
    }

    public RCMFloatMatrix2D convertToRCMFloatMatrix2D() {
        RCMFloatMatrix2D A = new RCMFloatMatrix2D(rows, columns);
        int nnz = cardinality();
        long[] keys = elements.keys().elements();
        float[] values = elements.values().elements();
        for (int i = 0; i < nnz; i++) {
            int row = (int) (keys[i] / columns);
            int column = (int) (keys[i] % columns);
            A.setQuick(row, column, values[i]);
        }
        return A;
    }

    public CCMFloatMatrix2D convertToCCMFloatMatrix2D() {
        CCMFloatMatrix2D A = new CCMFloatMatrix2D(rows, columns);
        int nnz = cardinality();
        long[] keys = elements.keys().elements();
        float[] values = elements.values().elements();
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
    public AbstractLongFloatMap elements() {
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

    public FloatMatrix2D forEachNonZero(final cern.colt.function.tfloat.IntIntFloatFunction function) {
        if (this.isNoView) {
            this.elements.forEachPair(new cern.colt.function.tfloat.LongFloatProcedure() {
                public boolean apply(long key, float value) {
                    int i = (int) (key / columns);
                    int j = (int) (key % columns);
                    float r = function.apply(i, j, value);
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
    public float getQuick(int row, int column) {
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
     * <tt>DenseFloatMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseFloatMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseFloatMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseFloatMatrix2D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public FloatMatrix2D like(int rows, int columns) {
        return new SparseFloatMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseFloatMatrix2D</tt> the new
     * matrix must be of type <tt>DenseFloatMatrix1D</tt>, if the receiver is
     * an instance of type <tt>SparseFloatMatrix2D</tt> the new matrix must be
     * of type <tt>SparseFloatMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public FloatMatrix1D like1D(int size) {
        return new SparseFloatMatrix1D(size);
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
    public FloatMatrix1D vectorize() {
        SparseFloatMatrix1D v = new SparseFloatMatrix1D(size());
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                float elem = getQuick(r, c);
                if (elem != 0) {
                    v.setQuick(idx++, elem);
                }
            }
        }
        return v;
    }

    public FloatMatrix1D zMult(FloatMatrix1D y, FloatMatrix1D z, float alpha, float beta, final boolean transposeA) {
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }

        boolean ignore = (z == null);
        if (z == null)
            z = new DenseFloatMatrix1D(m);

        if (!(this.isNoView && y instanceof DenseFloatMatrix1D && z instanceof DenseFloatMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (n != y.size() || m > z.size())
            throw new IllegalArgumentException("Incompatible args: " + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", " + z.toStringShort());

        if (!ignore)
            z.assign(cern.jet.math.tfloat.FloatFunctions.mult(beta / alpha));

        DenseFloatMatrix1D zz = (DenseFloatMatrix1D) z;
        final float[] zElements = zz.elements;
        final int zStride = zz.stride();
        final int zi = (int) z.index(0);

        DenseFloatMatrix1D yy = (DenseFloatMatrix1D) y;
        final float[] yElements = yy.elements;
        final int yStride = yy.stride();
        final int yi = (int) y.index(0);

        if (yElements == null || zElements == null)
            throw new InternalError();

        this.elements.forEachPair(new cern.colt.function.tfloat.LongFloatProcedure() {
            public boolean apply(long key, float value) {
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
         * forEachNonZero( new cern.colt.function.IntIntFloatFunction() {
         * public float apply(int i, int j, float value) { if (transposeA) {
         * int tmp=i; i=j; j=tmp; } zElements[zi + zStride*i] += value *
         * yElements[yi + yStride*j]; //z.setQuick(row,z.getQuick(row) + value *
         * y.getQuick(column)); //System.out.println("["+i+","+j+"]-->"+value);
         * return value; } } );
         */

        if (alpha != 1)
            z.assign(cern.jet.math.tfloat.FloatFunctions.mult(alpha));
        return z;
    }

    public FloatMatrix2D zMult(FloatMatrix2D B, FloatMatrix2D C, final float alpha, float beta, final boolean transposeA, boolean transposeB) {
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
            C = new DenseFloatMatrix2D(m, p);

        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
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

        this.elements.forEachPair(new cern.colt.function.tfloat.LongFloatProcedure() {
            public boolean apply(long key, float value) {
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
    protected boolean haveSharedCellsRaw(FloatMatrix2D other) {
        if (other instanceof SelectedSparseFloatMatrix2D) {
            SelectedSparseFloatMatrix2D otherMatrix = (SelectedSparseFloatMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseFloatMatrix2D) {
            SparseFloatMatrix2D otherMatrix = (SparseFloatMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseFloatMatrix2D</tt> the new matrix must be of
     * type <tt>DenseFloatMatrix1D</tt>, if the receiver is an instance of type
     * <tt>SparseFloatMatrix2D</tt> the new matrix must be of type
     * <tt>SparseFloatMatrix1D</tt>, etc.
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
    protected FloatMatrix1D like1D(int size, int offset, int stride) {
        return new SparseFloatMatrix1D(size, this.elements, (int) offset, stride);
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
    protected FloatMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseFloatMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }

    private void insert(int[] rowindexes, int[] columnindexes, float[] values) {
        int size = rowindexes.length;
        for (int i = 0; i < size; i++) {
            float value = values[i];
            long row = (long) rowindexes[i];
            long column = (long) columnindexes[i];
            if (row >= rows || column >= columns) {
                throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
            }
            if (value != 0) {
                long index = (long) rowZero + row * (long) rowStride + (long) columnZero + column * (long) columnStride;
                float elem = elements.get(index);
                if (elem == 0) {
                    elements.put(index, value);
                } else {
                    float sum = elem + value;
                    if (sum == 0) {
                        elements.removeKey(index);
                    } else {
                        elements.put(index, sum);
                    }
                }
            }
        }
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

}
