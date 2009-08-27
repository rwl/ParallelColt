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
 * {@link cern.colt.map.tdouble.OpenLongDoubleHashMap}, which is a compact and
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
 * @see cern.colt.map.tdouble.OpenLongDoubleHashMap
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseDoubleMatrix2D extends DoubleMatrix2D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected AbstractLongDoubleMap elements;

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
     * to memory usage see {@link cern.colt.map.tdouble.OpenLongDoubleHashMap}.
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
     * Constructs a matrix with a copy of the given indexes and a single value.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param rowIndexes
     *            row indexes
     * @param columnIndexes
     *            column indexes
     * @param value
     *            numerical value
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
     * Constructs a matrix with a copy of the given indexes and values.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param rowIndexes
     *            row indexes
     * @param columnIndexes
     *            column indexes
     * @param values
     *            numerical values
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
     * Constructs a matrix from MatrixVectorReader.
     * 
     * @param reader
     *            matrix reader
     * @throws IOException
     */
    public SparseDoubleMatrix2D(MatrixVectorReader reader) throws IOException {
        MatrixInfo info;
        if (reader.hasInfo())
            info = reader.readMatrixInfo();
        else
            info = new MatrixInfo(true, MatrixInfo.MatrixField.Real, MatrixInfo.MatrixSymmetry.General);

        if (info.isPattern())
            throw new UnsupportedOperationException("Pattern matrices are not supported");
        if (info.isDense())
            throw new UnsupportedOperationException("Dense matrices are not supported");
        if (info.isComplex())
            throw new UnsupportedOperationException("Complex matrices are not supported");

        MatrixSize size = reader.readMatrixSize(info);
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
        reader.readCoordinate(rowIndexes, columnIndexes, values);
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
    protected SparseDoubleMatrix2D(int rows, int columns, AbstractLongDoubleMap elements, int rowZero, int columnZero,
            int rowStride, int columnStride) {
        try {
            setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = elements;
        this.isNoView = false;
    }

    public DoubleMatrix2D assign(cern.colt.function.tdouble.DoubleFunction function) {
        if (this.isNoView && function instanceof cern.jet.math.tdouble.DoubleMult) { // x[i] = mult*x[i]
            this.elements.assign(function);
        } else {
            super.assign(function);
        }
        return this;
    }

    public DoubleMatrix2D assign(double value) {
        // overriden for performance only
        if (this.isNoView && value == 0)
            this.elements.clear();
        else
            super.assign(value);
        return this;
    }

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
        } else if (function == cern.jet.math.tdouble.DoubleFunctions.mult) { // x[i] = x[i] * y[i]
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
        } else if (function == cern.jet.math.tdouble.DoubleFunctions.div) { // x[i] = x[i] /  y[i]
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
        } else {
            super.assign(y, function);
        }
        return this;

    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[row,col] = function(x[row,col],y[row,col])</tt>, where y is given
     * in the coordinate form with single numerical value.
     * 
     * @param rowIndexes
     *            row indexes of y
     * @param columnIndexes
     *            column indexes of y
     * @param value
     *            numerical value of y
     * @param function
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     */
    public SparseDoubleMatrix2D assign(final int[] rowIndexes, final int[] columnIndexes, final double value,
            final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        int size = rowIndexes.length;
        if (function == cern.jet.math.tdouble.DoubleFunctions.plus) { // x[i] = x[i] + y[i]
            for (int i = 0; i < size; i++) {
                long row = rowIndexes[i];
                long column = columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = rowZero + row * rowStride + columnZero + column * columnStride;
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
                long row = rowIndexes[i];
                long column = columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = rowZero + row * rowStride + columnZero + column * columnStride;
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
     * Assigns the result of a function to each cell;
     * <tt>x[row,col] = function(x[row,col],y[row,col])</tt>, where y is given
     * in the coordinate form.
     * 
     * @param rowIndexes
     *            row indexes of y
     * @param columnIndexes
     *            column indexes of y
     * @param values
     *            numerical values of y
     * @param function
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     */
    public SparseDoubleMatrix2D assign(final int[] rowIndexes, final int[] columnIndexes, final double[] values,
            final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        int size = rowIndexes.length;
        if (function == cern.jet.math.tdouble.DoubleFunctions.plus) { // x[i] = x[i] + y[i]
            for (int i = 0; i < size; i++) {
                double value = values[i];
                long row = rowIndexes[i];
                long column = columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = rowZero + row * rowStride + columnZero + column * columnStride;
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
                long row = rowIndexes[i];
                long column = columnIndexes[i];
                if (row >= rows || column >= columns) {
                    throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
                }
                long index = rowZero + row * rowStride + columnZero + column * columnStride;
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

    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a column-compressed form. This method creates a new object (not a view),
     * so changes in the returned matrix are NOT reflected in this matrix.
     * 
     * @param sortRowIndexes
     *            if true, then row indexes in column compressed matrix are
     *            sorted
     * 
     * @return this matrix in a column-compressed form
     */
    public SparseCCDoubleMatrix2D getColumnCompressed(boolean sortRowIndexes) {
        int nnz = cardinality();
        long[] keys = elements.keys().elements();
        double[] values = elements.values().elements();
        int[] rowIndexes = new int[nnz];
        int[] columnIndexes = new int[nnz];

        for (int k = 0; k < nnz; k++) {
            long key = keys[k];
            rowIndexes[k] = (int) (key / columns);
            columnIndexes[k] = (int) (key % columns);
        }
        return new SparseCCDoubleMatrix2D(rows, columns, rowIndexes, columnIndexes, values, false, false,
                sortRowIndexes);
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a column-compressed modified form. This method creates a new object (not
     * a view), so changes in the returned matrix are NOT reflected in this
     * matrix.
     * 
     * @return this matrix in a column-compressed modified form
     */
    public SparseCCMDoubleMatrix2D getColumnCompressedModified() {
        SparseCCMDoubleMatrix2D A = new SparseCCMDoubleMatrix2D(rows, columns);
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
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a row-compressed form. This method creates a new object (not a view), so
     * changes in the returned matrix are NOT reflected in this matrix.
     * 
     * @param sortColumnIndexes
     *            if true, then column indexes in row compressed matrix are
     *            sorted
     * 
     * @return this matrix in a row-compressed form
     */
    public SparseRCDoubleMatrix2D getRowCompressed(boolean sortColumnIndexes) {
        int nnz = cardinality();
        long[] keys = elements.keys().elements();
        double[] values = elements.values().elements();
        final int[] rowIndexes = new int[nnz];
        final int[] columnIndexes = new int[nnz];
        for (int k = 0; k < nnz; k++) {
            long key = keys[k];
            rowIndexes[k] = (int) (key / columns);
            columnIndexes[k] = (int) (key % columns);
        }
        return new SparseRCDoubleMatrix2D(rows, columns, rowIndexes, columnIndexes, values, false, false,
                sortColumnIndexes);
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a row-compressed modified form. This method creates a new object (not a
     * view), so changes in the returned matrix are NOT reflected in this
     * matrix.
     * 
     * @return this matrix in a row-compressed modified form
     */
    public SparseRCMDoubleMatrix2D getRowCompressedModified() {
        SparseRCMDoubleMatrix2D A = new SparseRCMDoubleMatrix2D(rows, columns);
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

    public AbstractLongDoubleMap elements() {
        return elements;
    }

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

    public synchronized double getQuick(int row, int column) {
        return this.elements.get((long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column
                * (long) columnStride);
    }

    public long index(int row, int column) {
        return (long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
    }

    public DoubleMatrix2D like(int rows, int columns) {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    public DoubleMatrix1D like1D(int size) {
        return new SparseDoubleMatrix1D(size);
    }

    public synchronized void setQuick(int row, int column, double value) {
        long index = (long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column
                * (long) columnStride;
        if (value == 0)
            this.elements.removeKey(index);
        else
            this.elements.put(index, value);
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(rows).append(" x ").append(columns).append(" sparse matrix, nnz = ").append(cardinality())
                .append('\n');
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double elem = getQuick(r, c);
                if (elem != 0) {
                    builder.append('(').append(r).append(',').append(c).append(')').append('\t').append(elem).append(
                            '\n');
                }
            }
        }
        return builder.toString();
    }

    public void trimToSize() {
        this.elements.trimToSize();
    }

    public DoubleMatrix1D vectorize() {
        SparseDoubleMatrix1D v = new SparseDoubleMatrix1D((int) size());
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                double elem = getQuick(r, c);
                v.setQuick(idx++, elem);
            }
        }
        return v;
    }

    public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, final double alpha, double beta,
            final boolean transposeA) {
        int rowsA = rows;
        int columnsA = columns;
        if (transposeA) {
            rowsA = columns;
            columnsA = rows;
        }

        boolean ignore = (z == null);
        if (z == null)
            z = new DenseDoubleMatrix1D(rowsA);

        if (!(this.isNoView && y instanceof DenseDoubleMatrix1D && z instanceof DenseDoubleMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        if (!ignore)
            z.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta));

        DenseDoubleMatrix1D zz = (DenseDoubleMatrix1D) z;
        final double[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) z.index(0);

        DenseDoubleMatrix1D yy = (DenseDoubleMatrix1D) y;
        final double[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) y.index(0);

        if (elementsY == null || elementsZ == null)
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
                elementsZ[zeroZ + strideZ * i] += alpha * value * elementsY[zeroY + strideY * j];
                return true;
            }
        });

        return z;
    }

    public DoubleMatrix2D zMult(DoubleMatrix2D B, DoubleMatrix2D C, final double alpha, double beta,
            final boolean transposeA, boolean transposeB) {
        if (!(this.isNoView)) {
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        }
        if (transposeB)
            B = B.viewDice();
        int rowsA = rows;
        int columnsA = columns;
        if (transposeA) {
            rowsA = columns;
            columnsA = rows;
        }
        int p = B.columns();
        boolean ignore = (C == null);
        if (C == null)
            C = new DenseDoubleMatrix2D(rowsA, p);

        if (B.rows() != columnsA)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows() != rowsA || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", "
                    + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!ignore)
            C.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta));

        // cache views
        final DoubleMatrix1D[] Brows = new DoubleMatrix1D[columnsA];
        for (int i = columnsA; --i >= 0;)
            Brows[i] = B.viewRow(i);
        final DoubleMatrix1D[] Crows = new DoubleMatrix1D[rowsA];
        for (int i = rowsA; --i >= 0;)
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

    private void insert(int[] rowIndexes, int[] columnIndexes, double value) {
        int size = rowIndexes.length;
        for (int i = 0; i < size; i++) {
            long row = rowIndexes[i];
            long column = columnIndexes[i];
            if (row >= rows || column >= columns) {
                throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
            }
            if (value != 0) {
                long index = rowZero + row * rowStride + columnZero + column * columnStride;
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
            long row = rowIndexes[i];
            long column = columnIndexes[i];
            if (row >= rows || column >= columns) {
                throw new IndexOutOfBoundsException("row: " + row + ", column: " + column);
            }
            if (value != 0) {
                long index = rowZero + row * rowStride + columnZero + column * columnStride;
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

    protected DoubleMatrix1D like1D(int size, int offset, int stride) {
        return new SparseDoubleMatrix1D(size, this.elements, offset, stride);
    }

    protected DoubleMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseDoubleMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }

}
