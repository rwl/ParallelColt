/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import cern.colt.map.tlong.AbstractLongIntMap;
import cern.colt.map.tlong.OpenLongIntHashMap;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import cern.colt.matrix.tint.IntMatrix3D;

/**
 * Sparse hashed 3-d matrix holding <tt>int</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Note that this implementation is not synchronized. Uses a
 * {@link cern.colt.map.tlong.OpenLongIntHashMap}, which is a compact and
 * performant hashing technique.
 * <p>
 * Cells that
 * <ul>
 * <li>are never set to non-zero values do not use any memory.
 * <li>switch from zero to non-zero state do use memory.
 * <li>switch back from non-zero to zero state also do use memory. However,
 * their memory is automatically reclaimed from time to time. It can also
 * manually be reclaimed by calling {@link #trimToSize()}.
 * </ul>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseIntMatrix3D extends IntMatrix3D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected AbstractLongIntMap elements;

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
    public SparseIntMatrix3D(int[][][] values) {
        this(values.length, (values.length == 0 ? 0 : values[0].length), (values.length == 0 ? 0
                : values[0].length == 0 ? 0 : values[0][0].length));
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of slices, rows and columns and
     * default memory usage. All entries are initially <tt>0</tt>.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>(int)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public SparseIntMatrix3D(int slices, int rows, int columns) {
        this(slices, rows, columns, slices * rows * (columns / 1000), 0.2, 0.5);
    }

    /**
     * Constructs a matrix with a given number of slices, rows and columns using
     * memory as specified. All entries are initially <tt>0</tt>. For details
     * related to memory usage see
     * {@link cern.colt.map.tlong.OpenLongIntHashMap}.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
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
     *             if <tt>(double)columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public SparseIntMatrix3D(int slices, int rows, int columns, int initialCapacity, double minLoadFactor,
            double maxLoadFactor) {
        try {
            setUp(slices, rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold slices*rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = new OpenLongIntHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
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
     * @throws IllegalArgumentException
     *             if <tt>(int)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    protected SparseIntMatrix3D(int slices, int rows, int columns, AbstractLongIntMap elements, int sliceZero,
            int rowZero, int columnZero, int sliceStride, int rowStride, int columnStride) {
        try {
            setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
        } catch (IllegalArgumentException exc) { // we can hold slices*rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.elements = elements;
        this.isNoView = false;
    }

    public IntMatrix3D assign(int value) {
        // overriden for performance only
        if (this.isNoView && value == 0)
            this.elements.clear();
        else
            super.assign(value);
        return this;
    }

    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
    }

    public AbstractLongIntMap elements() {
        return elements;
    }

    public void ensureCapacity(int minCapacity) {
        this.elements.ensureCapacity(minCapacity);
    }

    public synchronized int getQuick(int slice, int row, int column) {
        // if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows ||
        // column<0 || column>=columns) throw new
        // IndexOutOfBoundsException("slice:"+slice+", row:"+row+",
        // column:"+column);
        // return elements.get(index(slice,row,column));
        // manually inlined:
        return elements.get((long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride);
    }

    public long index(int slice, int row, int column) {
        // return _sliceOffset(_sliceRank(slice)) + _rowOffset(_rowRank(row)) +
        // _columnOffset(_columnRank(column));
        // manually inlined:
        return (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row * (long) rowStride
                + (long) columnZero + (long) column * (long) columnStride;
    }

    public IntMatrix3D like(int slices, int rows, int columns) {
        return new SparseIntMatrix3D(slices, rows, columns);
    }

    public IntMatrix2D like2D(int rows, int columns) {
        return new SparseIntMatrix2D(rows, columns);
    }

    public synchronized void setQuick(int slice, int row, int column, int value) {
        // if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows ||
        // column<0 || column>=columns) throw new
        // IndexOutOfBoundsException("slice:"+slice+", row:"+row+",
        // column:"+column);
        // int index = index(slice,row,column);
        // manually inlined:
        long index = (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
        if (value == 0)
            this.elements.removeKey(index);
        else
            this.elements.put(index, value);
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(slices).append(" x ").append(rows).append(" x ").append(columns)
                .append(" sparse matrix, nnz = ").append(cardinality()).append('\n');
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    int elem = getQuick(s, r, c);
                    if (elem != 0) {
                        builder.append('(').append(s).append(',').append(r).append(',').append(c).append(')').append(
                                '\t').append(elem).append('\n');
                    }
                }
            }
        }
        return builder.toString();
    }

    public void trimToSize() {
        this.elements.trimToSize();
    }

    public IntMatrix1D vectorize() {
        IntMatrix1D v = new SparseIntMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    protected boolean haveSharedCellsRaw(IntMatrix3D other) {
        if (other instanceof SelectedSparseIntMatrix3D) {
            SelectedSparseIntMatrix3D otherMatrix = (SelectedSparseIntMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseIntMatrix3D) {
            SparseIntMatrix3D otherMatrix = (SparseIntMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    protected IntMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
        return new SparseIntMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride);
    }

    protected IntMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseIntMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
    }
}
