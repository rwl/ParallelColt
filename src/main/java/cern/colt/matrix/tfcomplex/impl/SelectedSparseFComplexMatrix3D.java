/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.impl;

import java.util.concurrent.ConcurrentHashMap;

import cern.colt.matrix.AbstractMatrix3D;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix3D;

/**
 * Selection view on sparse 3-d matrices holding <tt>complex</tt> elements. This
 * implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class SelectedSparseFComplexMatrix3D extends FComplexMatrix3D {
    private static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix.
     */
    protected ConcurrentHashMap<Long, float[]> elements;

    /**
     * The offsets of the visible cells of this matrix.
     */
    protected int[] sliceOffsets;

    protected int[] rowOffsets;

    protected int[] columnOffsets;

    /**
     * The offset.
     */
    protected int offset;

    /**
     * Constructs a matrix view with the given parameters.
     * 
     * @param elements
     *            the cells.
     * @param sliceOffsets
     *            The slice offsets of the cells that shall be visible.
     * @param rowOffsets
     *            The row offsets of the cells that shall be visible.
     * @param columnOffsets
     *            The column offsets of the cells that shall be visible.
     */
    protected SelectedSparseFComplexMatrix3D(ConcurrentHashMap<Long, float[]> elements, int[] sliceOffsets,
            int[] rowOffsets, int[] columnOffsets, int offset) {
        // be sure parameters are valid, we do not check...
        int slices = sliceOffsets.length;
        int rows = rowOffsets.length;
        int columns = columnOffsets.length;
        setUp(slices, rows, columns);

        this.elements = elements;

        this.sliceOffsets = sliceOffsets;
        this.rowOffsets = rowOffsets;
        this.columnOffsets = columnOffsets;

        this.offset = offset;

        this.isNoView = false;
    }

    protected int _columnOffset(int absRank) {
        return columnOffsets[absRank];
    }

    protected int _rowOffset(int absRank) {
        return rowOffsets[absRank];
    }

    protected int _sliceOffset(int absRank) {
        return sliceOffsets[absRank];
    }

    public float[] getQuick(int slice, int row, int column) {
        return elements.get((long) offset + (long) sliceOffsets[sliceZero + slice * sliceStride]
                + (long) rowOffsets[rowZero + row * rowStride]
                + (long) columnOffsets[columnZero + column * columnStride]);
    }

    public ConcurrentHashMap<Integer, float[]> elements() {
        throw new IllegalAccessError("This method is not supported.");
    }

    /**
     * Returns <tt>true</tt> if both matrices share common cells. More formally,
     * returns <tt>true</tt> if <tt>other != null</tt> and at least one of the
     * following conditions is met
     * <ul>
     * <li>the receiver is a view of the other matrix
     * <li>the other matrix is a view of the receiver
     * <li><tt>this == other</tt>
     * </ul>
     */

    protected boolean haveSharedCellsRaw(FComplexMatrix3D other) {
        if (other instanceof SelectedSparseFComplexMatrix3D) {
            SelectedSparseFComplexMatrix3D otherMatrix = (SelectedSparseFComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseFComplexMatrix3D) {
            SparseFComplexMatrix3D otherMatrix = (SparseFComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int slice, int row, int column) {
        return (long) this.offset + (long) sliceOffsets[sliceZero + slice * sliceStride]
                + (long) rowOffsets[rowZero + row * rowStride]
                + (long) columnOffsets[columnZero + column * columnStride];
    }

    public FComplexMatrix3D like(int slices, int rows, int columns) {
        return new SparseFComplexMatrix3D(slices, rows, columns);
    }

    public FComplexMatrix1D vectorize() {
        throw new IllegalArgumentException("This method is not supported.");
    }

    protected FComplexMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
            int columnStride) {
        throw new InternalError(); // this method is never called since
        // viewRow() and viewColumn are overridden
        // properly.
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        throw new InternalError(); // this method is never called
    }

    public void setQuick(int slice, int row, int column, float[] value) {
        long index = (long) offset + (long) sliceOffsets[sliceZero + slice * sliceStride]
                + (long) rowOffsets[rowZero + row * rowStride]
                + (long) columnOffsets[columnZero + column * columnStride];
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    public void setQuick(int slice, int row, int column, float re, float im) {
        long index = (long) offset + (long) sliceOffsets[sliceZero + slice * sliceStride]
                + (long) rowOffsets[rowZero + row * rowStride]
                + (long) columnOffsets[columnZero + column * columnStride];
        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new float[] { re, im });

    }

    protected void setUp(int slices, int rows, int columns) {
        super.setUp(slices, rows, columns);
        this.sliceStride = 1;
        this.rowStride = 1;
        this.columnStride = 1;
        this.offset = 0;
    }

    protected AbstractMatrix3D vDice(int axis0, int axis1, int axis2) {
        super.vDice(axis0, axis1, axis2);

        // swap offsets
        int[][] offsets = new int[3][];
        offsets[0] = this.sliceOffsets;
        offsets[1] = this.rowOffsets;
        offsets[2] = this.columnOffsets;

        this.sliceOffsets = offsets[axis0];
        this.rowOffsets = offsets[axis1];
        this.columnOffsets = offsets[axis2];

        return this;
    }

    public FComplexMatrix2D viewColumn(int column) {
        checkColumn(column);

        int viewRows = this.slices;
        int viewColumns = this.rows;

        int viewRowZero = sliceZero;
        int viewColumnZero = rowZero;
        int viewOffset = this.offset + _columnOffset(_columnRank(column));

        int viewRowStride = this.sliceStride;
        int viewColumnStride = this.rowStride;

        int[] viewRowOffsets = this.sliceOffsets;
        int[] viewColumnOffsets = this.rowOffsets;

        return new SelectedSparseFComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero,
                viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    public FComplexMatrix2D viewRow(int row) {
        checkRow(row);

        int viewRows = this.slices;
        int viewColumns = this.columns;

        int viewRowZero = sliceZero;
        int viewColumnZero = columnZero;
        int viewOffset = this.offset + _rowOffset(_rowRank(row));

        int viewRowStride = this.sliceStride;
        int viewColumnStride = this.columnStride;

        int[] viewRowOffsets = this.sliceOffsets;
        int[] viewColumnOffsets = this.columnOffsets;

        return new SelectedSparseFComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero,
                viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    protected FComplexMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseFComplexMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, this.offset);
    }

    public FComplexMatrix2D viewSlice(int slice) {
        checkSlice(slice);

        int viewRows = this.rows;
        int viewColumns = this.columns;

        int viewRowZero = rowZero;
        int viewColumnZero = columnZero;
        int viewOffset = this.offset + _sliceOffset(_sliceRank(slice));

        int viewRowStride = this.rowStride;
        int viewColumnStride = this.columnStride;

        int[] viewRowOffsets = this.rowOffsets;
        int[] viewColumnOffsets = this.columnOffsets;

        return new SelectedSparseFComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero,
                viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    public FloatMatrix3D getImaginaryPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

    public FloatMatrix3D getRealPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

}
