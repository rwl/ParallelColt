/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.AbstractMatrix3D;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;

/**
 * Selection view on dense 3-d matrices holding <tt>complex</tt> elements.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Objects of this class are typically constructed via <tt>viewIndexes</tt>
 * methods on some source matrix. The interface introduced in abstract super
 * classes defines everything a user can do. From a user point of view there is
 * nothing special about this class; it presents the same functionality with the
 * same signatures and semantics as its abstract superclass(es) while
 * introducing no additional functionality. Thus, this class need not be visible
 * to users.
 * <p>
 * This class uses no delegation. Its instances point directly to the data. Cell
 * addressing overhead is is 1 additional int addition and 3 additional array
 * index accesses per get/set.
 * <p>
 * Note that this implementation is not synchronized.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class SelectedDenseDComplexMatrix3D extends DComplexMatrix3D {

    private static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix.
     */
    protected double[] elements;

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
    protected SelectedDenseDComplexMatrix3D(double[] elements, int[] sliceOffsets, int[] rowOffsets,
            int[] columnOffsets, int offset) {
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

    public double[] getQuick(int slice, int row, int column) {
        int idxs = sliceZero + slice * sliceStride;
        int idxr = rowZero + row * rowStride;
        int idxc = columnZero + column * columnStride;
        return new double[] { elements[offset + sliceOffsets[idxs] + rowOffsets[idxr] + columnOffsets[idxc]],
                elements[offset + sliceOffsets[idxs] + rowOffsets[idxr] + columnOffsets[idxc] + 1] };
    }

    public double[] elements() {
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
     * *
     * 
     * @param other
     *            matrix
     * @return <tt>true</tt> if both matrices share common cells.
     */

    protected boolean haveSharedCellsRaw(DComplexMatrix3D other) {
        if (other instanceof SelectedDenseDComplexMatrix3D) {
            SelectedDenseDComplexMatrix3D otherMatrix = (SelectedDenseDComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseDComplexMatrix3D) {
            DenseDComplexMatrix3D otherMatrix = (DenseDComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int slice, int row, int column) {
        return this.offset + sliceOffsets[sliceZero + slice * sliceStride] + rowOffsets[rowZero + row * rowStride]
                + columnOffsets[columnZero + column * columnStride];
    }

    public DComplexMatrix3D like(int slices, int rows, int columns) {
        return new DenseDComplexMatrix3D(slices, rows, columns);
    }

    protected DComplexMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
            int columnStride) {
        throw new InternalError(); // this method is never called since
        // viewRow() and viewColumn are overridden
        // properly.
    }

    public DComplexMatrix2D like2D(int rows, int columns) {
        throw new InternalError(); // this method is never called
    }

    public void setQuick(int slice, int row, int column, double[] value) {
        int idxs = sliceZero + slice * sliceStride;
        int idxr = rowZero + row * rowStride;
        int idxc = columnZero + column * columnStride;
        elements[offset + sliceOffsets[idxs] + rowOffsets[idxr] + columnOffsets[idxc]] = value[0];
        elements[offset + sliceOffsets[idxs] + rowOffsets[idxr] + columnOffsets[idxc] + 1] = value[1];
    }

    public void setQuick(int slice, int row, int column, double re, double im) {
        int idxs = sliceZero + slice * sliceStride;
        int idxr = rowZero + row * rowStride;
        int idxc = columnZero + column * columnStride;
        elements[offset + sliceOffsets[idxs] + rowOffsets[idxr] + columnOffsets[idxc]] = re;
        elements[offset + sliceOffsets[idxs] + rowOffsets[idxr] + columnOffsets[idxc] + 1] = im;
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

    public DComplexMatrix2D viewColumn(int column) {
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

        return new SelectedDenseDComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero,
                viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    public DComplexMatrix2D viewRow(int row) {
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

        return new SelectedDenseDComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero,
                viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    protected DComplexMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseDComplexMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, this.offset);
    }

    public DComplexMatrix2D viewSlice(int slice) {
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

        return new SelectedDenseDComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero,
                viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    public DComplexMatrix1D vectorize() {
        throw new IllegalAccessError("This method is not supported.");
    }

    public DoubleMatrix3D getImaginaryPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

    public DoubleMatrix3D getRealPart() {
        throw new IllegalAccessError("This method is not supported.");
    }
}
