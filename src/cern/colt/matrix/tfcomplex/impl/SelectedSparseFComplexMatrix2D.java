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

import cern.colt.matrix.AbstractMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;

/**
 * Selection view on sparse 2-d matrices holding <tt>complex</tt> elements. This
 * implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class SelectedSparseFComplexMatrix2D extends FComplexMatrix2D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Integer, float[]> elements;

    /**
     * The offsets of the visible cells of this matrix.
     */
    protected int[] rowOffsets;

    protected int[] columnOffsets;

    /**
     * The offset.
     */
    protected int offset;

    /**
     * Constructs a matrix view with the given parameters.
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
     * @param rowOffsets
     *            The row offsets of the cells that shall be visible.
     * @param columnOffsets
     *            The column offsets of the cells that shall be visible.
     * @param offset
     */
    protected SelectedSparseFComplexMatrix2D(int rows, int columns, ConcurrentHashMap<Integer, float[]> elements,
            int rowZero, int columnZero, int rowStride, int columnStride, int[] rowOffsets, int[] columnOffsets,
            int offset) {
        // be sure parameters are valid, we do not check...
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);

        this.elements = elements;
        this.rowOffsets = rowOffsets;
        this.columnOffsets = columnOffsets;
        this.offset = offset;

        this.isNoView = false;
    }

    /**
     * Constructs a matrix view with the given parameters.
     * 
     * @param elements
     *            the cells.
     * @param rowOffsets
     *            The row offsets of the cells that shall be visible.
     * @param columnOffsets
     *            The column offsets of the cells that shall be visible.
     * @param offset
     */
    protected SelectedSparseFComplexMatrix2D(ConcurrentHashMap<Integer, float[]> elements, int[] rowOffsets,
            int[] columnOffsets, int offset) {
        this(rowOffsets.length, columnOffsets.length, elements, 0, 0, 1, 1, rowOffsets, columnOffsets, offset);
    }

    @Override
    protected int _columnOffset(int absRank) {
        return columnOffsets[absRank];
    }

    @Override
    protected int _rowOffset(int absRank) {
        return rowOffsets[absRank];
    }

    @Override
    public float[] getQuick(int row, int column) {
        return elements.get(offset + rowOffsets[rowZero + row * rowStride]
                + columnOffsets[columnZero + column * columnStride]);
    }

    @Override
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
    @Override
    protected boolean haveSharedCellsRaw(FComplexMatrix2D other) {
        if (other instanceof SelectedSparseFComplexMatrix2D) {
            SelectedSparseFComplexMatrix2D otherMatrix = (SelectedSparseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseFComplexMatrix2D) {
            SparseFComplexMatrix2D otherMatrix = (SparseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    @Override
    public long index(int row, int column) {
        return this.offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];
    }

    @Override
    public FComplexMatrix2D like(int rows, int columns) {
        return new SparseFComplexMatrix2D(rows, columns);
    }

    @Override
    public FComplexMatrix1D like1D(int size) {
        return new SparseFComplexMatrix1D(size);
    }

    @Override
    protected FComplexMatrix1D like1D(int size, int zero, int stride) {
        throw new InternalError(); // this method is never called since
        // viewRow() and viewColumn are overridden
        // properly.
    }

    @Override
    public void setQuick(int row, int column, float[] value) {
        int index = offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];

        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    @Override
    public FComplexMatrix1D vectorize() {
        throw new IllegalAccessError("This method is not supported.");
    }

    @Override
    public void setQuick(int row, int column, float re, float im) {
        int index = offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];

        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new float[] { re, im });

    }

    @Override
    protected void setUp(int rows, int columns) {
        super.setUp(rows, columns);
        this.rowStride = 1;
        this.columnStride = 1;
        this.offset = 0;
    }

    @Override
    protected AbstractMatrix2D vDice() {
        super.vDice();
        // swap
        int[] tmp = rowOffsets;
        rowOffsets = columnOffsets;
        columnOffsets = tmp;

        // flips stay unaffected

        this.isNoView = false;
        return this;
    }

    @Override
    public FComplexMatrix1D viewColumn(int column) {
        checkColumn(column);
        int viewSize = this.rows;
        int viewZero = this.rowZero;
        int viewStride = this.rowStride;
        int[] viewOffsets = this.rowOffsets;
        int viewOffset = this.offset + _columnOffset(_columnRank(column));
        return new SelectedSparseFComplexMatrix1D(viewSize, this.elements, viewZero, viewStride, viewOffsets,
                viewOffset);
    }

    @Override
    public FComplexMatrix1D viewRow(int row) {
        checkRow(row);
        int viewSize = this.columns;
        int viewZero = columnZero;
        int viewStride = this.columnStride;
        int[] viewOffsets = this.columnOffsets;
        int viewOffset = this.offset + _rowOffset(_rowRank(row));
        return new SelectedSparseFComplexMatrix1D(viewSize, this.elements, viewZero, viewStride, viewOffsets,
                viewOffset);
    }

    @Override
    protected FComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseFComplexMatrix2D(this.elements, rowOffsets, columnOffsets, this.offset);
    }

    @Override
    public FloatMatrix2D getImaginaryPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

    @Override
    public FloatMatrix2D getRealPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

}
