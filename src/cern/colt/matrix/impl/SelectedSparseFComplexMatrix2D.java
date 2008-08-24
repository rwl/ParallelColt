/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.impl;

import java.util.concurrent.ConcurrentHashMap;

import cern.colt.matrix.FComplexMatrix1D;
import cern.colt.matrix.FComplexMatrix2D;
import cern.colt.matrix.FloatMatrix2D;

/**
 * Selection view on sparse 2-d matrices holding <tt>complex</tt> elements.
 * Note that this implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.0, 12/10/2007
 */
class SelectedSparseFComplexMatrix2D extends FComplexMatrix2D {
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
    protected SelectedSparseFComplexMatrix2D(int rows, int columns, ConcurrentHashMap<Integer, float[]> elements, int rowZero, int columnZero, int rowStride, int columnStride, int[] rowOffsets, int[] columnOffsets, int offset) {
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
    protected SelectedSparseFComplexMatrix2D(ConcurrentHashMap<Integer, float[]> elements, int[] rowOffsets, int[] columnOffsets, int offset) {
        this(rowOffsets.length, columnOffsets.length, elements, 0, 0, 1, 1, rowOffsets, columnOffsets, offset);
    }

    /**
     * Returns the position of the given absolute rank within the (virtual or
     * non-virtual) internal 1-dimensional array. Default implementation.
     * Override, if necessary.
     * 
     * @param rank
     *            the absolute rank of the element.
     * @return the position.
     */
    protected int _columnOffset(int absRank) {
        return columnOffsets[absRank];
    }

    /**
     * Returns the position of the given absolute rank within the (virtual or
     * non-virtual) internal 1-dimensional array. Default implementation.
     * Override, if necessary.
     * 
     * @param rank
     *            the absolute rank of the element.
     * @return the position.
     */
    protected int _rowOffset(int absRank) {
        return rowOffsets[absRank];
    }

    public void fft2() {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void ifft2(boolean scale) {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void fftRows() {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void ifftRows(boolean scale) {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void fftColumns() {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void ifftColumns(boolean scale) {
        throw new IllegalArgumentException("This method is not supported yet");
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
    public float[] getQuick(int row, int column) {
        return elements.get(offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride]);
    }

    public ConcurrentHashMap<Integer, float[]> elements() {
        return elements;
    }

    /**
     * Returns <tt>true</tt> if both matrices share common cells. More
     * formally, returns <tt>true</tt> if <tt>other != null</tt> and at
     * least one of the following conditions is met
     * <ul>
     * <li>the receiver is a view of the other matrix
     * <li>the other matrix is a view of the receiver
     * <li><tt>this == other</tt>
     * </ul>
     */
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

    /**
     * Returns the position of the given coordinate within the (virtual or
     * non-virtual) internal 1-dimensional array.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     */
    public int index(int row, int column) {
        return this.offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseComplexMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseComplexMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseComplexMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseComplexMatrix2D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public FComplexMatrix2D like(int rows, int columns) {
        return new SparseFComplexMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseComplexMatrix2D</tt> the new
     * matrix must be of type <tt>DenseComplexMatrix1D</tt>, if the receiver
     * is an instance of type <tt>SparseComplexMatrix2D</tt> the new matrix
     * must be of type <tt>SparseComplexMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public FComplexMatrix1D like1D(int size) {
        return new SparseFComplexMatrix1D(size);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseComplexMatrix2D</tt> the new matrix must be
     * of type <tt>DenseComplexMatrix1D</tt>, if the receiver is an instance
     * of type <tt>SparseComplexMatrix2D</tt> the new matrix must be of type
     * <tt>SparseComplexMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param zero
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @return a new matrix of the corresponding dynamic type.
     */
    protected FComplexMatrix1D like1D(int size, int zero, int stride) {
        throw new InternalError(); // this method is never called since
        // viewRow() and viewColumn are overridden
        // properly.
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the
     * specified value.
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
    public void setQuick(int row, int column, float[] value) {
        int index = offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];

        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    /**
     * Returns a vector obtained by stacking the columns of the matrix on top of
     * one another.
     * 
     * @return
     */
    public FComplexMatrix1D vectorize() {
        SparseFComplexMatrix1D v = new SparseFComplexMatrix1D(size());
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                v.setQuick(idx++, getQuick(c, r));
            }
        }
        return v;
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the
     * specified value.
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
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     */
    public void setQuick(int row, int column, float re, float im) {
        int index = offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];

        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new float[] { re, im });

    }

    /**
     * Sets up a matrix with a given number of rows and columns.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>(float)columns*rows > Integer.MAX_VALUE</tt>.
     */
    protected void setUp(int rows, int columns) {
        super.setUp(rows, columns);
        this.rowStride = 1;
        this.columnStride = 1;
        this.offset = 0;
    }

    /**
     * Self modifying version of viewDice().
     */
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

    /**
     * Constructs and returns a new <i>slice view</i> representing the rows of
     * the given column. The returned view is backed by this matrix, so changes
     * in the returned view are reflected in this matrix, and vice-versa. To
     * obtain a slice view on subranges, construct a sub-ranging view (<tt>viewPart(...)</tt>),
     * then apply this method to the sub-range view.
     * 
     * @param the
     *            column to fix.
     * @return a new slice view.
     * @throws IllegalArgumentException
     *             if <tt>column < 0 || column >= columns()</tt>.
     * @see #viewRow(int)
     */
    public FComplexMatrix1D viewColumn(int column) {
        checkColumn(column);
        int viewSize = this.rows;
        int viewZero = this.rowZero;
        int viewStride = this.rowStride;
        int[] viewOffsets = this.rowOffsets;
        int viewOffset = this.offset + _columnOffset(_columnRank(column));
        return new SelectedSparseFComplexMatrix1D(viewSize, this.elements, viewZero, viewStride, viewOffsets, viewOffset);
    }

    /**
     * Constructs and returns a new <i>slice view</i> representing the columns
     * of the given row. The returned view is backed by this matrix, so changes
     * in the returned view are reflected in this matrix, and vice-versa. To
     * obtain a slice view on subranges, construct a sub-ranging view (<tt>viewPart(...)</tt>),
     * then apply this method to the sub-range view.
     * 
     * @param the
     *            row to fix.
     * @return a new slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>row < 0 || row >= rows()</tt>.
     * @see #viewColumn(int)
     */
    public FComplexMatrix1D viewRow(int row) {
        checkRow(row);
        int viewSize = this.columns;
        int viewZero = columnZero;
        int viewStride = this.columnStride;
        int[] viewOffsets = this.columnOffsets;
        int viewOffset = this.offset + _rowOffset(_rowRank(row));
        return new SelectedSparseFComplexMatrix1D(viewSize, this.elements, viewZero, viewStride, viewOffsets, viewOffset);
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
    protected FComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseFComplexMatrix2D(this.elements, rowOffsets, columnOffsets, this.offset);
    }

    /**
     * Returns the imaginary part of this matrix
     * 
     * @return the imaginary part
     */
    public FloatMatrix2D getImaginaryPart() {
        FloatMatrix2D Im = new SparseFloatMatrix2D(rows, columns);
        float[] tmp;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                tmp = getQuick(r, c);
                Im.setQuick(r, c, tmp[1]);
            }
        }
        return Im;
    }

    /**
     * Returns the real part of this matrix
     * 
     * @return the real part
     */
    public FloatMatrix2D getRealPart() {
        FloatMatrix2D R = new SparseFloatMatrix2D(rows, columns);
        float[] tmp;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                tmp = getQuick(r, c);
                R.setQuick(r, c, tmp[0]);
            }
        }
        return R;
    }

}
