/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import java.util.concurrent.ConcurrentHashMap;

import cern.colt.matrix.AbstractMatrix3D;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix3D;

/**
 * Selection view on sparse 3-d matrices holding <tt>complex</tt> elements. Note
 * that this implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.0, 12/10/2007
 */
class SelectedSparseDComplexMatrix3D extends DComplexMatrix3D {
    /**
     * The elements of this matrix.
     */
    protected ConcurrentHashMap<Integer, double[]> elements;

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
    protected SelectedSparseDComplexMatrix3D(ConcurrentHashMap<Integer, double[]> elements, int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets, int offset) {
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

    /**
     * Returns the position of the given absolute rank within the (virtual or
     * non-virtual) internal 1-dimensional array. Default implementation.
     * Override, if necessary.
     * 
     * @param rank
     *            the absolute rank of the element.
     * @return the position.
     */
    protected int _sliceOffset(int absRank) {
        return sliceOffsets[absRank];
    }

    /**
     * Returns the matrix cell value at coordinate <tt>[slice,row,column]</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @return the value at the specified coordinate.
     */
    public double[] getQuick(int slice, int row, int column) {
        return elements.get(offset + sliceOffsets[sliceZero + slice * sliceStride] + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride]);
    }

    public ConcurrentHashMap<Integer, double[]> elements() {
        return elements;
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
    protected boolean haveSharedCellsRaw(DComplexMatrix3D other) {
        if (other instanceof SelectedSparseDComplexMatrix3D) {
            SelectedSparseDComplexMatrix3D otherMatrix = (SelectedSparseDComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseDComplexMatrix3D) {
            SparseDComplexMatrix3D otherMatrix = (SparseDComplexMatrix3D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Returns the position of the given coordinate within the (virtual or
     * non-virtual) internal 1-dimensional array.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the third-coordinate.
     */
    public long index(int slice, int row, int column) {
        return this.offset + sliceOffsets[sliceZero + slice * sliceStride] + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of slices, rows and columns.
     * For example, if the receiver is an instance of type
     * <tt>DenseComplexMatrix3D</tt> the new matrix must also be of type
     * <tt>DenseComplexMatrix3D</tt>, if the receiver is an instance of type
     * <tt>SparseComplexMatrix3D</tt> the new matrix must also be of type
     * <tt>SparseComplexMatrix3D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public DComplexMatrix3D like(int slices, int rows, int columns) {
        return new SparseDComplexMatrix3D(slices, rows, columns);
    }

    /**
     * Returns a vector obtained by stacking the columns of each slice of the
     * matrix on top of one another.
     * 
     * @return
     */
    public DComplexMatrix1D vectorize() {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseComplexMatrix3D</tt> the new matrix must also
     * be of type <tt>DenseComplexMatrix2D</tt>, if the receiver is an instance
     * of type <tt>SparseComplexMatrix3D</tt> the new matrix must also be of
     * type <tt>SparseComplexMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
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
     * @return a new matrix of the corresponding dynamic type.
     */
    protected DComplexMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
        throw new InternalError(); // this method is never called since
        // viewRow() and viewColumn are overridden
        // properly.
    }

    /**
     * Sets the matrix cell at coordinate <tt>[slice,row,column]</tt> to the
     * specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>.
     * 
     * @param slice
     *            the index of the slice-coordinate.
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param value
     *            the value to be filled into the specified cell.
     */
    public void setQuick(int slice, int row, int column, double[] value) {
        int index = offset + sliceOffsets[sliceZero + slice * sliceStride] + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    /**
     * Sets the matrix cell at coordinate <tt>[slice,row,column]</tt> to the
     * specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>.
     * 
     * @param slice
     *            the index of the slice-coordinate.
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
    public void setQuick(int slice, int row, int column, double re, double im) {
        int index = offset + sliceOffsets[sliceZero + slice * sliceStride] + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];
        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new double[] { re, im });

    }

    /**
     * Sets up a matrix with a given number of slices and rows.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>(double)rows*slices > Integer.MAX_VALUE</tt>.
     */
    protected void setUp(int slices, int rows, int columns) {
        super.setUp(slices, rows, columns);
        this.sliceStride = 1;
        this.rowStride = 1;
        this.columnStride = 1;
        this.offset = 0;
    }

    /**
     * Self modifying version of viewDice().
     * 
     * @throws IllegalArgumentException
     *             if some of the parameters are equal or not in range 0..2.
     */
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

    /**
     * Constructs and returns a new 2-dimensional <i>slice view</i> representing
     * the slices and rows of the given column. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * <p>
     * To obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>view().part(...)</tt>), then apply this method to the sub-range view.
     * To obtain 1-dimensional views, apply this method, then apply another
     * slice view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the
     * intermediate 2-dimensional view. To obtain 1-dimensional views on
     * subranges, apply both steps.
     * 
     * @param column
     *            the index of the column to fix.
     * @return a new 2-dimensional slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>column < 0 || column >= columns()</tt>.
     * @see #viewSlice(int)
     * @see #viewRow(int)
     */
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

        return new SelectedSparseDComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero, viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    /**
     * Constructs and returns a new 2-dimensional <i>slice view</i> representing
     * the slices and columns of the given row. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * <p>
     * To obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>view().part(...)</tt>), then apply this method to the sub-range view.
     * To obtain 1-dimensional views, apply this method, then apply another
     * slice view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the
     * intermediate 2-dimensional view. To obtain 1-dimensional views on
     * subranges, apply both steps.
     * 
     * @param row
     *            the index of the row to fix.
     * @return a new 2-dimensional slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>row < 0 || row >= row()</tt>.
     * @see #viewSlice(int)
     * @see #viewColumn(int)
     */
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

        return new SelectedSparseDComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero, viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param sliceOffsets
     *            the offsets of the visible elements.
     * @param rowOffsets
     *            the offsets of the visible elements.
     * @param columnOffsets
     *            the offsets of the visible elements.
     * @return a new view.
     */
    protected DComplexMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseDComplexMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, this.offset);
    }

    /**
     * Constructs and returns a new 2-dimensional <i>slice view</i> representing
     * the rows and columns of the given slice. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * <p>
     * To obtain a slice view on subranges, construct a sub-ranging view (
     * <tt>view().part(...)</tt>), then apply this method to the sub-range view.
     * To obtain 1-dimensional views, apply this method, then apply another
     * slice view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the
     * intermediate 2-dimensional view. To obtain 1-dimensional views on
     * subranges, apply both steps.
     * 
     * @param slice
     *            the index of the slice to fix.
     * @return a new 2-dimensional slice view.
     * @throws IndexOutOfBoundsException
     *             if <tt>slice < 0 || slice >= slices()</tt>.
     * @see #viewRow(int)
     * @see #viewColumn(int)
     */
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

        return new SelectedSparseDComplexMatrix2D(viewRows, viewColumns, this.elements, viewRowZero, viewColumnZero, viewRowStride, viewColumnStride, viewRowOffsets, viewColumnOffsets, viewOffset);
    }

    /**
     * Returns the imaginary part of this matrix
     * 
     * @return the imaginary part
     */
    public DoubleMatrix3D getImaginaryPart() {
        DoubleMatrix3D Im = new SparseDoubleMatrix3D(slices, rows, columns);
        double[] tmp;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    tmp = getQuick(s, r, c);
                    Im.setQuick(s, r, c, tmp[1]);
                }
            }
        }
        return Im;
    }

    /**
     * Returns the real part of this matrix
     * 
     * @return the real part
     */
    public DoubleMatrix3D getRealPart() {
        DoubleMatrix3D R = new SparseDoubleMatrix3D(slices, rows, columns);
        double[] tmp;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    tmp = getQuick(s, r, c);
                    R.setQuick(s, r, c, tmp[0]);
                }
            }
        }
        return R;
    }

}
