/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.impl;

import java.util.concurrent.Future;

import cern.colt.matrix.AbstractMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Selection view on dense 2-d matrices holding <tt>complex</tt> elements.
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
 * addressing overhead is 1 additional int addition and 2 additional array index
 * accesses per get/set.
 * <p>
 * Note that this implementation is not synchronized.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class SelectedDenseFComplexMatrix2D extends FComplexMatrix2D {

    private static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix.
     */
    protected float[] elements;

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
     * @param elements
     *            the cells.
     * @param rowOffsets
     *            The row offsets of the cells that shall be visible.
     * @param columnOffsets
     *            The column offsets of the cells that shall be visible.
     * @param offset
     */
    protected SelectedDenseFComplexMatrix2D(float[] elements, int[] rowOffsets, int[] columnOffsets, int offset) {
        this(rowOffsets.length, columnOffsets.length, elements, 0, 0, 1, 1, rowOffsets, columnOffsets, offset);
    }

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
    protected SelectedDenseFComplexMatrix2D(int rows, int columns, float[] elements, int rowZero, int columnZero,
            int rowStride, int columnStride, int[] rowOffsets, int[] columnOffsets, int offset) {
        // be sure parameters are valid, we do not check...
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);

        this.elements = elements;
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

    public float[] getQuick(int row, int column) {
        int idxr = rowZero + row * rowStride;
        int idxc = columnZero + column * columnStride;
        return new float[] { elements[offset + rowOffsets[idxr] + columnOffsets[idxc]],
                elements[offset + rowOffsets[idxr] + columnOffsets[idxc] + 1] };
    }

    public float[] elements() {
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

    protected boolean haveSharedCellsRaw(FComplexMatrix2D other) {
        if (other instanceof SelectedDenseFComplexMatrix2D) {
            SelectedDenseFComplexMatrix2D otherMatrix = (SelectedDenseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFComplexMatrix2D) {
            DenseFComplexMatrix2D otherMatrix = (DenseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int row, int column) {
        return this.offset + rowOffsets[rowZero + row * rowStride] + columnOffsets[columnZero + column * columnStride];
    }

    public FComplexMatrix2D like(int rows, int columns) {
        return new DenseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix1D like1D(int size) {
        return new DenseFComplexMatrix1D(size);
    }

    protected FComplexMatrix1D like1D(int size, int zero, int stride) {
        throw new InternalError(); // this method is never called since
        // viewRow() and viewColumn are overridden
        // properly.
    }

    public void setQuick(int row, int column, float[] value) {
        int idxr = rowZero + row * rowStride;
        int idxc = columnZero + column * columnStride;
        elements[offset + rowOffsets[idxr] + columnOffsets[idxc]] = value[0];
        elements[offset + rowOffsets[idxr] + columnOffsets[idxc] + 1] = value[1];
    }

    public FComplexMatrix1D vectorize() {
        throw new IllegalAccessError("This method is not supported.");
    }

    public void setQuick(int row, int column, float re, float im) {
        int idxr = rowZero + row * rowStride;
        int idxc = columnZero + column * columnStride;
        elements[offset + rowOffsets[idxr] + columnOffsets[idxc]] = re;
        elements[offset + rowOffsets[idxr] + columnOffsets[idxc] + 1] = im;
    }

    protected void setUp(int rows, int columns) {
        super.setUp(rows, columns);
        this.rowStride = 1;
        this.columnStride = 2;
        this.offset = 0;
    }

    protected AbstractMatrix2D vDice() {
        super.vDice();
        // swap
        int[] tmp = rowOffsets;
        rowOffsets = columnOffsets;
        columnOffsets = tmp;

        this.isNoView = false;
        return this;
    }

    public FComplexMatrix1D viewColumn(int column) {
        checkColumn(column);
        int viewSize = this.rows;
        int viewZero = this.rowZero;
        int viewStride = this.rowStride;
        int[] viewOffsets = this.rowOffsets;
        int viewOffset = this.offset + _columnOffset(_columnRank(column));
        return new SelectedDenseFComplexMatrix1D(viewSize, this.elements, viewZero, viewStride, viewOffsets, viewOffset);
    }

    public FComplexMatrix1D viewRow(int row) {
        checkRow(row);
        int viewSize = this.columns;
        int viewZero = columnZero;
        int viewStride = this.columnStride;
        int[] viewOffsets = this.columnOffsets;
        int viewOffset = this.offset + _rowOffset(_rowRank(row));
        return new SelectedDenseFComplexMatrix1D(viewSize, this.elements, viewZero, viewStride, viewOffsets, viewOffset);
    }

    protected FComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseFComplexMatrix2D(this.elements, rowOffsets, columnOffsets, this.offset);
    }

    public FloatMatrix2D getRealPart() {
        final DenseFloatMatrix2D R = new DenseFloatMatrix2D(rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        float[] tmp;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                tmp = getQuick(r, c);
                                R.setQuick(r, c, tmp[0]);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    tmp = getQuick(r, c);
                    R.setQuick(r, c, tmp[0]);
                }
            }
        }
        return R;
    }

    public FloatMatrix2D getImaginaryPart() {
        final DenseFloatMatrix2D Im = new DenseFloatMatrix2D(rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        float[] tmp;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                tmp = getQuick(r, c);
                                Im.setQuick(r, c, tmp[1]);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    tmp = getQuick(r, c);
                    Im.setQuick(r, c, tmp[1]);
                }
            }
        }
        return Im;
    }
}
