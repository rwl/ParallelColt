/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix3D;

/**
 * Selection view on dense 1-d matrices holding <tt>float</tt> elements. First
 * see the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Objects of this class are typically constructed via <tt>viewIndexes</tt>
 * methods on some source matrix. The interface introduced in abstract super
 * classes defines everything a user can do. From a user point of view there is
 * nothing special about this class; it presents the same functionality with the
 * same signatures and semantics as its abstract superclass(es) while
 * introducing no additional functionality. Thus, this class need not be visible
 * to users. By the way, the same principle applies to concrete DenseXXX,
 * SparseXXX classes: they presents the same functionality with the same
 * signatures and semantics as abstract superclass(es) while introducing no
 * additional functionality. Thus, they need not be visible to users, either.
 * Factory methods could hide all these concrete types.
 * <p>
 * This class uses no delegation. Its instances point directly to the data. Cell
 * addressing overhead is 1 additional array index access per get/set.
 * <p>
 * Note that this implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 4*indexes.length</tt>. Thus, an index view with 1000
 * indexes additionally uses 4 KB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * Depends on the parent view holding cells.
 * <p>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.1, 08/22/2007
 */
class SelectedDenseFloatMatrix1D extends FloatMatrix1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /**
     * The elements of this matrix.
     */
    protected float[] elements;

    /**
     * The offsets of visible indexes of this matrix.
     */
    protected int[] offsets;

    /**
     * The offset.
     */
    protected int offset;

    /**
     * Constructs a matrix view with the given parameters.
     * 
     * @param elements
     *            the cells.
     * @param indexes
     *            The indexes of the cells that shall be visible.
     */
    protected SelectedDenseFloatMatrix1D(float[] elements, int[] offsets) {
        this(offsets.length, elements, 0, 1, offsets, 0);
    }

    /**
     * Constructs a matrix view with the given parameters.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param elements
     *            the cells.
     * @param zero
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @param offsets
     *            the offsets of the cells that shall be visible.
     * @param offset
     */
    protected SelectedDenseFloatMatrix1D(int size, float[] elements, int zero, int stride, int[] offsets, int offset) {
        setUp(size, zero, stride);

        this.elements = elements;
        this.offsets = offsets;
        this.offset = offset;
        this.isNoView = false;
    }

    public float[] elements() {
        return elements;
    }

    /**
     * Returns the matrix cell value at coordinate <tt>index</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @return the value of the specified cell.
     */

    public float getQuick(int index) {
        // if (debug) if (index<0 || index>=size) checkIndex(index);
        // return elements[index(index)];
        // manually inlined:
        return elements[offset + offsets[zero + index * stride]];
    }

    /**
     * Returns the position of the element with the given relative rank within
     * the (virtual or non-virtual) internal 1-dimensional array. You may want
     * to override this method for performance.
     * 
     * @param rank
     *            the rank of the element.
     */

    public long index(int rank) {
        // return this.offset + super.index(rank);
        // manually inlined:
        return offset + offsets[zero + rank * stride];
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified size. For example, if the receiver
     * is an instance of type <tt>DenseFloatMatrix1D</tt> the new matrix must
     * also be of type <tt>DenseFloatMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseFloatMatrix1D</tt> the new matrix must also be
     * of type <tt>SparseFloatMatrix1D</tt>, etc. In general, the new matrix
     * should have internal parametrization as similar as possible.
     * 
     * @param size
     *            the number of cell the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */

    public FloatMatrix1D like(int size) {
        return new DenseFloatMatrix1D(size);
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseFloatMatrix1D</tt> the new
     * matrix must be of type <tt>DenseFloatMatrix2D</tt>, if the receiver is an
     * instance of type <tt>SparseFloatMatrix1D</tt> the new matrix must be of
     * type <tt>SparseFloatMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */

    public FloatMatrix2D like2D(int rows, int columns) {
        return new DenseFloatMatrix2D(rows, columns);
    }

    public FloatMatrix2D reshape(int rows, int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        FloatMatrix2D M = new DenseFloatMatrix2D(rows, columns);
        final float[] elementsOther = (float[]) M.elements();
        final int zeroOther = (int) M.index(0, 0);
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int idxOther;
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            idxOther = zeroOther + c * colStrideOther;
            for (int r = 0; r < rows; r++) {
                elementsOther[idxOther] = getQuick(idx++);
                idxOther += rowStrideOther;
            }
        }
        return M;
    }

    public FloatMatrix3D reshape(int slices, int rows, int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        FloatMatrix3D M = new DenseFloatMatrix3D(slices, rows, columns);
        final float[] elementsOther = (float[]) M.elements();
        final int zeroOther = (int) M.index(0, 0, 0);
        final int sliceStrideOther = M.sliceStride();
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int idxOther;
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < columns; c++) {
                idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
                for (int r = 0; r < rows; r++) {
                    elementsOther[idxOther] = getQuick(idx++);
                    idxOther += rowStrideOther;
                }
            }
        }
        return M;
    }

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @param value
     *            the value to be filled into the specified cell.
     */

    public void setQuick(int index, float value) {
        // if (debug) if (index<0 || index>=size) checkIndex(index);
        // elements[index(index)] = value;
        // manually inlined:
        elements[offset + offsets[zero + index * stride]] = value;
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

    protected int _offset(int absRank) {
        return offsets[absRank];
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */

    protected boolean haveSharedCellsRaw(FloatMatrix1D other) {
        if (other instanceof SelectedDenseFloatMatrix1D) {
            SelectedDenseFloatMatrix1D otherMatrix = (SelectedDenseFloatMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFloatMatrix1D) {
            DenseFloatMatrix1D otherMatrix = (DenseFloatMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Sets up a matrix with a given number of cells.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     */

    protected void setUp(int size) {
        super.setUp(size);
        this.stride = 1;
        this.offset = 0;
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param offsets
     *            the offsets of the visible elements.
     * @return a new view.
     */

    protected FloatMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseFloatMatrix1D(this.elements, offsets);
    }
}
