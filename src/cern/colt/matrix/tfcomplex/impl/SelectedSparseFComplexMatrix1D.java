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

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix1D;

/**
 * Selection view on sparse 1-d matrices holding <tt>complex</tt> elements. This
 * implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class SelectedSparseFComplexMatrix1D extends FComplexMatrix1D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Long, float[]> elements;

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
    protected SelectedSparseFComplexMatrix1D(int size, ConcurrentHashMap<Long, float[]> elements, int zero, int stride,
            int[] offsets, int offset) {
        setUp(size, zero, stride);

        this.elements = elements;
        this.offsets = offsets;
        this.offset = offset;
        this.isNoView = false;
    }

    /**
     * Constructs a matrix view with the given parameters.
     * 
     * @param elements
     *            the cells.
     * @param indexes
     *            The indexes of the cells that shall be visible.
     */
    protected SelectedSparseFComplexMatrix1D(ConcurrentHashMap<Long, float[]> elements, int[] offsets) {
        this(offsets.length, elements, 0, 1, offsets, 0);
    }

    protected int _offset(int absRank) {
        return offsets[absRank];
    }

    public float[] getQuick(int index) {
        return elements.get((long) offset + (long) offsets[zero + index * stride]);
    }

    public ConcurrentHashMap<Integer, float[]> elements() {
        throw new IllegalAccessError("This method is not supported.");
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */

    protected boolean haveSharedCellsRaw(FComplexMatrix1D other) {
        if (other instanceof SelectedSparseFComplexMatrix1D) {
            SelectedSparseFComplexMatrix1D otherMatrix = (SelectedSparseFComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseFComplexMatrix1D) {
            SparseFComplexMatrix1D otherMatrix = (SparseFComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        // return this.offset + super.index(rank);
        // manually inlined:
        return (long) offset + (long) offsets[zero + rank * stride];
    }

    public FComplexMatrix1D like(int size) {
        return new SparseFComplexMatrix1D(size);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return new SparseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix2D reshape(int rows, int columns) {
        throw new IllegalAccessError("This method is not supported.");
    }

    public FComplexMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalAccessError("This method is not supported.");
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

    public void setQuick(int index, float[] value) {
        long i = (long) offset + (long) offsets[zero + index * stride];
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, value);
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

    public void setQuick(int index, float re, float im) {
        long i = (long) offset + (long) offsets[zero + index * stride];
        if (re == 0 && im == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, new float[] { re, im });
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

    protected FComplexMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedSparseFComplexMatrix1D(this.elements, offsets);
    }

    public FloatMatrix1D getImaginaryPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

    public FloatMatrix1D getRealPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

}
