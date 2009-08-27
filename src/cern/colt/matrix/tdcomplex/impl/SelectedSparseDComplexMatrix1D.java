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

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;

/**
 * Selection view on sparse 1-d matrices holding <tt>complex</tt> elements. This
 * implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class SelectedSparseDComplexMatrix1D extends DComplexMatrix1D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Long, double[]> elements;

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
    protected SelectedSparseDComplexMatrix1D(int size, ConcurrentHashMap<Long, double[]> elements, int zero,
            int stride, int[] offsets, int offset) {
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
    protected SelectedSparseDComplexMatrix1D(ConcurrentHashMap<Long, double[]> elements, int[] offsets) {
        this(offsets.length, elements, 0, 1, offsets, 0);
    }

    protected int _offset(int absRank) {
        return offsets[absRank];
    }

    public double[] getQuick(int index) {
        return elements.get((long) offset + (long) offsets[zero + index * stride]);
    }

    public ConcurrentHashMap<Long, double[]> elements() {
        throw new IllegalAccessError("This method is not supported.");
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */

    protected boolean haveSharedCellsRaw(DComplexMatrix1D other) {
        if (other instanceof SelectedSparseDComplexMatrix1D) {
            SelectedSparseDComplexMatrix1D otherMatrix = (SelectedSparseDComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseDComplexMatrix1D) {
            SparseDComplexMatrix1D otherMatrix = (SparseDComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        // return this.offset + super.index(rank);
        // manually inlined:
        return (long) offset + (long) offsets[zero + rank * stride];
    }

    public DComplexMatrix1D like(int size) {
        return new SparseDComplexMatrix1D(size);
    }

    public DComplexMatrix2D like2D(int rows, int columns) {
        return new SparseDComplexMatrix2D(rows, columns);
    }

    public DComplexMatrix2D reshape(int rows, int columns) {
        throw new IllegalAccessError("This method is not supported.");
    }

    public DComplexMatrix3D reshape(int slices, int rows, int columns) {
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

    public void setQuick(int index, double[] value) {
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

    public void setQuick(int index, double re, double im) {
        long i = (long) offset + (long) offsets[zero + index * stride];
        if (re == 0 && im == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, new double[] { re, im });
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

    protected DComplexMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedSparseDComplexMatrix1D(this.elements, offsets);
    }

    public DoubleMatrix1D getImaginaryPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

    public DoubleMatrix1D getRealPart() {
        throw new IllegalAccessError("This method is not supported.");
    }

}
