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
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix1D;

/**
 * Sparse hashed 1-d matrix (aka <i>vector</i>) holding <tt>complex</tt>
 * elements. Note that this implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.0, 12/10/2007
 */
public class SparseFComplexMatrix1D extends FComplexMatrix1D {
    private static final long serialVersionUID = -7792866167410993582L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Integer, float[]> elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public SparseFComplexMatrix1D(float[] values) {
        this(values.length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of cells.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public SparseFComplexMatrix1D(int size) {
        setUp(size);
        this.elements = new ConcurrentHashMap<Integer, float[]>(size / 1000);
    }

    /**
     * Constructs a matrix view with a given number of parameters.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param elements
     *            the cells.
     * @param offset
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    protected SparseFComplexMatrix1D(int size, ConcurrentHashMap<Integer, float[]> elements, int offset, int stride) {
        setUp(size, offset, stride);
        this.elements = elements;
        this.isNoView = false;
    }

    /**
     * Sets all cells to the state specified by <tt>value</tt>.
     * 
     * @param value
     *            the value to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     */
    public FComplexMatrix1D assign(float[] value) {
        // overriden for performance only
        if (this.isNoView && value[0] == 0 && value[1] == 0)
            this.elements.clear();
        else
            super.assign(value);
        return this;
    }

    /**
     * Returns the number of cells having non-zero values.
     */
    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
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
    public float[] getQuick(int index) {
        float[] elem = elements.get(zero + index * stride);
        if(elem != null) {
            return new float[] {elem[0], elem[1]};
        }
        else {
            return new float[2];
        }
    }

    /**
     * Returns the elements of this matrix.
     * 
     * @return the elements
     */
    public ConcurrentHashMap<Integer, float[]> elements() {
        return elements;
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

    /**
     * Returns the position of the element with the given relative rank within
     * the (virtual or non-virtual) internal 1-dimensional array. You may want
     * to override this method for performance.
     * 
     * @param rank
     *            the rank of the element.
     */
    public long index(int rank) {
        return zero + rank * stride;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified size. For example, if the receiver
     * is an instance of type <tt>DenseComplexMatrix1D</tt> the new matrix must
     * also be of type <tt>DenseComplexMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseComplexMatrix1D</tt> the new matrix must also
     * be of type <tt>SparseComplexMatrix1D</tt>, etc. In general, the new
     * matrix should have internal parametrization as similar as possible.
     * 
     * @param size
     *            the number of cell the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public FComplexMatrix1D like(int size) {
        return new SparseFComplexMatrix1D(size);
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseComplexMatrix1D</tt> the new
     * matrix must be of type <tt>DenseComplexMatrix2D</tt>, if the receiver is
     * an instance of type <tt>SparseComplexMatrix1D</tt> the new matrix must be
     * of type <tt>SparseComplexMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public FComplexMatrix2D like2D(int rows, int columns) {
        return new SparseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix2D reshape(int rows, int cols) {
        if (rows * cols != size) {
            throw new IllegalArgumentException("rows*cols != size");
        }
        FComplexMatrix2D M = new SparseFComplexMatrix2D(rows, cols);
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                float[] elem = getQuick(idx++);
                if ((elem[0] != 0) || (elem[1] != 0)) {
                    M.setQuick(r, c, elem);
                }
            }
        }
        return M;
    }

    public FComplexMatrix3D reshape(int slices, int rows, int cols) {
        if (slices * rows * cols != size) {
            throw new IllegalArgumentException("slices*rows*cols != size");
        }
        FComplexMatrix3D M = new SparseFComplexMatrix3D(slices, rows, cols);
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    float[] elem = getQuick(idx++);
                    if ((elem[0] != 0) || (elem[1] != 0)) {
                        M.setQuick(s, r, c, elem);
                    }
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
    public void setQuick(int index, float[] value) {
        int i = zero + index * stride;
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
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     * 
     */
    public void setQuick(int index, float re, float im) {
        int i = zero + index * stride;
        if (re == 0 && im == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, new float[] { re, im });
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
        int n = size();
        FloatMatrix1D Im = new SparseFloatMatrix1D(n);
        float[] tmp = new float[2];
        for (int i = 0; i < n; i++) {
            tmp = getQuick(i);
            Im.setQuick(i, tmp[1]);
        }
        return Im;
    }

    public FloatMatrix1D getRealPart() {
        int n = size();
        FloatMatrix1D R = new SparseFloatMatrix1D(n);
        float[] tmp = new float[2];
        for (int i = 0; i < n; i++) {
            tmp = getQuick(i);
            R.setQuick(i, tmp[0]);
        }
        return R;
    }
}
