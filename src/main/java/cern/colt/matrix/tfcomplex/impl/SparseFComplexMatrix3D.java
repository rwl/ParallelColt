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
import java.util.concurrent.Future;

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse hashed 3-d matrix holding <tt>complex</tt> elements. This
 * implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */

public class SparseFComplexMatrix3D extends FComplexMatrix3D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Long, float[]> elements;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form <tt>values[slice][row][column]</tt> and have
     * exactly the same number of rows in in every slice and exactly the same
     * number of columns in in every row.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 1 &lt;= slice &lt; values.length: values[slice].length != values[slice-1].length</tt>
     *             .
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 1 &lt;= row &lt; values[0].length: values[slice][row].length != values[slice][row-1].length</tt>
     *             .
     */
    public SparseFComplexMatrix3D(float[][][] values) {
        this(values.length, (values.length == 0 ? 0 : values[0].length), (values.length == 0 ? 0
                : values[0].length == 0 ? 0 : values[0][0].length));
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of slices, rows and columns and
     * default memory usage.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>(float)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public SparseFComplexMatrix3D(int slices, int rows, int columns) {
        setUp(slices, rows, columns);
        this.elements = new ConcurrentHashMap<Long, float[]>(slices * rows * (columns / 1000));
    }

    /**
     * Constructs a view with the given parameters.
     * 
     * @param slices
     *            the number of slices the matrix shall have.
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param elements
     *            the cells.
     * @param sliceZero
     *            the position of the first element.
     * @param rowZero
     *            the position of the first element.
     * @param columnZero
     *            the position of the first element.
     * @param sliceStride
     *            the number of elements between two slices, i.e.
     *            <tt>index(k+1,i,j)-index(k,i,j)</tt>.
     * @param rowStride
     *            the number of elements between two rows, i.e.
     *            <tt>index(k,i+1,j)-index(k,i,j)</tt>.
     * @param columnnStride
     *            the number of elements between two columns, i.e.
     *            <tt>index(k,i,j+1)-index(k,i,j)</tt>.
     * @throws IllegalArgumentException
     *             if <tt>(float)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    protected SparseFComplexMatrix3D(int slices, int rows, int columns, ConcurrentHashMap<Long, float[]> elements,
            int sliceZero, int rowZero, int columnZero, int sliceStride, int rowStride, int columnStride) {
        setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = false;
    }

    public FComplexMatrix3D assign(float[] value) {
        // overriden for performance only
        if (this.isNoView && value[0] == 0 && value[1] == 0)
            this.elements.clear();
        else
            super.assign(value);
        return this;
    }

    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
    }

    public float[] getQuick(int slice, int row, int column) {
        float[] elem = elements.get((long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride);
        if (elem != null) {
            return new float[] { elem[0], elem[1] };
        } else {
            return new float[2];
        }
    }

    public ConcurrentHashMap<Long, float[]> elements() {
        return elements;
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
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
        return (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row * (long) rowStride
                + (long) columnZero + (long) column * (long) columnStride;
    }

    public FComplexMatrix3D like(int slices, int rows, int columns) {
        return new SparseFComplexMatrix3D(slices, rows, columns);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return new SparseFComplexMatrix2D(rows, columns);
    }

    protected FComplexMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
            int columnStride) {
        return new SparseFComplexMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride);
    }

    public void setQuick(int slice, int row, int column, float[] value) {
        long index = (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    public void setQuick(int slice, int row, int column, float re, float im) {
        long index = (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new float[] { re, im });
    }

    protected FComplexMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseFComplexMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
    }

    public FComplexMatrix1D vectorize() {
        FComplexMatrix1D v = new SparseFComplexMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    public FloatMatrix3D getImaginaryPart() {
        final FloatMatrix3D Im = new SparseFloatMatrix3D(slices, rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    Im.setQuick(s, r, c, getQuick(s, r, c)[1]);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        Im.setQuick(s, r, c, getQuick(s, r, c)[1]);
                    }
                }
            }
        }
        return Im;
    }

    public FloatMatrix3D getRealPart() {
        final FloatMatrix3D Re = new SparseFloatMatrix3D(slices, rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    Re.setQuick(s, r, c, getQuick(s, r, c)[0]);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        Re.setQuick(s, r, c, getQuick(s, r, c)[0]);
                    }
                }
            }
        }
        return Re;
    }
}
