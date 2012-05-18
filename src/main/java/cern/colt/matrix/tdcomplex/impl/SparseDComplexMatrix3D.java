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
import java.util.concurrent.Future;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse hashed 3-d matrix holding <tt>complex</tt> elements. This
 * implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */

public class SparseDComplexMatrix3D extends DComplexMatrix3D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Long, double[]> elements;

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
    public SparseDComplexMatrix3D(double[][][] values) {
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
     *             if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    public SparseDComplexMatrix3D(int slices, int rows, int columns) {
        setUp(slices, rows, columns);
        this.elements = new ConcurrentHashMap<Long, double[]>(slices * rows * (columns / 1000));
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
     *             if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
     * @throws IllegalArgumentException
     *             if <tt>slices<0 || rows<0 || columns<0</tt>.
     */
    protected SparseDComplexMatrix3D(int slices, int rows, int columns, ConcurrentHashMap<Long, double[]> elements,
            int sliceZero, int rowZero, int columnZero, int sliceStride, int rowStride, int columnStride) {
        setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = false;
    }

    public DComplexMatrix3D assign(double[] value) {
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

    public synchronized double[] getQuick(int slice, int row, int column) {
        double[] elem = elements.get((long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride);
        if (elem != null) {
            return new double[] { elem[0], elem[1] };
        } else {
            return new double[2];
        }
    }

    public ConcurrentHashMap<Long, double[]> elements() {
        return elements;
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
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

    public long index(int slice, int row, int column) {
        return (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row * (long) rowStride
                + (long) columnZero + (long) column * columnStride;
    }

    public DComplexMatrix3D like(int slices, int rows, int columns) {
        return new SparseDComplexMatrix3D(slices, rows, columns);
    }

    public DComplexMatrix2D like2D(int rows, int columns) {
        return new SparseDComplexMatrix2D(rows, columns);
    }

    protected DComplexMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
            int columnStride) {
        return new SparseDComplexMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride);
    }

    public synchronized void setQuick(int slice, int row, int column, double[] value) {
        long index = (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    public synchronized void setQuick(int slice, int row, int column, double re, double im) {
        long index = (long) sliceZero + (long) slice * (long) sliceStride + (long) rowZero + (long) row
                * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new double[] { re, im });
    }

    protected DComplexMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseDComplexMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
    }

    public DComplexMatrix1D vectorize() {
        DComplexMatrix1D v = new SparseDComplexMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    public DoubleMatrix3D getImaginaryPart() {
        final DoubleMatrix3D Im = new SparseDoubleMatrix3D(slices, rows, columns);
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

    public DoubleMatrix3D getRealPart() {
        final DoubleMatrix3D Re = new SparseDoubleMatrix3D(slices, rows, columns);
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
