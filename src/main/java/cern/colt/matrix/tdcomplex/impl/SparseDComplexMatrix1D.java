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
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse hashed 1-d matrix (aka <i>vector</i>) holding <tt>complex</tt>
 * elements. This implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseDComplexMatrix1D extends DComplexMatrix1D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Long, double[]> elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public SparseDComplexMatrix1D(double[] values) {
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
    public SparseDComplexMatrix1D(int size) {
        setUp(size);
        this.elements = new ConcurrentHashMap<Long, double[]>(size / 1000);
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
    protected SparseDComplexMatrix1D(int size, ConcurrentHashMap<Long, double[]> elements, int offset, int stride) {
        setUp(size, offset, stride);
        this.elements = elements;
        this.isNoView = false;
    }

    public DComplexMatrix1D assign(double[] value) {
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

    public synchronized double[] getQuick(int index) {
        double[] elem = elements.get((long) zero + (long) index * (long) stride);
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
        return (long) zero + (long) rank * (long) stride;
    }

    public DComplexMatrix1D like(int size) {
        return new SparseDComplexMatrix1D(size);
    }

    public DComplexMatrix2D like2D(int rows, int columns) {
        return new SparseDComplexMatrix2D(rows, columns);
    }

    public DComplexMatrix2D reshape(final int rows, final int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        final DComplexMatrix2D M = new SparseDComplexMatrix2D(rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            idx = c * rows;
                            for (int r = 0; r < rows; r++) {
                                double[] elem = getQuick(idx++);
                                if ((elem[0] != 0) || (elem[1] != 0)) {
                                    M.setQuick(r, c, elem);
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = 0;
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    double[] elem = getQuick(idx++);
                    if ((elem[0] != 0) || (elem[1] != 0)) {
                        M.setQuick(r, c, elem);
                    }
                }
            }
        }
        return M;
    }

    public DComplexMatrix3D reshape(final int slices, final int rows, final int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        final DComplexMatrix3D M = new SparseDComplexMatrix3D(slices, rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                idx = s * rows * columns + c * rows;
                                for (int r = 0; r < rows; r++) {
                                    double[] elem = getQuick(idx++);
                                    if ((elem[0] != 0) || (elem[1] != 0)) {
                                        M.setQuick(s, r, c, elem);
                                    }
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = 0;
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        double[] elem = getQuick(idx++);
                        if ((elem[0] != 0) || (elem[1] != 0)) {
                            M.setQuick(s, r, c, elem);
                        }
                    }
                }
            }
        }
        return M;
    }

    public synchronized void setQuick(int index, double[] value) {
        long i = (long) zero + (long) index * (long) stride;
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, value);
    }

    public synchronized void setQuick(int index, double re, double im) {
        long i = (long) zero + (long) index * (long) stride;
        if (re == 0 && im == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, new double[] { re, im });
    }

    protected DComplexMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedSparseDComplexMatrix1D(this.elements, offsets);
    }

    public DoubleMatrix1D getImaginaryPart() {
        final DoubleMatrix1D Im = new SparseDoubleMatrix1D(size);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            Im.setQuick(i, getQuick(i)[1]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                Im.setQuick(i, getQuick(i)[1]);
            }
        }
        return Im;
    }

    public DoubleMatrix1D getRealPart() {
        final DoubleMatrix1D Re = new SparseDoubleMatrix1D(size);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            Re.setQuick(i, getQuick(i)[0]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < size; i++) {
                Re.setQuick(i, getQuick(i)[0]);
            }
        }
        return Re;
    }
}
