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
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse hashed 1-d matrix (aka <i>vector</i>) holding <tt>complex</tt>
 * elements. This implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseFComplexMatrix1D extends FComplexMatrix1D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Long, float[]> elements;

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
        this.elements = new ConcurrentHashMap<Long, float[]>(size / 1000);
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
    protected SparseFComplexMatrix1D(int size, ConcurrentHashMap<Long, float[]> elements, int offset, int stride) {
        setUp(size, offset, stride);
        this.elements = elements;
        this.isNoView = false;
    }

    public FComplexMatrix1D assign(float[] value) {
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

    public synchronized float[] getQuick(int index) {
        float[] elem = elements.get((long) zero + (long) index * (long) stride);
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
        return (long) zero + (long) rank * (long) stride;
    }

    public FComplexMatrix1D like(int size) {
        return new SparseFComplexMatrix1D(size);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return new SparseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix2D reshape(final int rows, final int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        final FComplexMatrix2D M = new SparseFComplexMatrix2D(rows, columns);
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
                                float[] elem = getQuick(idx++);
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
                    float[] elem = getQuick(idx++);
                    if ((elem[0] != 0) || (elem[1] != 0)) {
                        M.setQuick(r, c, elem);
                    }
                }
            }
        }
        return M;
    }

    public FComplexMatrix3D reshape(final int slices, final int rows, final int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        final FComplexMatrix3D M = new SparseFComplexMatrix3D(slices, rows, columns);
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
                                    float[] elem = getQuick(idx++);
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
                        float[] elem = getQuick(idx++);
                        if ((elem[0] != 0) || (elem[1] != 0)) {
                            M.setQuick(s, r, c, elem);
                        }
                    }
                }
            }
        }
        return M;
    }

    public synchronized void setQuick(int index, float[] value) {
        long i = (long) zero + (long) index * (long) stride;
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, value);
    }

    public synchronized void setQuick(int index, float re, float im) {
        long i = (long) zero + (long) index * (long) stride;
        if (re == 0 && im == 0)
            this.elements.remove(i);
        else
            this.elements.put(i, new float[] { re, im });
    }

    protected FComplexMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedSparseFComplexMatrix1D(this.elements, offsets);
    }

    public FloatMatrix1D getImaginaryPart() {
        final FloatMatrix1D Im = new SparseFloatMatrix1D(size);
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

    public FloatMatrix1D getRealPart() {
        final FloatMatrix1D Re = new SparseFloatMatrix1D(size);
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
