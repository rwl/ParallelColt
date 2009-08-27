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

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Selection view on dense 1-d matrices holding <tt>complex</tt> elements.
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
 * addressing overhead is 1 additional array index access per get/set.
 * <p>
 * Note that this implementation is not synchronized.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class SelectedDenseFComplexMatrix1D extends FComplexMatrix1D {

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
    protected SelectedDenseFComplexMatrix1D(float[] elements, int[] offsets) {
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
    protected SelectedDenseFComplexMatrix1D(int size, float[] elements, int zero, int stride, int[] offsets, int offset) {
        setUp(size, zero, stride);

        this.elements = elements;
        this.offsets = offsets;
        this.offset = offset;
        this.isNoView = false;
    }

    protected int _offset(int absRank) {
        return offsets[absRank];
    }

    public float[] getQuick(int index) {
        int idx = zero + index * stride;
        return new float[] { elements[offset + offsets[idx]], elements[offset + offsets[idx] + 1] };
    }

    public FloatMatrix1D getRealPart() {
        final DenseFloatMatrix1D R = new DenseFloatMatrix1D(size);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    float[] tmp;

                    public void run() {
                        for (int k = firstIdx; k < lastIdx; k++) {
                            tmp = getQuick(k);
                            R.setQuick(k, tmp[0]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp;
            for (int i = 0; i < size; i++) {
                tmp = getQuick(i);
                R.setQuick(i, tmp[0]);
            }
        }
        return R;
    }

    public FloatMatrix1D getImaginaryPart() {
        final DenseFloatMatrix1D Im = new DenseFloatMatrix1D(size);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    float[] tmp;

                    public void run() {
                        for (int k = firstIdx; k < lastIdx; k++) {
                            tmp = getQuick(k);
                            Im.setQuick(k, tmp[1]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float[] tmp;
            for (int i = 0; i < size; i++) {
                tmp = getQuick(i);
                Im.setQuick(i, tmp[1]);
            }
        }

        return Im;
    }

    public float[] elements() {
        throw new IllegalAccessError("This method is not supported.");
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     * 
     * @param other
     *            matrix
     * @return <tt>true</tt> if both matrices share at least one identical cell.
     */

    protected boolean haveSharedCellsRaw(FComplexMatrix1D other) {
        if (other instanceof SelectedDenseFComplexMatrix1D) {
            SelectedDenseFComplexMatrix1D otherMatrix = (SelectedDenseFComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFComplexMatrix1D) {
            DenseFComplexMatrix1D otherMatrix = (DenseFComplexMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int rank) {
        return offset + offsets[zero + rank * stride];
    }

    public FComplexMatrix1D like(int size) {
        return new DenseFComplexMatrix1D(size);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return new DenseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix2D reshape(int rows, int columns) {
        throw new IllegalAccessError("This method is not supported.");
    }

    public FComplexMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalAccessError("This method is not supported.");
    }

    public void setQuick(int index, float[] value) {
        int idx = zero + index * stride;
        elements[offset + offsets[idx]] = value[0];
        elements[offset + offsets[idx] + 1] = value[1];
    }

    public void setQuick(int index, float re, float im) {
        int idx = zero + index * stride;
        elements[offset + offsets[idx]] = re;
        elements[offset + offsets[idx] + 1] = im;
    }

    protected void setUp(int size) {
        super.setUp(size, 0, 1);
        this.offset = 0;
    }

    protected FComplexMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseFComplexMatrix1D(this.elements, offsets);
    }

}
