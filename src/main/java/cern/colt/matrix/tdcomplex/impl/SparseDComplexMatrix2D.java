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
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplex;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse hashed 2-d matrix holding <tt>complex</tt> elements.
 * 
 * This implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseDComplexMatrix2D extends DComplexMatrix2D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Long, double[]> elements;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form <tt>values[row][column]</tt> and have exactly
     * the same number of columns in every row.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 1 &lt;= row &lt; values.length: values[row].length != values[row-1].length</tt>
     *             .
     */
    public SparseDComplexMatrix2D(double[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of rows and columns and default
     * memory usage.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseDComplexMatrix2D(int rows, int columns) {
        setUp(rows, columns);
        this.elements = new ConcurrentHashMap<Long, double[]>(rows * (columns / 1000));
    }

    /**
     * Constructs a view with the given parameters.
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
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    protected SparseDComplexMatrix2D(int rows, int columns, ConcurrentHashMap<Long, double[]> elements, int rowZero,
            int columnZero, int rowStride, int columnStride) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = false;
    }

    public DComplexMatrix2D assign(double[] value) {
        // overriden for performance only
        if (this.isNoView && value[0] == 0 && value[1] == 0)
            this.elements.clear();
        else
            super.assign(value);
        return this;
    }

    public DComplexMatrix2D assign(DComplexMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof SparseDComplexMatrix2D)) {
            return super.assign(source);
        }
        SparseDComplexMatrix2D other = (SparseDComplexMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);

        if (this.isNoView && other.isNoView) { // quickest
            this.elements.clear();
            this.elements.putAll(other.elements);
            return this;
        }
        return super.assign(source);
    }

    public DComplexMatrix2D assign(final DComplexMatrix2D y,
            cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction function) {
        if (!this.isNoView)
            return super.assign(y, function);

        checkShape(y);

        if (function instanceof cern.jet.math.tdcomplex.DComplexPlusMultSecond) {
            // x[i] = x[i] + alpha*y[i]
            final double[] alpha = ((cern.jet.math.tdcomplex.DComplexPlusMultSecond) function).multiplicator;
            if (alpha[0] == 0 && alpha[1] == 1)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tdcomplex.IntIntDComplexFunction() {
                public double[] apply(int i, int j, double[] value) {
                    setQuick(i, j, DComplex.plus(getQuick(i, j), DComplex.mult(alpha, value)));
                    return value;
                }
            });
            return this;
        }
        return super.assign(y, function);
    }

    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
    }

    public synchronized double[] getQuick(int row, int column) {
        double[] elem = this.elements.get((long) rowZero + (long) row * (long) rowStride + (long) columnZero
                + (long) column * (long) columnStride);
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
     * Returns <tt>true</tt> if both matrices share common cells. More formally,
     * returns <tt>true</tt> if at least one of the following conditions is met
     * <ul>
     * <li>the receiver is a view of the other matrix
     * <li>the other matrix is a view of the receiver
     * <li><tt>this == other</tt>
     * </ul>
     */

    protected boolean haveSharedCellsRaw(DComplexMatrix2D other) {
        if (other instanceof SelectedSparseDComplexMatrix2D) {
            SelectedSparseDComplexMatrix2D otherMatrix = (SelectedSparseDComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseDComplexMatrix2D) {
            SparseDComplexMatrix2D otherMatrix = (SparseDComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public long index(int row, int column) {
        return (long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column * (long) columnStride;
    }

    public DComplexMatrix2D like(int rows, int columns) {
        return new SparseDComplexMatrix2D(rows, columns);
    }

    public DComplexMatrix1D like1D(int size) {
        return new SparseDComplexMatrix1D(size);
    }

    protected DComplexMatrix1D like1D(int size, int offset, int stride) {
        return new SparseDComplexMatrix1D(size, this.elements, offset, stride);
    }

    public synchronized void setQuick(int row, int column, double[] value) {
        long index = (long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column
                * (long) columnStride;
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    public DComplexMatrix1D vectorize() {
        final SparseDComplexMatrix1D v = new SparseDComplexMatrix1D((int) size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = 0;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            idx = c * rows;
                            for (int r = 0; r < rows; r++) {
                                double[] elem = getQuick(r, c);
                                if ((elem[0] != 0) || (elem[1] != 0)) {
                                    v.setQuick(idx++, elem);
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
                    double[] elem = getQuick(r, c);
                    if ((elem[0] != 0) || (elem[1] != 0)) {
                        v.setQuick(idx++, elem);
                    }
                }
            }
        }
        return v;
    }

    public synchronized void setQuick(int row, int column, double re, double im) {
        long index = (long) rowZero + (long) row * (long) rowStride + (long) columnZero + (long) column
                * (long) columnStride;
        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new double[] { re, im });

    }

    protected DComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseDComplexMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }

    public DoubleMatrix2D getImaginaryPart() {
        final DoubleMatrix2D Im = new SparseDoubleMatrix2D(rows, columns);
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                Im.setQuick(r, c, getQuick(r, c)[1]);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    Im.setQuick(r, c, getQuick(r, c)[1]);
                }
            }
        }

        return Im;
    }

    public DoubleMatrix2D getRealPart() {
        final DoubleMatrix2D Re = new SparseDoubleMatrix2D(rows, columns);
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
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                Re.setQuick(r, c, getQuick(r, c)[0]);
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    Re.setQuick(r, c, getQuick(r, c)[0]);
                }
            }
        }

        return Re;
    }
}
