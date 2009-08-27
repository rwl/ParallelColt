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
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseLargeFloatMatrix2D;
import cern.jet.math.tfcomplex.FComplex;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * 2-d matrix holding <tt>complex</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperFComplexMatrix2D extends FComplexMatrix2D {
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected FComplexMatrix2D content;

    public WrapperFComplexMatrix2D(FComplexMatrix2D newContent) {
        if (newContent != null)
            setUp(newContent.rows(), newContent.columns());
        this.content = newContent;
    }

    public FComplexMatrix2D assign(final float[] values) {
        if (content instanceof DiagonalFComplexMatrix2D) {
            int dlength = ((DiagonalFComplexMatrix2D) content).dlength;
            final float[] elems = ((DiagonalFComplexMatrix2D) content).elements;
            if (values.length != 2 * dlength)
                throw new IllegalArgumentException("Must have same length: length=" + values.length + " 2 * dlength="
                        + 2 * dlength);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, dlength);
                Future<?>[] futures = new Future[nthreads];
                int k = dlength / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == nthreads - 1) ? dlength : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            for (int i = firstIdx; i < lastIdx; i++) {
                                elems[2 * i] = values[2 * i];
                                elems[2 * i + 1] = values[2 * i + 1];
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int i = 0; i < dlength; i++) {
                    elems[2 * i] = values[2 * i];
                    elems[2 * i + 1] = values[2 * i + 1];
                }
            }
            return this;
        } else {
            return super.assign(values);
        }
    }

    public boolean equals(float[] value) {
        if (content instanceof DiagonalFComplexMatrix2D) {
            float epsilon = cern.colt.matrix.tfcomplex.algo.FComplexProperty.DEFAULT.tolerance();
            float[] elements = (float[]) content.elements();
            int dlength = ((DiagonalFComplexMatrix2D) content).dlength;
            float[] x = new float[2];
            float[] diff = new float[2];
            for (int i = 0; i < dlength; i++) {
                x[0] = elements[2 * i];
                x[1] = elements[2 * i + 1];
                diff[0] = Math.abs(value[0] - x[0]);
                diff[1] = Math.abs(value[1] - x[1]);
                if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                        && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                        || (FComplex.isEqual(value, x, epsilon))) {
                    diff[0] = 0;
                    diff[1] = 0;
                }
                if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(value);
        }
    }

    public boolean equals(Object obj) {
        if (content instanceof DiagonalFComplexMatrix2D && obj instanceof DiagonalFComplexMatrix2D) {
            DiagonalFComplexMatrix2D other = (DiagonalFComplexMatrix2D) obj;
            int dlength = ((DiagonalFComplexMatrix2D) content).dlength;
            float epsilon = cern.colt.matrix.tfcomplex.algo.FComplexProperty.DEFAULT.tolerance();
            if (this == obj)
                return true;
            if (!(this != null && obj != null))
                return false;
            DiagonalFComplexMatrix2D A = (DiagonalFComplexMatrix2D) content;
            DiagonalFComplexMatrix2D B = (DiagonalFComplexMatrix2D) obj;
            if (A.columns() != B.columns() || A.rows() != B.rows() || A.diagonalIndex() != B.diagonalIndex()
                    || A.diagonalLength() != B.diagonalLength())
                return false;
            float[] otherElements = other.elements;
            float[] elements = ((DiagonalFComplexMatrix2D) content).elements;
            float[] x = new float[2];
            float[] value = new float[2];
            float[] diff = new float[2];
            for (int i = 0; i < dlength; i++) {
                x[0] = elements[2 * i];
                x[1] = elements[2 * i + 1];
                value[0] = otherElements[2 * i];
                value[1] = otherElements[2 * i + 1];
                diff[0] = Math.abs(value[0] - x[0]);
                diff[1] = Math.abs(value[1] - x[1]);
                if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                        && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                        || (FComplex.isEqual(value, x, epsilon))) {
                    diff[0] = 0;
                    diff[1] = 0;
                }
                if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(obj);
        }
    }

    public Object elements() {
        return content.elements();
    }

    public synchronized float[] getQuick(int row, int column) {
        return content.getQuick(row, column);
    }

    public FComplexMatrix2D like(int rows, int columns) {
        return content.like(rows, columns);
    }

    public FComplexMatrix1D like1D(int size) {
        return content.like1D(size);
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of this matrix.
     * 
     */
    public void fft2() {
        if (content instanceof DenseLargeFComplexMatrix2D) {
            if (isNoView == true) {
                ((DenseLargeFComplexMatrix2D) content).fft2();
            } else {
                DenseLargeFComplexMatrix2D copy = (DenseLargeFComplexMatrix2D) copy();
                copy.fft2();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete Fourier transform (DFT) of each column of this
     * matrix.
     * 
     */
    public void fftColumns() {
        if (content instanceof DenseLargeFComplexMatrix2D) {
            if (isNoView == true) {
                ((DenseLargeFComplexMatrix2D) content).fftColumns();
            } else {
                DenseLargeFComplexMatrix2D copy = (DenseLargeFComplexMatrix2D) copy();
                copy.fftColumns();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete Fourier transform (DFT) of each row of this matrix.
     * 
     */
    public void fftRows() {
        if (content instanceof DenseLargeFComplexMatrix2D) {
            if (isNoView == true) {
                ((DenseLargeFComplexMatrix2D) content).fftRows();
            } else {
                DenseLargeFComplexMatrix2D copy = (DenseLargeFComplexMatrix2D) copy();
                copy.fftRows();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete Fourier transform (DFT) of each
     * column of this matrix.
     * 
     */
    public void ifftColumns(boolean scale) {
        if (content instanceof DenseLargeFComplexMatrix2D) {
            if (isNoView == true) {
                ((DenseLargeFComplexMatrix2D) content).ifftColumns(scale);
            } else {
                DenseLargeFComplexMatrix2D copy = (DenseLargeFComplexMatrix2D) copy();
                copy.ifftColumns(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete Fourier transform (DFT) of each row
     * of this matrix.
     * 
     */
    public void ifftRows(boolean scale) {
        if (content instanceof DenseLargeFComplexMatrix2D) {
            if (isNoView == true) {
                ((DenseLargeFComplexMatrix2D) content).ifftRows(scale);
            } else {
                DenseLargeFComplexMatrix2D copy = (DenseLargeFComplexMatrix2D) copy();
                copy.ifftRows(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D inverse of the discrete Fourier transform (IDFT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void ifft2(boolean scale) {
        if (content instanceof DenseLargeFComplexMatrix2D) {
            if (isNoView == true) {
                ((DenseLargeFComplexMatrix2D) content).ifft2(scale);
            } else {
                DenseLargeFComplexMatrix2D copy = (DenseLargeFComplexMatrix2D) copy();
                copy.ifft2(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    public synchronized void setQuick(int row, int column, float[] value) {
        content.setQuick(row, column, value);
    }

    public synchronized void setQuick(int row, int column, float re, float im) {
        content.setQuick(row, column, re, im);
    }

    public FComplexMatrix1D vectorize() {
        final DenseFComplexMatrix1D v = new DenseFComplexMatrix1D((int) size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = firstColumn * rows;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = 0; r < rows; r++) {
                                v.setQuick(idx++, getQuick(r, c));
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
                    v.setQuick(idx++, getQuick(r, c));
                }
            }
        }
        return v;
    }

    public FComplexMatrix1D viewColumn(int column) {
        return viewDice().viewRow(column);
    }

    public FComplexMatrix2D viewColumnFlip() {
        if (columns == 0)
            return this;
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int row, int column) {
                return content.getQuick(row, columns - 1 - column);
            }

            public synchronized void setQuick(int row, int column, float[] value) {
                content.setQuick(row, columns - 1 - column, value);
            }

            public synchronized void setQuick(int row, int column, float re, float im) {
                content.setQuick(row, columns - 1 - column, re, im);
            }

            public synchronized float[] get(int row, int column) {
                return content.get(row, columns - 1 - column);
            }

            public synchronized void set(int row, int column, float[] value) {
                content.set(row, columns - 1 - column, value);
            }

            public synchronized void set(int row, int column, float re, float im) {
                content.set(row, columns - 1 - column, re, im);
            }

        };
        view.isNoView = false;
        return view;
    }

    public FComplexMatrix2D viewDice() {
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int row, int column) {
                return content.getQuick(column, row);
            }

            public synchronized void setQuick(int row, int column, float[] value) {
                content.setQuick(column, row, value);
            }

            public synchronized void setQuick(int row, int column, float re, float im) {
                content.setQuick(column, row, re, im);
            }

            public synchronized float[] get(int row, int column) {
                return content.get(column, row);
            }

            public synchronized void set(int row, int column, float[] value) {
                content.set(column, row, value);
            }

            public synchronized void set(int row, int column, float re, float im) {
                content.set(column, row, re, im);
            }

        };
        view.rows = columns;
        view.columns = rows;
        view.isNoView = false;

        return view;
    }

    public FComplexMatrix2D viewPart(final int row, final int column, int height, int width) {
        checkBox(row, column, height, width);
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int i, int j) {
                return content.getQuick(row + i, column + j);
            }

            public synchronized void setQuick(int i, int j, float[] value) {
                content.setQuick(row + i, column + j, value);
            }

            public synchronized void setQuick(int i, int j, float re, float im) {
                content.setQuick(row + i, column + j, re, im);
            }

            public synchronized float[] get(int i, int j) {
                return content.get(row + i, column + j);
            }

            public synchronized void set(int i, int j, float[] value) {
                content.set(row + i, column + j, value);
            }

            public synchronized void set(int i, int j, float re, float im) {
                content.set(row + i, column + j, re, im);
            }

        };
        view.rows = height;
        view.columns = width;
        view.isNoView = false;

        return view;
    }

    public FComplexMatrix1D viewRow(int row) {
        checkRow(row);
        return new DelegateFComplexMatrix1D(this, row);
    }

    public FComplexMatrix2D viewRowFlip() {
        if (rows == 0)
            return this;
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int row, int column) {
                return content.getQuick(rows - 1 - row, column);
            }

            public synchronized void setQuick(int row, int column, float[] value) {
                content.setQuick(rows - 1 - row, column, value);
            }

            public synchronized void setQuick(int row, int column, float re, float im) {
                content.setQuick(rows - 1 - row, column, re, im);
            }

            public synchronized float[] get(int row, int column) {
                return content.get(rows - 1 - row, column);
            }

            public synchronized void set(int row, int column, float[] value) {
                content.set(rows - 1 - row, column, value);
            }

            public synchronized void set(int row, int column, float re, float im) {
                content.set(rows - 1 - row, column, re, im);
            }
        };
        view.isNoView = false;

        return view;
    }

    public FComplexMatrix2D viewSelection(int[] rowIndexes, int[] columnIndexes) {
        // check for "all"
        if (rowIndexes == null) {
            rowIndexes = new int[rows];
            for (int i = rows; --i >= 0;)
                rowIndexes[i] = i;
        }
        if (columnIndexes == null) {
            columnIndexes = new int[columns];
            for (int i = columns; --i >= 0;)
                columnIndexes[i] = i;
        }

        checkRowIndexes(rowIndexes);
        checkColumnIndexes(columnIndexes);
        final int[] rix = rowIndexes;
        final int[] cix = columnIndexes;

        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int i, int j) {
                return content.getQuick(rix[i], cix[j]);
            }

            public synchronized void setQuick(int i, int j, float[] value) {
                content.setQuick(rix[i], cix[j], value);
            }

            public synchronized void setQuick(int i, int j, float re, float im) {
                content.setQuick(rix[i], cix[j], re, im);
            }

            public synchronized float[] get(int i, int j) {
                return content.get(rix[i], cix[j]);
            }

            public synchronized void set(int i, int j, float[] value) {
                content.set(rix[i], cix[j], value);
            }

            public synchronized void set(int i, int j, float re, float im) {
                content.set(rix[i], cix[j], re, im);
            }

        };
        view.rows = rowIndexes.length;
        view.columns = columnIndexes.length;
        view.isNoView = false;

        return view;
    }

    public FComplexMatrix2D viewStrides(final int _rowStride, final int _columnStride) {
        if (_rowStride <= 0 || _columnStride <= 0)
            throw new IndexOutOfBoundsException("illegal stride");
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int row, int column) {
                return content.getQuick(_rowStride * row, _columnStride * column);
            }

            public synchronized void setQuick(int row, int column, float[] value) {
                content.setQuick(_rowStride * row, _columnStride * column, value);
            }

            public synchronized void setQuick(int row, int column, float re, float im) {
                content.setQuick(_rowStride * row, _columnStride * column, re, im);
            }

            public synchronized float[] get(int row, int column) {
                return content.get(_rowStride * row, _columnStride * column);
            }

            public synchronized void set(int row, int column, float[] value) {
                content.set(_rowStride * row, _columnStride * column, value);
            }

            public synchronized void set(int row, int column, float re, float im) {
                content.set(_rowStride * row, _columnStride * column, re, im);
            }
        };
        if (rows != 0)
            view.rows = (rows - 1) / _rowStride + 1;
        if (columns != 0)
            view.columns = (columns - 1) / _columnStride + 1;
        view.isNoView = false;

        return view;
    }

    protected FComplexMatrix2D getContent() {
        return content;
    }

    protected FComplexMatrix1D like1D(int size, int offset, int stride) {
        throw new InternalError(); // should never get called
    }

    protected FComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        throw new InternalError(); // should never be called
    }

    public FloatMatrix2D getImaginaryPart() {
        final DenseLargeFloatMatrix2D Im = new DenseLargeFloatMatrix2D(rows, columns);
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

    public FloatMatrix2D getRealPart() {
        final DenseLargeFloatMatrix2D Re = new DenseLargeFloatMatrix2D(rows, columns);
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
