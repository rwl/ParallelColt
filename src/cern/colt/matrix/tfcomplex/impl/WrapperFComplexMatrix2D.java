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

    @Override
    public Object elements() {
        return content.elements();
    }

    @Override
    public float[] getQuick(int row, int column) {
        return content.getQuick(row, column);
    }

    @Override
    public FComplexMatrix2D like(int rows, int columns) {
        return content.like(rows, columns);
    }

    @Override
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

    @Override
    public void setQuick(int row, int column, float[] value) {
        content.setQuick(row, column, value);
    }

    @Override
    public void setQuick(int row, int column, float re, float im) {
        content.setQuick(row, column, re, im);
    }

    @Override
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

    @Override
    public FComplexMatrix1D viewColumn(int column) {
        return viewDice().viewRow(column);
    }

    @Override
    public FComplexMatrix2D viewColumnFlip() {
        if (columns == 0)
            return this;
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int row, int column) {
                return content.getQuick(row, columns - 1 - column);
            }

            @Override
            public void setQuick(int row, int column, float[] value) {
                content.setQuick(row, columns - 1 - column, value);
            }

            @Override
            public void setQuick(int row, int column, float re, float im) {
                content.setQuick(row, columns - 1 - column, re, im);
            }

            @Override
            public float[] get(int row, int column) {
                return content.get(row, columns - 1 - column);
            }

            @Override
            public void set(int row, int column, float[] value) {
                content.set(row, columns - 1 - column, value);
            }

            @Override
            public void set(int row, int column, float re, float im) {
                content.set(row, columns - 1 - column, re, im);
            }

        };
        view.isNoView = false;
        return view;
    }

    @Override
    public FComplexMatrix2D viewDice() {
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int row, int column) {
                return content.getQuick(column, row);
            }

            @Override
            public void setQuick(int row, int column, float[] value) {
                content.setQuick(column, row, value);
            }

            @Override
            public void setQuick(int row, int column, float re, float im) {
                content.setQuick(column, row, re, im);
            }

            @Override
            public float[] get(int row, int column) {
                return content.get(column, row);
            }

            @Override
            public void set(int row, int column, float[] value) {
                content.set(column, row, value);
            }

            @Override
            public void set(int row, int column, float re, float im) {
                content.set(column, row, re, im);
            }

        };
        view.rows = columns;
        view.columns = rows;
        view.isNoView = false;

        return view;
    }

    @Override
    public FComplexMatrix2D viewPart(final int row, final int column, int height, int width) {
        checkBox(row, column, height, width);
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int i, int j) {
                return content.getQuick(row + i, column + j);
            }

            @Override
            public void setQuick(int i, int j, float[] value) {
                content.setQuick(row + i, column + j, value);
            }

            @Override
            public void setQuick(int i, int j, float re, float im) {
                content.setQuick(row + i, column + j, re, im);
            }

            @Override
            public float[] get(int i, int j) {
                return content.get(row + i, column + j);
            }

            @Override
            public void set(int i, int j, float[] value) {
                content.set(row + i, column + j, value);
            }

            @Override
            public void set(int i, int j, float re, float im) {
                content.set(row + i, column + j, re, im);
            }

        };
        view.rows = height;
        view.columns = width;
        view.isNoView = false;

        return view;
    }

    @Override
    public FComplexMatrix1D viewRow(int row) {
        checkRow(row);
        return new DelegateFComplexMatrix1D(this, row);
    }

    @Override
    public FComplexMatrix2D viewRowFlip() {
        if (rows == 0)
            return this;
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int row, int column) {
                return content.getQuick(rows - 1 - row, column);
            }

            @Override
            public void setQuick(int row, int column, float[] value) {
                content.setQuick(rows - 1 - row, column, value);
            }

            @Override
            public void setQuick(int row, int column, float re, float im) {
                content.setQuick(rows - 1 - row, column, re, im);
            }

            @Override
            public float[] get(int row, int column) {
                return content.get(rows - 1 - row, column);
            }

            @Override
            public void set(int row, int column, float[] value) {
                content.set(rows - 1 - row, column, value);
            }

            @Override
            public void set(int row, int column, float re, float im) {
                content.set(rows - 1 - row, column, re, im);
            }
        };
        view.isNoView = false;

        return view;
    }

    @Override
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

            @Override
            public float[] getQuick(int i, int j) {
                return content.getQuick(rix[i], cix[j]);
            }

            @Override
            public void setQuick(int i, int j, float[] value) {
                content.setQuick(rix[i], cix[j], value);
            }

            @Override
            public void setQuick(int i, int j, float re, float im) {
                content.setQuick(rix[i], cix[j], re, im);
            }

            @Override
            public float[] get(int i, int j) {
                return content.get(rix[i], cix[j]);
            }

            @Override
            public void set(int i, int j, float[] value) {
                content.set(rix[i], cix[j], value);
            }

            @Override
            public void set(int i, int j, float re, float im) {
                content.set(rix[i], cix[j], re, im);
            }

        };
        view.rows = rowIndexes.length;
        view.columns = columnIndexes.length;
        view.isNoView = false;

        return view;
    }

    @Override
    public FComplexMatrix2D viewStrides(final int _rowStride, final int _columnStride) {
        if (_rowStride <= 0 || _columnStride <= 0)
            throw new IndexOutOfBoundsException("illegal stride");
        WrapperFComplexMatrix2D view = new WrapperFComplexMatrix2D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int row, int column) {
                return content.getQuick(_rowStride * row, _columnStride * column);
            }

            @Override
            public void setQuick(int row, int column, float[] value) {
                content.setQuick(_rowStride * row, _columnStride * column, value);
            }

            @Override
            public void setQuick(int row, int column, float re, float im) {
                content.setQuick(_rowStride * row, _columnStride * column, re, im);
            }

            @Override
            public float[] get(int row, int column) {
                return content.get(_rowStride * row, _columnStride * column);
            }

            @Override
            public void set(int row, int column, float[] value) {
                content.set(_rowStride * row, _columnStride * column, value);
            }

            @Override
            public void set(int row, int column, float re, float im) {
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

    @Override
    protected FComplexMatrix2D getContent() {
        return content;
    }

    @Override
    protected FComplexMatrix1D like1D(int size, int offset, int stride) {
        throw new InternalError(); // should never get called
    }

    @Override
    protected FComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        throw new InternalError(); // should never be called
    }

    @Override
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

    @Override
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
