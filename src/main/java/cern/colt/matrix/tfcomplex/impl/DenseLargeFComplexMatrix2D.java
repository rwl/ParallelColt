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
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>complex</tt> elements.<br>
 * <b>Implementation:</b>
 * <p>
 * This data structure allows to store more than 2^31 elements. Internally holds
 * one two-dimensional array, elements[rows][2*columns]. Complex data is
 * represented by 2 float values in sequence, i.e. elements[row][2*column]
 * constitute the real part and elements[row][2*column+1] constitute the
 * imaginary part. Note that this implementation is not synchronized.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseLargeFComplexMatrix2D extends WrapperFComplexMatrix2D {

    private static final long serialVersionUID = 1L;

    private float[][] elements;

    private FloatFFT_2D fft2;

    private FloatFFT_1D fftRows;

    private FloatFFT_1D fftColumns;

    public DenseLargeFComplexMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new float[rows][2 * columns];
        content = this;
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of this matrix.
     */

    public void fft2() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.complexForward(elements);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the discrete Fourier transform (DFT) of each column of this
     * matrix.
     */

    public void fftColumns() {
        if (fftColumns == null) {
            fftColumns = new FloatFFT_1D(rows);
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            float[] column = (float[]) viewColumn(c).copy().elements();
                            fftColumns.complexForward(column);
                            viewColumn(c).assign(column);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                float[] column = (float[]) viewColumn(c).copy().elements();
                fftColumns.complexForward(column);
                viewColumn(c).assign(column);
            }
        }
    }

    /**
     * Computes the discrete Fourier transform (DFT) of each row of this matrix.
     */

    public void fftRows() {
        if (fftRows == null) {
            fftRows = new FloatFFT_1D(columns);
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            fftRows.complexForward(elements[r]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                fftRows.complexForward(elements[r]);
            }
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
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.complexInverse(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete Fourier transform (IDFT) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */

    public void ifftColumns(final boolean scale) {
        if (fftColumns == null) {
            fftColumns = new FloatFFT_1D(rows);
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            float[] column = (float[]) viewColumn(c).copy().elements();
                            fftColumns.complexInverse(column, scale);
                            viewColumn(c).assign(column);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {

            for (int c = 0; c < columns; c++) {
                float[] column = (float[]) viewColumn(c).copy().elements();
                fftColumns.complexInverse(column, scale);
                viewColumn(c).assign(column);
            }
        }
    }

    /**
     * Computes the inverse of the discrete Fourier transform (IDFT) of each row
     * of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */

    public void ifftRows(final boolean scale) {
        if (fftRows == null) {
            fftRows = new FloatFFT_1D(columns);
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            fftRows.complexInverse(elements[r], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                fftRows.complexInverse(elements[r], scale);
            }
        }
    }

    public float[] getQuick(int row, int column) {
        return new float[] { elements[row][2 * column], elements[row][2 * column + 1] };
    }

    public void setQuick(int row, int column, float[] value) {
        elements[row][2 * column] = value[0];
        elements[row][2 * column + 1] = value[1];
    }

    public void setQuick(int row, int column, float re, float im) {
        elements[row][2 * column] = re;
        elements[row][2 * column + 1] = im;
    }

    public float[][] elements() {
        return elements;
    }

    protected FComplexMatrix2D getContent() {
        return this;
    }

    public FComplexMatrix2D like(int rows, int columns) {
        return new DenseLargeFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix1D like1D(int size) {
        return new DenseFComplexMatrix1D(size);
    }
}
