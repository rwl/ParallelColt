/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import java.util.concurrent.Future;

import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 3-d matrix holding <tt>complex</tt> elements.<br>
 * <b>Implementation:</b>
 * <p>
 * This data structure allows to store more than 2^31 elements. Internally holds
 * one three-dimensional array, elements[slices][rows][2*columns]. Complex data
 * is represented by 2 double values in sequence, i.e.
 * elements[slice][row][2*column] constitute the real part and
 * elements[slice][row][2*column+1] constitute the imaginary part. Note that
 * this implementation is not synchronized.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseLargeDComplexMatrix3D extends WrapperDComplexMatrix3D {

    private static final long serialVersionUID = 1L;

    private double[][][] elements;

    private DoubleFFT_3D fft3;

    private DoubleFFT_2D fft2Slices;

    public DenseLargeDComplexMatrix3D(int slices, int rows, int columns) {
        super(null);
        try {
            setUp(slices, rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold slices*rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new double[slices][rows][2 * columns];
    }

    public double[] getQuick(int slice, int row, int column) {
        return new double[] { elements[slice][row][2 * column], elements[slice][row][2 * column + 1] };
    }

    public void setQuick(int slice, int row, int column, double[] value) {
        elements[slice][row][2 * column] = value[0];
        elements[slice][row][2 * column + 1] = value[1];
    }

    public void setQuick(int slice, int row, int column, double re, double im) {
        elements[slice][row][2 * column] = re;
        elements[slice][row][2 * column + 1] = im;
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of each slice of this
     * matrix.
     */

    public void fft2Slices() {
        if (fft2Slices == null) {
            fft2Slices = new DoubleFFT_2D(rows, columns);
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            ConcurrencyUtils.setThreadsBeginN_2D(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            fft2Slices.complexForward(elements[s]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                fft2Slices.complexForward(elements[s]);
            }
        }
    }

    /**
     * Computes the 3D discrete Fourier transform (DFT) of this matrix.
     */

    public void fft3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.complexForward(elements);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete Fourier transform (IDFT) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */

    public void ifft2Slices(final boolean scale) {
        if (fft2Slices == null) {
            fft2Slices = new DoubleFFT_2D(rows, columns);
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            ConcurrencyUtils.setThreadsBeginN_2D(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            fft2Slices.complexInverse(elements[s], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                fft2Slices.complexInverse(elements[s], scale);
            }
        }
    }

    /**
     * Computes the 3D inverse of the discrete Fourier transform (IDFT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */

    public void ifft3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.complexInverse(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public double[][][] elements() {
        return elements;
    }

    protected DComplexMatrix3D getContent() {
        return this;
    }

    public DComplexMatrix3D like(int slices, int rows, int columns) {
        return new DenseLargeDComplexMatrix3D(slices, rows, columns);
    }

}
