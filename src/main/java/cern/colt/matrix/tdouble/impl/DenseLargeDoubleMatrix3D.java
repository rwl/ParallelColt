/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import java.util.concurrent.Future;

import cern.colt.matrix.tdcomplex.impl.DenseLargeDComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import edu.emory.mathcs.jtransforms.dct.DoubleDCT_2D;
import edu.emory.mathcs.jtransforms.dct.DoubleDCT_3D;
import edu.emory.mathcs.jtransforms.dht.DoubleDHT_2D;
import edu.emory.mathcs.jtransforms.dht.DoubleDHT_3D;
import edu.emory.mathcs.jtransforms.dst.DoubleDST_2D;
import edu.emory.mathcs.jtransforms.dst.DoubleDST_3D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 3-d matrix holding <tt>double</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * This data structure allows to store more than 2^31 elements. Internally holds
 * one three-dimensional array, elements[slices][rows][columns]. Note that this
 * implementation is not synchronized.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseLargeDoubleMatrix3D extends WrapperDoubleMatrix3D {

    private static final long serialVersionUID = 1L;

    private double[][][] elements;

    private DoubleFFT_3D fft3;

    private DoubleDCT_3D dct3;

    private DoubleDST_3D dst3;

    private DoubleDHT_3D dht3;

    private DoubleFFT_2D fft2Slices;

    private DoubleDCT_2D dct2Slices;

    private DoubleDST_2D dst2Slices;

    private DoubleDHT_2D dht2Slices;

    public DenseLargeDoubleMatrix3D(int slices, int rows, int columns) {
        super(null);
        try {
            setUp(slices, rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold slices*rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new double[slices][rows][columns];
    }

    /**
     * Computes the 3D discrete cosine transform (DCT-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */

    public void dct3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dct3 == null) {
            dct3 = new DoubleDCT_3D(slices, rows, columns);
        }
        dct3.forward(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D discrete cosine transform (DCT-II) of each slice of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */

    public void dct2Slices(final boolean scale) {
        if (dct2Slices == null) {
            dct2Slices = new DoubleDCT_2D(rows, columns);
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
                            dct2Slices.forward(elements[s], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                dct2Slices.forward(elements[s], scale);
            }
        }
    }

    /**
     * Computes the 3D discrete Hartley transform (DHT) of this matrix.
     */

    public void dht3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dht3 == null) {
            dht3 = new DoubleDHT_3D(slices, rows, columns);
        }
        dht3.forward(elements);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D discrete Hartley transform (DHT) of each slice of this
     * matrix.
     * 
     */

    public void dht2Slices() {
        if (dht2Slices == null) {
            dht2Slices = new DoubleDHT_2D(rows, columns);
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
                            dht2Slices.forward(elements[s]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                dht2Slices.forward(elements[s]);
            }
        }
    }

    /**
     * Computes the 3D discrete sine transform (DST-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */

    public void dst3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dst3 == null) {
            dst3 = new DoubleDST_3D(slices, rows, columns);
        }
        dst3.forward(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D discrete sine transform (DST-II) of each slice of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */

    public void dst2Slices(final boolean scale) {
        if (dst2Slices == null) {
            dst2Slices = new DoubleDST_2D(rows, columns);
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
                            dst2Slices.forward(elements[s], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                dst2Slices.forward(elements[s], scale);
            }
        }
    }

    /**
     * Computes the 3D discrete Fourier transform (DFT) of this matrix. The
     * physical layout of the output data is as follows:
     * 
     * <pre>
     * this[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * this[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * this[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * this[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * this[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * this[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * this[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * this[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * this[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * this[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * this[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * this[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * this[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * this[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * this[0][0][0] = Re[0][0][0], 
     * this[0][0][1] = Re[0][0][n3/2], 
     * this[0][n2/2][0] = Re[0][n2/2][0], 
     * this[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * this[n1/2][0][0] = Re[n1/2][0][0], 
     * this[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * this[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * this[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>getFft3</code>. To get back the original
     * data, use <code>ifft3</code>.
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     */

    public void fft3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.realForward(elements);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Returns new complex matrix which is the 2D discrete Fourier transform
     * (DFT) of each slice of this matrix.
     * 
     * @return the 2D discrete Fourier transform (DFT) of each slice of this
     *         matrix.
     * 
     */

    public DenseLargeDComplexMatrix3D getFft2Slices() {
        if (fft2Slices == null) {
            fft2Slices = new DoubleFFT_2D(rows, columns);
        }
        final DenseLargeDComplexMatrix3D C = new DenseLargeDComplexMatrix3D(slices, rows, columns);
        final double[][][] cElems = C.elements();
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
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                            }
                            fft2Slices.realForwardFull(cElems[s]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                }
                fft2Slices.realForwardFull(cElems[s]);
            }
        }
        return C;
    }

    /**
     * Returns new complex matrix which is the 3D discrete Fourier transform
     * (DFT) of this matrix.
     * 
     * @return the 3D discrete Fourier transform (DFT) of this matrix.
     */

    public DenseLargeDComplexMatrix3D getFft3() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        DenseLargeDComplexMatrix3D C = new DenseLargeDComplexMatrix3D(slices, rows, columns);
        final double[][][] cElems = (C).elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int startslice = j * k;
                final int stopslice;
                if (j == nthreads - 1) {
                    stopslice = slices;
                } else {
                    stopslice = startslice + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int s = startslice; s < stopslice; s++) {
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                            }
                        }

                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                }
            }
        }
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.realForwardFull(cElems);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the 2D inverse of the discrete
     * Fourier transform (IDFT) of each slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @return the 2D inverse of the discrete Fourier transform (IDFT) of each
     *         slice of this matrix.
     * 
     */

    public DenseLargeDComplexMatrix3D getIfft2Slices(final boolean scale) {
        if (fft2Slices == null) {
            fft2Slices = new DoubleFFT_2D(rows, columns);
        }
        final DenseLargeDComplexMatrix3D C = new DenseLargeDComplexMatrix3D(slices, rows, columns);
        final double[][][] cElems = C.elements();
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
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                            }
                            fft2Slices.realInverseFull(cElems[s], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                }
                fft2Slices.realInverseFull(cElems[s], scale);
            }
        }
        return C;
    }

    /**
     * Returns new complex matrix which is the 3D inverse of the discrete
     * Fourier transform (IDFT) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @return the 3D inverse of the discrete Fourier transform (IDFT) of this
     *         matrix.
     * 
     */

    public DenseLargeDComplexMatrix3D getIfft3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        DenseLargeDComplexMatrix3D C = new DenseLargeDComplexMatrix3D(slices, rows, columns);
        final double[][][] cElems = (C).elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int startslice = j * k;
                final int stopslice;
                if (j == nthreads - 1) {
                    stopslice = slices;
                } else {
                    stopslice = startslice + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int s = startslice; s < stopslice; s++) {
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                            }
                        }

                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    System.arraycopy(elements[s][r], 0, cElems[s][r], 0, columns);
                }
            }
        }
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.realInverseFull(cElems, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    public double getQuick(int slice, int row, int column) {
        return elements[slice][row][column];
    }

    /**
     * Computes the 2D inverse of the discrete cosine transform (DCT-III) of
     * each slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */

    public void idct2Slices(final boolean scale) {
        if (dct2Slices == null) {
            dct2Slices = new DoubleDCT_2D(rows, columns);
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
                            dct2Slices.inverse(elements[s], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                dct2Slices.inverse(elements[s], scale);
            }
        }
    }

    /**
     * Computes the 3D inverse of the discrete Hartley transform (IDHT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     * 
     */

    public void idht3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dht3 == null) {
            dht3 = new DoubleDHT_3D(slices, rows, columns);
        }
        dht3.inverse(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete Hartley transform (IDHT) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     * 
     */

    public void idht2Slices(final boolean scale) {
        if (dht2Slices == null) {
            dht2Slices = new DoubleDHT_2D(rows, columns);
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
                            dht2Slices.inverse(elements[s], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                dht2Slices.inverse(elements[s], scale);
            }
        }
    }

    /**
     * Computes the 3D inverse of the discrete cosine transform (DCT-III) of
     * this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */

    public void idct3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dct3 == null) {
            dct3 = new DoubleDCT_3D(slices, rows, columns);
        }
        dct3.inverse(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete sine transform (DST-III) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */

    public void idst2Slices(final boolean scale) {
        if (dst2Slices == null) {
            dst2Slices = new DoubleDST_2D(rows, columns);
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
                            dst2Slices.inverse(elements[s], scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
            ConcurrencyUtils.resetThreadsBeginN_FFT();

        } else {
            for (int s = 0; s < slices; s++) {
                dst2Slices.inverse(elements[s], scale);
            }
        }
    }

    /**
     * Computes the 3D inverse of the discrete sine transform (DST-III) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */

    public void idst3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dst3 == null) {
            dst3 = new DoubleDST_3D(slices, rows, columns);
        }
        dst3.inverse(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 3D inverse of the discrete Fourier transform (IDFT) of this
     * matrix. The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * this[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * this[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * this[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * this[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * this[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * this[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * this[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * this[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * this[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * this[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * this[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * this[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * this[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * this[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * this[0][0][0] = Re[0][0][0], 
     * this[0][0][1] = Re[0][0][n3/2], 
     * this[0][n2/2][0] = Re[0][n2/2][0], 
     * this[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * this[n1/2][0][0] = Re[n1/2][0][0], 
     * this[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * this[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * this[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>getIfft3</code>.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     */

    public void ifft3(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft3 == null) {
            fft3 = new DoubleFFT_3D(slices, rows, columns);
        }
        fft3.realInverse(elements, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public void setQuick(int slice, int row, int column, double value) {
        elements[slice][row][column] = value;
    }

    public double[][][] elements() {
        return elements;
    }

    protected DoubleMatrix3D getContent() {
        return this;
    }

    public DoubleMatrix3D like(int slices, int rows, int columns) {
        return new DenseLargeDoubleMatrix3D(slices, rows, columns);
    }

}
