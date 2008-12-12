/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is JTransforms.
 *
 * The Initial Developer of the Original Code is
 * Piotr Wendykier, Emory University.
 * Portions created by the Initial Developer are Copyright (C) 2007
 * the Initial Developer. All Rights Reserved.
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or
 * the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the MPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the MPL, the GPL or the LGPL.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jtransforms.fft;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Computes 3D Discrete Fourier Transform (DFT) of complex and real, double
 * precision data. The sizes of all three dimensions can be arbitrary numbers.
 * This is a parallel implementation of split-radix and mixed-radix algorithms
 * optimized for SMP systems. <br>
 * <br>
 * This code is derived from General Purpose FFT Package written by Takuya Ooura
 * (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DoubleFFT_3D {

    private int n1;

    private int n2;

    private int n3;

    private int sliceStride;

    private int rowStride;

    private double[] t;

    private DoubleFFT_1D fftn1, fftn2, fftn3;

    private int oldNthreads;

    private int nt;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of DoubleFFT_3D.
     * 
     * @param n1
     *            number of slices
     * @param n2
     *            number of rows
     * @param n3
     *            number of columns
     * 
     */
    public DoubleFFT_3D(int n1, int n2, int n3) {
        if (n1 <= 1 || n2 <= 1 || n3 <= 1) {
            throw new IllegalArgumentException("n1, n2 and n3 must be greater than 1");
        }
        this.n1 = n1;
        this.n2 = n2;
        this.n3 = n3;
        this.sliceStride = n2 * n3;
        this.rowStride = n3;
        if (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D()) {
            this.useThreads = true;
        }
        if (ConcurrencyUtils.isPowerOf2(n1) && ConcurrencyUtils.isPowerOf2(n2) && ConcurrencyUtils.isPowerOf2(n3)) {
            isPowerOfTwo = true;
            oldNthreads = ConcurrencyUtils.getNumberOfProcessors();
            nt = n1;
            if (nt < n2) {
                nt = n2;
            }
            nt *= 8;
            if (oldNthreads > 1) {
                nt *= oldNthreads;
            }
            if (2 * n3 == 4) {
                nt >>= 1;
            } else if (2 * n3 < 4) {
                nt >>= 2;
            }
            t = new double[nt];
        }
        fftn1 = new DoubleFFT_1D(n1);
        if (n1 == n2) {
            fftn2 = fftn1;
        } else {
            fftn2 = new DoubleFFT_1D(n2);
        }
        if (n1 == n3) {
            fftn3 = fftn1;
        } else if (n2 == n3) {
            fftn3 = fftn2;
        } else {
            fftn3 = new DoubleFFT_1D(n3);
        }

    }

    /**
     * Computes 3D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array addressed in slice-major,
     * then row-major, then column-major, in order of significance, i.e. element
     * (i,j,k) of 3D array x[n1][n2][2*n3] is stored in a[i*sliceStride +
     * j*rowStride + k], where sliceStride = n2 * 2 * n3 and rowStride = 2 * n3.
     * Complex number is stored as two double values in sequence: the real and
     * imaginary part, i.e. the input array must be of size n1*n2*2*n3. The
     * physical layout of the input data is as follows:
     * 
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3], 
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;=k3&lt;n3,
     * </pre>
     * 
     * @param a
     *            data to transform
     */
    public void complexForward(final double[] a) {
        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if (isPowerOfTwo) {
            int oldn3 = n3;
            n3 = 2 * n3;

            sliceStride = n2 * n3;
            rowStride = n3;

            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, -1, a, true);
                cdft3db_subth(-1, a, true);
            } else {
                xdft3da_sub2(0, -1, a, true);
                cdft3db_sub(-1, a, true);
            }
            n3 = oldn3;
            sliceStride = n2 * n3;
            rowStride = n3;
        } else {
            sliceStride = 2 * n2 * n3;
            rowStride = 2 * n3;
            if ((nthreads > 1) && useThreads && (n1 >= nthreads) && (n2 >= nthreads) && (n3 >= nthreads)) {
                Future[] futures = new Future[nthreads];
                int p = n1 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int s = startSlice; s < stopSlice; s++) {
                                int idx1 = s * sliceStride;
                                for (int r = 0; r < n2; r++) {
                                    fftn3.complexForward(a, idx1 + r * rowStride);
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n2];
                            for (int s = startSlice; s < stopSlice; s++) {
                                int idx1 = s * sliceStride;
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int r = 0; r < n2; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftn2.complexForward(temp);
                                    for (int r = 0; r < n2; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                p = n2 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthreads - 1) {
                        stopRow = n2;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n1];
                            for (int r = startRow; r < stopRow; r++) {
                                int idx1 = r * rowStride;
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int s = 0; s < n1; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftn1.complexForward(temp);
                                    for (int s = 0; s < n1; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            } else {
                for (int s = 0; s < n1; s++) {
                    int idx1 = s * sliceStride;
                    for (int r = 0; r < n2; r++) {
                        fftn3.complexForward(a, idx1 + r * rowStride);
                    }
                }

                double[] temp = new double[2 * n2];
                for (int s = 0; s < n1; s++) {
                    int idx1 = s * sliceStride;
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int r = 0; r < n2; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftn2.complexForward(temp);
                        for (int r = 0; r < n2; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }

                temp = new double[2 * n1];
                for (int r = 0; r < n2; r++) {
                    int idx1 = r * rowStride;
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int s = 0; s < n1; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftn1.complexForward(temp);
                        for (int s = 0; s < n1; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }
            }
            sliceStride = n2 * n3;
            rowStride = n3;
        }
    }

    /**
     * Computes 3D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 3D array. Complex data is
     * represented by 2 double values in sequence: the real and imaginary part,
     * i.e. the input array must be of size n1 by n2 by 2*n3. The physical
     * layout of the input data is as follows:
     * 
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3], 
     * a[k1][k2][2*k3+1] = Im[k1][k2][k3], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;=k3&lt;n3,
     * </pre>
     * 
     * @param a
     *            data to transform
     */
    public void complexForward(final double[][][] a) {
        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if (isPowerOfTwo) {
            int oldn3 = n3;
            n3 = 2 * n3;

            sliceStride = n2 * n3;
            rowStride = n3;

            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, -1, a, true);
                cdft3db_subth(-1, a, true);
            } else {
                xdft3da_sub2(0, -1, a, true);
                cdft3db_sub(-1, a, true);
            }
            n3 = oldn3;
            sliceStride = n2 * n3;
            rowStride = n3;
        } else {
            if ((nthreads > 1) && useThreads && (n1 >= nthreads) && (n2 >= nthreads) && (n3 >= nthreads)) {
                Future[] futures = new Future[nthreads];
                int p = n1 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int s = startSlice; s < stopSlice; s++) {
                                for (int r = 0; r < n2; r++) {
                                    fftn3.complexForward(a[s][r]);
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n2];
                            for (int s = startSlice; s < stopSlice; s++) {
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int r = 0; r < n2; r++) {
                                        int idx4 = 2 * r;
                                        temp[idx4] = a[s][r][idx2];
                                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                                    }
                                    fftn2.complexForward(temp);
                                    for (int r = 0; r < n2; r++) {
                                        int idx4 = 2 * r;
                                        a[s][r][idx2] = temp[idx4];
                                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                p = n2 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthreads - 1) {
                        stopRow = n2;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n1];
                            for (int r = startRow; r < stopRow; r++) {
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int s = 0; s < n1; s++) {
                                        int idx4 = 2 * s;
                                        temp[idx4] = a[s][r][idx2];
                                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                                    }
                                    fftn1.complexForward(temp);
                                    for (int s = 0; s < n1; s++) {
                                        int idx4 = 2 * s;
                                        a[s][r][idx2] = temp[idx4];
                                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            } else {
                for (int s = 0; s < n1; s++) {
                    for (int r = 0; r < n2; r++) {
                        fftn3.complexForward(a[s][r]);
                    }
                }

                double[] temp = new double[2 * n2];
                for (int s = 0; s < n1; s++) {
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int r = 0; r < n2; r++) {
                            int idx4 = 2 * r;
                            temp[idx4] = a[s][r][idx2];
                            temp[idx4 + 1] = a[s][r][idx2 + 1];
                        }
                        fftn2.complexForward(temp);
                        for (int r = 0; r < n2; r++) {
                            int idx4 = 2 * r;
                            a[s][r][idx2] = temp[idx4];
                            a[s][r][idx2 + 1] = temp[idx4 + 1];
                        }
                    }
                }

                //				double elapsedTime = System.nanoTime();

                temp = new double[2 * n1];
                for (int r = 0; r < n2; r++) {
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int s = 0; s < n1; s++) {
                            int idx4 = 2 * s;
                            temp[idx4] = a[s][r][idx2];
                            temp[idx4 + 1] = a[s][r][idx2 + 1];
                        }
                        fftn1.complexForward(temp);
                        for (int s = 0; s < n1; s++) {
                            int idx4 = 2 * s;
                            a[s][r][idx2] = temp[idx4];
                            a[s][r][idx2 + 1] = temp[idx4 + 1];
                        }
                    }
                }

                //				elapsedTime = System.nanoTime() - elapsedTime;
                //				System.out.println("elapsedTime = " + elapsedTime / 1000000.0);

            }
        }
    }

    /**
     * Computes 3D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[n1][n2][2*n3] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = n2 * 2 * n3 and
     * rowStride = 2 * n3. Complex number is stored as two double values in
     * sequence: the real and imaginary part, i.e. the input array must be of
     * size n1*n2*2*n3. The physical layout of the input data is as follows:
     * 
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3], 
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;=k3&lt;n3,
     * </pre>
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void complexInverse(final double[] a, final boolean scale) {

        int nthreads = ConcurrencyUtils.getNumberOfProcessors();

        if (isPowerOfTwo) {
            int oldn3 = n3;
            n3 = 2 * n3;
            sliceStride = n2 * n3;
            rowStride = n3;
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, 1, a, scale);
                cdft3db_subth(1, a, scale);
            } else {
                xdft3da_sub2(0, 1, a, scale);
                cdft3db_sub(1, a, scale);
            }
            n3 = oldn3;
            sliceStride = n2 * n3;
            rowStride = n3;
        } else {
            sliceStride = 2 * n2 * n3;
            rowStride = 2 * n3;
            if ((nthreads > 1) && useThreads && (n1 >= nthreads) && (n2 >= nthreads) && (n3 >= nthreads)) {
                Future[] futures = new Future[nthreads];
                int p = n1 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int s = startSlice; s < stopSlice; s++) {
                                int idx1 = s * sliceStride;
                                for (int r = 0; r < n2; r++) {
                                    fftn3.complexInverse(a, idx1 + r * rowStride, scale);
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n2];
                            for (int s = startSlice; s < stopSlice; s++) {
                                int idx1 = s * sliceStride;
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int r = 0; r < n2; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftn2.complexInverse(temp, scale);
                                    for (int r = 0; r < n2; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                p = n2 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthreads - 1) {
                        stopRow = n2;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n1];
                            for (int r = startRow; r < stopRow; r++) {
                                int idx1 = r * rowStride;
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int s = 0; s < n1; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftn1.complexInverse(temp, scale);
                                    for (int s = 0; s < n1; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            } else {
                for (int s = 0; s < n1; s++) {
                    int idx1 = s * sliceStride;
                    for (int r = 0; r < n2; r++) {
                        fftn3.complexInverse(a, idx1 + r * rowStride, scale);
                    }
                }
                double[] temp = new double[2 * n2];
                for (int s = 0; s < n1; s++) {
                    int idx1 = s * sliceStride;
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int r = 0; r < n2; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftn2.complexInverse(temp, scale);
                        for (int r = 0; r < n2; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }
                temp = new double[2 * n1];
                for (int r = 0; r < n2; r++) {
                    int idx1 = r * rowStride;
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int s = 0; s < n1; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftn1.complexInverse(temp, scale);
                        for (int s = 0; s < n1; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }
            }
            sliceStride = n2 * n3;
            rowStride = n3;
        }
    }

    /**
     * Computes 3D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in a 3D array. Complex data is
     * represented by 2 double values in sequence: the real and imaginary part,
     * i.e. the input array must be of size n1 by n2 by 2*n3. The physical
     * layout of the input data is as follows:
     * 
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3], 
     * a[k1][k2][2*k3+1] = Im[k1][k2][k3], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;=k3&lt;n3,
     * </pre>
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void complexInverse(final double[][][] a, final boolean scale) {
        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if (isPowerOfTwo) {
            int oldn3 = n3;
            n3 = 2 * n3;
            sliceStride = n2 * n3;
            rowStride = n3;
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, 1, a, scale);
                cdft3db_subth(1, a, scale);
            } else {
                xdft3da_sub2(0, 1, a, scale);
                cdft3db_sub(1, a, scale);
            }
            n3 = oldn3;
            sliceStride = n2 * n3;
            rowStride = n3;
        } else {
            if ((nthreads > 1) && useThreads && (n1 >= nthreads) && (n2 >= nthreads) && (n3 >= nthreads)) {
                Future[] futures = new Future[nthreads];
                int p = n1 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int s = startSlice; s < stopSlice; s++) {
                                for (int r = 0; r < n2; r++) {
                                    fftn3.complexInverse(a[s][r], scale);
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                for (int l = 0; l < nthreads; l++) {
                    final int startSlice = l * p;
                    final int stopSlice;
                    if (l == nthreads - 1) {
                        stopSlice = n1;
                    } else {
                        stopSlice = startSlice + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n2];
                            for (int s = startSlice; s < stopSlice; s++) {
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int r = 0; r < n2; r++) {
                                        int idx4 = 2 * r;
                                        temp[idx4] = a[s][r][idx2];
                                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                                    }
                                    fftn2.complexInverse(temp, scale);
                                    for (int r = 0; r < n2; r++) {
                                        int idx4 = 2 * r;
                                        a[s][r][idx2] = temp[idx4];
                                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                p = n2 / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthreads - 1) {
                        stopRow = n2;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            double[] temp = new double[2 * n1];
                            for (int r = startRow; r < stopRow; r++) {
                                for (int c = 0; c < n3; c++) {
                                    int idx2 = 2 * c;
                                    for (int s = 0; s < n1; s++) {
                                        int idx4 = 2 * s;
                                        temp[idx4] = a[s][r][idx2];
                                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                                    }
                                    fftn1.complexInverse(temp, scale);
                                    for (int s = 0; s < n1; s++) {
                                        int idx4 = 2 * s;
                                        a[s][r][idx2] = temp[idx4];
                                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthreads; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            } else {
                for (int s = 0; s < n1; s++) {
                    for (int r = 0; r < n2; r++) {
                        fftn3.complexInverse(a[s][r], scale);
                    }
                }
                double[] temp = new double[2 * n2];
                for (int s = 0; s < n1; s++) {
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int r = 0; r < n2; r++) {
                            int idx4 = 2 * r;
                            temp[idx4] = a[s][r][idx2];
                            temp[idx4 + 1] = a[s][r][idx2 + 1];
                        }
                        fftn2.complexInverse(temp, scale);
                        for (int r = 0; r < n2; r++) {
                            int idx4 = 2 * r;
                            a[s][r][idx2] = temp[idx4];
                            a[s][r][idx2 + 1] = temp[idx4 + 1];
                        }
                    }
                }
                temp = new double[2 * n1];
                for (int r = 0; r < n2; r++) {
                    for (int c = 0; c < n3; c++) {
                        int idx2 = 2 * c;
                        for (int s = 0; s < n1; s++) {
                            int idx4 = 2 * s;
                            temp[idx4] = a[s][r][idx2];
                            temp[idx4 + 1] = a[s][r][idx2 + 1];
                        }
                        fftn1.complexInverse(temp, scale);
                        for (int s = 0; s < n1; s++) {
                            int idx4 = 2 * s;
                            a[s][r][idx2] = temp[idx4];
                            a[s][r][idx2 + 1] = temp[idx4 + 1];
                        }
                    }
                }
            }
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[n1][n2][2*n3] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = n2 * 2 * n3 and
     * rowStride = 2 * n3. The physical layout of the output data is as follows:
     * 
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * a[k1*sliceStride + k2*rowStride] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * a[k1*sliceStride + k2*rowStride + 1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * a[k1*sliceStride + (n2-k2)*rowStride + 1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * a[k1*sliceStride + (n2-k2)*rowStride] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[k1*sliceStride] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * a[k1*sliceStride + 1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * a[k1*sliceStride + (n2/2)*rowStride] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * a[k1*sliceStride + (n2/2)*rowStride + 1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * a[(n1-k1)*sliceStride + 1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * a[(n1-k1)*sliceStride] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * a[(n1-k1)*sliceStride + (n2/2)*rowStride + 1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * a[(n1-k1)*sliceStride + (n2/2) * rowStride] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * a[0] = Re[0][0][0], 
     * a[1] = Re[0][0][n3/2], 
     * a[(n2/2)*rowStride] = Re[0][n2/2][0], 
     * a[(n2/2)*rowStride + 1] = Re[0][n2/2][n3/2], 
     * a[(n1/2)*sliceStride] = Re[n1/2][0][0], 
     * a[(n1/2)*sliceStride + 1] = Re[n1/2][0][n3/2], 
     * a[(n1/2)*sliceStride + (n2/2)*rowStride] = Re[n1/2][n2/2][0], 
     * a[(n1/2)*sliceStride + (n2/2)*rowStride + 1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForward(double[] a) {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("n1, n2 and n3 must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth1(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub1(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 3D array. The physical
     * layout of the output data is as follows:
     * 
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * a[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * a[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * a[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * a[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * a[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * a[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * a[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * a[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * a[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * a[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * a[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * a[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * a[0][0][0] = Re[0][0][0], 
     * a[0][0][1] = Re[0][0][n3/2], 
     * a[0][n2/2][0] = Re[0][n2/2][0], 
     * a[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * a[n1/2][0][0] = Re[n1/2][0][0], 
     * a[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * a[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * a[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForward(double[][][] a) {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("n1, n2 and n3 must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth1(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub1(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size n1*n2*2*n3, with only the first n1*n2*n3 elements
     * filled with real data. To get back the original data, use
     * <code>complexInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForwardFull(double[] a) {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size n1 by n2 by 2*n3, with only the first n1 by n2 by
     * n3 elements filled with real data. To get back the original data, use
     * <code>complexInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForwardFull(double[][][] a) {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[n1][n2][2*n3] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = n2 * 2 * n3 and
     * rowStride = 2 * n3. The physical layout of the input data has to be as
     * follows:
     * 
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * a[k1*sliceStride + k2*rowStride] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * a[k1*sliceStride + k2*rowStride + 1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * a[k1*sliceStride + (n2-k2)*rowStride + 1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * a[k1*sliceStride + (n2-k2)*rowStride] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[k1*sliceStride] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * a[k1*sliceStride + 1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * a[k1*sliceStride + (n2/2)*rowStride] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * a[k1*sliceStride + (n2/2)*rowStride + 1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * a[(n1-k1)*sliceStride + 1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * a[(n1-k1)*sliceStride] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * a[(n1-k1)*sliceStride + (n2/2)*rowStride + 1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * a[(n1-k1)*sliceStride + (n2/2) * rowStride] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * a[0] = Re[0][0][0], 
     * a[1] = Re[0][0][n3/2], 
     * a[(n2/2)*rowStride] = Re[0][n2/2][0], 
     * a[(n2/2)*rowStride + 1] = Re[0][n2/2][n3/2], 
     * a[(n1/2)*sliceStride] = Re[n1/2][0][0], 
     * a[(n1/2)*sliceStride + 1] = Re[n1/2][0][n3/2], 
     * a[(n1/2)*sliceStride + (n2/2)*rowStride] = Re[n1/2][n2/2][0], 
     * a[(n1/2)*sliceStride + (n2/2)*rowStride + 1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>realInverseFull</code>.
     * 
     * @param a
     *            data to transform
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverse(double[] a, boolean scale) {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("n1, n2 and n3 must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                rdft3d_sub(-1, a);
                cdft3db_subth(1, a, scale);
                xdft3da_subth1(1, 1, a, scale);
            } else {
                rdft3d_sub(-1, a);
                cdft3db_sub(1, a, scale);
                xdft3da_sub1(1, 1, a, scale);
            }
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 3D array. The physical
     * layout of the input data has to be as follows:
     * 
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * a[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * a[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * a[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * a[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * a[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * a[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * a[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * a[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * a[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * a[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * a[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * a[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * a[0][0][0] = Re[0][0][0], 
     * a[0][0][1] = Re[0][0][n3/2], 
     * a[0][n2/2][0] = Re[0][n2/2][0], 
     * a[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * a[n1/2][0][0] = Re[n1/2][0][0], 
     * a[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * a[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * a[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>realInverseFull</code>.
     * 
     * @param a
     *            data to transform
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverse(double[][][] a, boolean scale) {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("n1, n2 and n3 must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                rdft3d_sub(-1, a);
                cdft3db_subth(1, a, scale);
                xdft3da_subth1(1, 1, a, scale);
            } else {
                rdft3d_sub(-1, a);
                cdft3db_sub(1, a, scale);
                xdft3da_sub1(1, 1, a, scale);
            }
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size n1*n2*2*n3, with only the first n1*n2*n3 elements
     * filled with real data.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverseFull(double[] a, boolean scale) {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, 1, a, scale);
                cdft3db_subth(1, a, scale);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, 1, a, scale);
                cdft3db_sub(1, a, scale);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size n1 by n2 by 2*n3, with only the first n1 by n2 by
     * n3 elements filled with real data.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverseFull(double[][][] a, boolean scale) {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfProcessors();
            if (nthreads != oldNthreads) {
                nt = n1;
                if (nt < n2) {
                    nt = n2;
                }
                nt *= 8;
                if (nthreads > 1) {
                    nt *= nthreads;
                }
                if (n3 == 4) {
                    nt >>= 1;
                } else if (n3 < 4) {
                    nt >>= 2;
                }
                t = new double[nt];
                oldNthreads = nthreads;
            }
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, 1, a, scale);
                cdft3db_subth(1, a, scale);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, 1, a, scale);
                cdft3db_sub(1, a, scale);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    /* -------- child routines -------- */

    private void mixedRadixRealForwardFull(final double[][][] a) {
        double[] temp = new double[2 * n2];
        int ldimn2 = n2 / 2 + 1;
        final int newn3 = 2 * n3;
        final int n2d2;
        if (n2 % 2 == 0) {
            n2d2 = n2 / 2;
        } else {
            n2d2 = (n2 + 1) / 2;
        }

        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if ((nthreads > 1) && useThreads && (n1 >= nthreads) && (n3 >= nthreads) && (ldimn2 >= nthreads)) {
            Future[] futures = new Future[nthreads];
            int p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            for (int r = 0; r < n2; r++) {
                                fftn3.realForwardFull(a[s][r]);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n2];

                        for (int s = startSlice; s < stopSlice; s++) {
                            for (int c = 0; c < n3; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < n2; r++) {
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftn2.complexForward(temp);
                                for (int r = 0; r < n2; r++) {
                                    int idx4 = 2 * r;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startRow = l * p;
                final int stopRow;
                if (l == nthreads - 1) {
                    stopRow = ldimn2;
                } else {
                    stopRow = startRow + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n1];

                        for (int r = startRow; r < stopRow; r++) {
                            for (int c = 0; c < n3; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    temp[idx2] = a[s][r][idx1];
                                    temp[idx2 + 1] = a[s][r][idx1 + 1];
                                }
                                fftn1.complexForward(temp);
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    a[s][r][idx1] = temp[idx2];
                                    a[s][r][idx1 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {

                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx2 = (n1 - s) % n1;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = n2 - r;
                                for (int c = 0; c < n3; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = newn3 - idx1;
                                    a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                                    a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {

            for (int s = 0; s < n1; s++) {
                for (int r = 0; r < n2; r++) {
                    fftn3.realForwardFull(a[s][r]);
                }
            }

            for (int s = 0; s < n1; s++) {
                for (int c = 0; c < n3; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftn2.complexForward(temp);
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * n1];

            for (int r = 0; r < ldimn2; r++) {
                for (int c = 0; c < n3; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        temp[idx2] = a[s][r][idx1];
                        temp[idx2 + 1] = a[s][r][idx1 + 1];
                    }
                    fftn1.complexForward(temp);
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        a[s][r][idx1] = temp[idx2];
                        a[s][r][idx1 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < n1; s++) {
                int idx2 = (n1 - s) % n1;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = n2 - r;
                    for (int c = 0; c < n3; c++) {
                        int idx1 = 2 * c;
                        int idx3 = newn3 - idx1;
                        a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                        a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                    }
                }
            }

        }
    }

    private void mixedRadixRealInverseFull(final double[][][] a, final boolean scale) {
        double[] temp = new double[2 * n2];
        int ldimn2 = n2 / 2 + 1;
        final int newn3 = 2 * n3;
        final int n2d2;
        if (n2 % 2 == 0) {
            n2d2 = n2 / 2;
        } else {
            n2d2 = (n2 + 1) / 2;
        }

        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if ((nthreads > 1) && useThreads && (n1 >= nthreads) && (n3 >= nthreads) && (ldimn2 >= nthreads)) {
            Future[] futures = new Future[nthreads];
            int p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            for (int r = 0; r < n2; r++) {
                                fftn3.realInverseFull(a[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n2];

                        for (int s = startSlice; s < stopSlice; s++) {
                            for (int c = 0; c < n3; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < n2; r++) {
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftn2.complexInverse(temp, scale);
                                for (int r = 0; r < n2; r++) {
                                    int idx4 = 2 * r;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startRow = l * p;
                final int stopRow;
                if (l == nthreads - 1) {
                    stopRow = ldimn2;
                } else {
                    stopRow = startRow + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n1];

                        for (int r = startRow; r < stopRow; r++) {
                            for (int c = 0; c < n3; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    temp[idx2] = a[s][r][idx1];
                                    temp[idx2 + 1] = a[s][r][idx1 + 1];
                                }
                                fftn1.complexInverse(temp, scale);
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    a[s][r][idx1] = temp[idx2];
                                    a[s][r][idx1 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {

                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx2 = (n1 - s) % n1;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = n2 - r;
                                for (int c = 0; c < n3; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = newn3 - idx1;
                                    a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                                    a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {

            for (int s = 0; s < n1; s++) {
                for (int r = 0; r < n2; r++) {
                    fftn3.realInverseFull(a[s][r], scale);
                }
            }

            for (int s = 0; s < n1; s++) {
                for (int c = 0; c < n3; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftn2.complexInverse(temp, scale);
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * n1];

            for (int r = 0; r < ldimn2; r++) {
                for (int c = 0; c < n3; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        temp[idx2] = a[s][r][idx1];
                        temp[idx2 + 1] = a[s][r][idx1 + 1];
                    }
                    fftn1.complexInverse(temp, scale);
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        a[s][r][idx1] = temp[idx2];
                        a[s][r][idx1 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < n1; s++) {
                int idx2 = (n1 - s) % n1;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = n2 - r;
                    for (int c = 0; c < n3; c++) {
                        int idx1 = 2 * c;
                        int idx3 = newn3 - idx1;
                        a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                        a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                    }
                }
            }

        }
    }

    private void mixedRadixRealForwardFull(final double[] a) {
        final int twon3 = 2 * n3;
        double[] temp = new double[twon3];
        int ldimn2 = n2 / 2 + 1;
        final int n2d2;
        if (n2 % 2 == 0) {
            n2d2 = n2 / 2;
        } else {
            n2d2 = (n2 + 1) / 2;
        }

        final int twoSliceStride = 2 * sliceStride;
        final int twoRowStride = 2 * rowStride;

        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if ((nthreads > 1) && useThreads && (n1 / 2 >= nthreads) && (n3 >= nthreads) && (ldimn2 >= nthreads)) {
            Future[] futures = new Future[nthreads];
            int p = n1 / 2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = n1 - 1 - l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1 / 2;
                } else {
                    stopSlice = startSlice - p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[twon3];
                        for (int s = startSlice; s >= stopSlice; s--) {
                            int idx1 = s * sliceStride;
                            int idx2 = s * twoSliceStride;
                            for (int r = n2 - 1; r >= 0; r--) {
                                System.arraycopy(a, idx1 + r * rowStride, temp, 0, n3);
                                fftn3.realForwardFull(temp);
                                System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            final double[][][] temp2 = new double[n1 / 2][n2][twon3];

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1 / 2;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int r = 0; r < n2; r++) {
                                System.arraycopy(a, idx1 + r * rowStride, temp2[s][r], 0, n3);
                                fftn3.realForwardFull(temp2[s][r]);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1 / 2;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int r = 0; r < n2; r++) {
                                System.arraycopy(temp2[s][r], 0, a, idx1 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            //			for (int s = n1 / 2 - 1; s >= 0; s--) {
            //				int idx1 = s * sliceStride;
            //				int idx2 = s * twoSliceStride;
            //				for (int r = n2 - 1; r >= 0; r--) {
            //					System.arraycopy(a, idx1 + r * rowStride, temp, 0, n3);
            //					fftn3.realForwardFull(temp);
            //					System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
            //				}
            //			}

            p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n2];

                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int c = 0; c < n3; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < n2; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[idx3];
                                    temp[idx4 + 1] = a[idx3 + 1];
                                }
                                fftn2.complexForward(temp);
                                for (int r = 0; r < n2; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    a[idx3] = temp[idx4];
                                    a[idx3 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startRow = l * p;
                final int stopRow;
                if (l == nthreads - 1) {
                    stopRow = ldimn2;
                } else {
                    stopRow = startRow + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n1];

                        for (int r = startRow; r < stopRow; r++) {
                            int idx3 = r * twoRowStride;
                            for (int c = 0; c < n3; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    temp[idx2] = a[idx4];
                                    temp[idx2 + 1] = a[idx4 + 1];
                                }
                                fftn1.complexForward(temp);
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    a[idx4] = temp[idx2];
                                    a[idx4 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {

                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx2 = (n1 - s) % n1;
                            int idx5 = idx2 * twoSliceStride;
                            int idx6 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = n2 - r;
                                int idx7 = idx4 * twoRowStride;
                                int idx8 = r * twoRowStride;
                                int idx9 = idx5 + idx7;
                                for (int c = 0; c < n3; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = twon3 - idx1;
                                    int idx10 = idx6 + idx8 + idx1;
                                    a[idx9 + idx3 % twon3] = a[idx10];
                                    a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {

            for (int s = n1 - 1; s >= 0; s--) {
                int idx1 = s * sliceStride;
                int idx2 = s * twoSliceStride;
                for (int r = n2 - 1; r >= 0; r--) {
                    System.arraycopy(a, idx1 + r * rowStride, temp, 0, n3);
                    fftn3.realForwardFull(temp);
                    System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                }
            }

            temp = new double[2 * n2];

            for (int s = 0; s < n1; s++) {
                int idx1 = s * twoSliceStride;
                for (int c = 0; c < n3; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        temp[idx4] = a[idx3];
                        temp[idx4 + 1] = a[idx3 + 1];
                    }
                    fftn2.complexForward(temp);
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        a[idx3] = temp[idx4];
                        a[idx3 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * n1];

            for (int r = 0; r < ldimn2; r++) {
                int idx3 = r * twoRowStride;
                for (int c = 0; c < n3; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        temp[idx2] = a[idx4];
                        temp[idx2 + 1] = a[idx4 + 1];
                    }
                    fftn1.complexForward(temp);
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        a[idx4] = temp[idx2];
                        a[idx4 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < n1; s++) {
                int idx2 = (n1 - s) % n1;
                int idx5 = idx2 * twoSliceStride;
                int idx6 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = n2 - r;
                    int idx7 = idx4 * twoRowStride;
                    int idx8 = r * twoRowStride;
                    int idx9 = idx5 + idx7;
                    for (int c = 0; c < n3; c++) {
                        int idx1 = 2 * c;
                        int idx3 = twon3 - idx1;
                        int idx10 = idx6 + idx8 + idx1;
                        a[idx9 + idx3 % twon3] = a[idx10];
                        a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                    }
                }
            }

        }
    }

    private void mixedRadixRealInverseFull(final double[] a, final boolean scale) {
        final int twon3 = 2 * n3;
        double[] temp = new double[twon3];
        int ldimn2 = n2 / 2 + 1;
        final int n2d2;
        if (n2 % 2 == 0) {
            n2d2 = n2 / 2;
        } else {
            n2d2 = (n2 + 1) / 2;
        }

        final int twoSliceStride = 2 * sliceStride;
        final int twoRowStride = 2 * rowStride;

        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if ((nthreads > 1) && useThreads && (n1 / 2 >= nthreads) && (n3 >= nthreads) && (ldimn2 >= nthreads)) {
            Future[] futures = new Future[nthreads];
            int p = n1 / 2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = n1 - 1 - l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1 / 2;
                } else {
                    stopSlice = startSlice - p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[twon3];
                        for (int s = startSlice; s >= stopSlice; s--) {
                            int idx1 = s * sliceStride;
                            int idx2 = s * twoSliceStride;
                            for (int r = n2 - 1; r >= 0; r--) {
                                System.arraycopy(a, idx1 + r * rowStride, temp, 0, n3);
                                fftn3.realInverseFull(temp, scale);
                                System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            final double[][][] temp2 = new double[n1 / 2][n2][twon3];

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1 / 2;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int r = 0; r < n2; r++) {
                                System.arraycopy(a, idx1 + r * rowStride, temp2[s][r], 0, n3);
                                fftn3.realInverseFull(temp2[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1 / 2;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int r = 0; r < n2; r++) {
                                System.arraycopy(temp2[s][r], 0, a, idx1 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n2];

                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int c = 0; c < n3; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < n2; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[idx3];
                                    temp[idx4 + 1] = a[idx3 + 1];
                                }
                                fftn2.complexInverse(temp, scale);
                                for (int r = 0; r < n2; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    a[idx3] = temp[idx4];
                                    a[idx3 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startRow = l * p;
                final int stopRow;
                if (l == nthreads - 1) {
                    stopRow = ldimn2;
                } else {
                    stopRow = startRow + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        double[] temp = new double[2 * n1];

                        for (int r = startRow; r < stopRow; r++) {
                            int idx3 = r * twoRowStride;
                            for (int c = 0; c < n3; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    temp[idx2] = a[idx4];
                                    temp[idx2 + 1] = a[idx4 + 1];
                                }
                                fftn1.complexInverse(temp, scale);
                                for (int s = 0; s < n1; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    a[idx4] = temp[idx2];
                                    a[idx4 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {

                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx2 = (n1 - s) % n1;
                            int idx5 = idx2 * twoSliceStride;
                            int idx6 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = n2 - r;
                                int idx7 = idx4 * twoRowStride;
                                int idx8 = r * twoRowStride;
                                int idx9 = idx5 + idx7;
                                for (int c = 0; c < n3; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = twon3 - idx1;
                                    int idx10 = idx6 + idx8 + idx1;
                                    a[idx9 + idx3 % twon3] = a[idx10];
                                    a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {

            for (int s = n1 - 1; s >= 0; s--) {
                int idx1 = s * sliceStride;
                int idx2 = s * twoSliceStride;
                for (int r = n2 - 1; r >= 0; r--) {
                    System.arraycopy(a, idx1 + r * rowStride, temp, 0, n3);
                    fftn3.realInverseFull(temp, scale);
                    System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                }
            }

            temp = new double[2 * n2];

            for (int s = 0; s < n1; s++) {
                int idx1 = s * twoSliceStride;
                for (int c = 0; c < n3; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        temp[idx4] = a[idx3];
                        temp[idx4 + 1] = a[idx3 + 1];
                    }
                    fftn2.complexInverse(temp, scale);
                    for (int r = 0; r < n2; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        a[idx3] = temp[idx4];
                        a[idx3 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * n1];

            for (int r = 0; r < ldimn2; r++) {
                int idx3 = r * twoRowStride;
                for (int c = 0; c < n3; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        temp[idx2] = a[idx4];
                        temp[idx2 + 1] = a[idx4 + 1];
                    }
                    fftn1.complexInverse(temp, scale);
                    for (int s = 0; s < n1; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        a[idx4] = temp[idx2];
                        a[idx4 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < n1; s++) {
                int idx2 = (n1 - s) % n1;
                int idx5 = idx2 * twoSliceStride;
                int idx6 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = n2 - r;
                    int idx7 = idx4 * twoRowStride;
                    int idx8 = r * twoRowStride;
                    int idx9 = idx5 + idx7;
                    for (int c = 0; c < n3; c++) {
                        int idx1 = 2 * c;
                        int idx3 = twon3 - idx1;
                        int idx10 = idx6 + idx8 + idx1;
                        a[idx9 + idx3 % twon3] = a[idx10];
                        a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                    }
                }
            }

        }
    }

    private void xdft3da_sub1(int icr, int isgn, double[] a, boolean scale) {
        int i, j, k, idx0, idx1, idx2, idx3, idx4, idx5;

        if (isgn == -1) {
            for (i = 0; i < n1; i++) {
                idx0 = i * sliceStride;
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexForward(a, idx0 + j * rowStride);
                    }
                } else {
                    for (j = 0; j < n2; j++) {
                        fftn3.realInverse(a, idx0 + j * rowStride, scale);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftn2.complexForward(t, 0);
                        fftn2.complexForward(t, 2 * n2);
                        fftn2.complexForward(t, 4 * n2);
                        fftn2.complexForward(t, 6 * n2);
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftn2.complexForward(t, 0);
                    fftn2.complexForward(t, 2 * n2);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftn2.complexForward(t, 0);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (i = 0; i < n1; i++) {
                idx0 = i * sliceStride;
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexInverse(a, idx0 + j * rowStride, scale);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftn2.complexInverse(t, 0, scale);
                        fftn2.complexInverse(t, 2 * n2, scale);
                        fftn2.complexInverse(t, 4 * n2, scale);
                        fftn2.complexInverse(t, 6 * n2, scale);
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    fftn2.complexInverse(t, 2 * n2, scale);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
                if (icr != 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.realForward(a, idx0 + j * rowStride);
                    }
                }
            }
        }
    }

    private void xdft3da_sub2(int icr, int isgn, double[] a, boolean scale) {
        int i, j, k, idx0, idx1, idx2, idx3, idx4, idx5;

        if (isgn == -1) {
            for (i = 0; i < n1; i++) {
                idx0 = i * sliceStride;
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexForward(a, idx0 + j * rowStride);
                    }
                } else {
                    for (j = 0; j < n2; j++) {
                        fftn3.realForward(a, idx0 + j * rowStride);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftn2.complexForward(t, 0);
                        fftn2.complexForward(t, 2 * n2);
                        fftn2.complexForward(t, 4 * n2);
                        fftn2.complexForward(t, 6 * n2);
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftn2.complexForward(t, 0);
                    fftn2.complexForward(t, 2 * n2);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftn2.complexForward(t, 0);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (i = 0; i < n1; i++) {
                idx0 = i * sliceStride;
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexInverse(a, idx0 + j * rowStride, scale);
                    }
                } else {
                    for (j = 0; j < n2; j++) {
                        fftn3.realInverse2(a, idx0 + j * rowStride, scale);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftn2.complexInverse(t, 0, scale);
                        fftn2.complexInverse(t, 2 * n2, scale);
                        fftn2.complexInverse(t, 4 * n2, scale);
                        fftn2.complexInverse(t, 6 * n2, scale);
                        for (j = 0; j < n2; j++) {
                            idx1 = idx0 + j * rowStride + k;
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    fftn2.complexInverse(t, 2 * n2, scale);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    for (j = 0; j < n2; j++) {
                        idx1 = idx0 + j * rowStride;
                        idx2 = 2 * j;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        }
    }

    private void xdft3da_sub1(int icr, int isgn, double[][][] a, boolean scale) {
        int i, j, k, idx2, idx3, idx4, idx5;

        if (isgn == -1) {
            for (i = 0; i < n1; i++) {
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexForward(a[i][j]);
                    }
                } else {
                    for (j = 0; j < n2; j++) {
                        fftn3.realInverse(a[i][j], 0, scale);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[i][j][k];
                            t[idx2 + 1] = a[i][j][k + 1];
                            t[idx3] = a[i][j][k + 2];
                            t[idx3 + 1] = a[i][j][k + 3];
                            t[idx4] = a[i][j][k + 4];
                            t[idx4 + 1] = a[i][j][k + 5];
                            t[idx5] = a[i][j][k + 6];
                            t[idx5 + 1] = a[i][j][k + 7];
                        }
                        fftn2.complexForward(t, 0);
                        fftn2.complexForward(t, 2 * n2);
                        fftn2.complexForward(t, 4 * n2);
                        fftn2.complexForward(t, 6 * n2);
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[i][j][k] = t[idx2];
                            a[i][j][k + 1] = t[idx2 + 1];
                            a[i][j][k + 2] = t[idx3];
                            a[i][j][k + 3] = t[idx3 + 1];
                            a[i][j][k + 4] = t[idx4];
                            a[i][j][k + 5] = t[idx4 + 1];
                            a[i][j][k + 6] = t[idx5];
                            a[i][j][k + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                        t[idx3] = a[i][j][2];
                        t[idx3 + 1] = a[i][j][3];
                    }
                    fftn2.complexForward(t, 0);
                    fftn2.complexForward(t, 2 * n2);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                        a[i][j][2] = t[idx3];
                        a[i][j][3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                    }
                    fftn2.complexForward(t, 0);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (i = 0; i < n1; i++) {
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexInverse(a[i][j], scale);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[i][j][k];
                            t[idx2 + 1] = a[i][j][k + 1];
                            t[idx3] = a[i][j][k + 2];
                            t[idx3 + 1] = a[i][j][k + 3];
                            t[idx4] = a[i][j][k + 4];
                            t[idx4 + 1] = a[i][j][k + 5];
                            t[idx5] = a[i][j][k + 6];
                            t[idx5 + 1] = a[i][j][k + 7];
                        }
                        fftn2.complexInverse(t, 0, scale);
                        fftn2.complexInverse(t, 2 * n2, scale);
                        fftn2.complexInverse(t, 4 * n2, scale);
                        fftn2.complexInverse(t, 6 * n2, scale);
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[i][j][k] = t[idx2];
                            a[i][j][k + 1] = t[idx2 + 1];
                            a[i][j][k + 2] = t[idx3];
                            a[i][j][k + 3] = t[idx3 + 1];
                            a[i][j][k + 4] = t[idx4];
                            a[i][j][k + 5] = t[idx4 + 1];
                            a[i][j][k + 6] = t[idx5];
                            a[i][j][k + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                        t[idx3] = a[i][j][2];
                        t[idx3 + 1] = a[i][j][3];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    fftn2.complexInverse(t, 2 * n2, scale);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                        a[i][j][2] = t[idx3];
                        a[i][j][3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                    }
                }
                if (icr != 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.realForward(a[i][j], 0);
                    }
                }
            }
        }
    }

    private void xdft3da_sub2(int icr, int isgn, double[][][] a, boolean scale) {
        int i, j, k, idx2, idx3, idx4, idx5;

        if (isgn == -1) {
            for (i = 0; i < n1; i++) {
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexForward(a[i][j]);
                    }
                } else {
                    for (j = 0; j < n2; j++) {
                        fftn3.realForward(a[i][j]);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[i][j][k];
                            t[idx2 + 1] = a[i][j][k + 1];
                            t[idx3] = a[i][j][k + 2];
                            t[idx3 + 1] = a[i][j][k + 3];
                            t[idx4] = a[i][j][k + 4];
                            t[idx4 + 1] = a[i][j][k + 5];
                            t[idx5] = a[i][j][k + 6];
                            t[idx5 + 1] = a[i][j][k + 7];
                        }
                        fftn2.complexForward(t, 0);
                        fftn2.complexForward(t, 2 * n2);
                        fftn2.complexForward(t, 4 * n2);
                        fftn2.complexForward(t, 6 * n2);
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[i][j][k] = t[idx2];
                            a[i][j][k + 1] = t[idx2 + 1];
                            a[i][j][k + 2] = t[idx3];
                            a[i][j][k + 3] = t[idx3 + 1];
                            a[i][j][k + 4] = t[idx4];
                            a[i][j][k + 5] = t[idx4 + 1];
                            a[i][j][k + 6] = t[idx5];
                            a[i][j][k + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                        t[idx3] = a[i][j][2];
                        t[idx3 + 1] = a[i][j][3];
                    }
                    fftn2.complexForward(t, 0);
                    fftn2.complexForward(t, 2 * n2);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                        a[i][j][2] = t[idx3];
                        a[i][j][3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                    }
                    fftn2.complexForward(t, 0);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (i = 0; i < n1; i++) {
                if (icr == 0) {
                    for (j = 0; j < n2; j++) {
                        fftn3.complexInverse(a[i][j], scale);
                    }
                } else {
                    for (j = 0; j < n2; j++) {
                        fftn3.realInverse2(a[i][j], 0, scale);
                    }
                }
                if (n3 > 4) {
                    for (k = 0; k < n3; k += 8) {
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            t[idx2] = a[i][j][k];
                            t[idx2 + 1] = a[i][j][k + 1];
                            t[idx3] = a[i][j][k + 2];
                            t[idx3 + 1] = a[i][j][k + 3];
                            t[idx4] = a[i][j][k + 4];
                            t[idx4 + 1] = a[i][j][k + 5];
                            t[idx5] = a[i][j][k + 6];
                            t[idx5 + 1] = a[i][j][k + 7];
                        }
                        fftn2.complexInverse(t, 0, scale);
                        fftn2.complexInverse(t, 2 * n2, scale);
                        fftn2.complexInverse(t, 4 * n2, scale);
                        fftn2.complexInverse(t, 6 * n2, scale);
                        for (j = 0; j < n2; j++) {
                            idx2 = 2 * j;
                            idx3 = 2 * n2 + 2 * j;
                            idx4 = idx3 + 2 * n2;
                            idx5 = idx4 + 2 * n2;
                            a[i][j][k] = t[idx2];
                            a[i][j][k + 1] = t[idx2 + 1];
                            a[i][j][k + 2] = t[idx3];
                            a[i][j][k + 3] = t[idx3 + 1];
                            a[i][j][k + 4] = t[idx4];
                            a[i][j][k + 5] = t[idx4 + 1];
                            a[i][j][k + 6] = t[idx5];
                            a[i][j][k + 7] = t[idx5 + 1];
                        }
                    }
                } else if (n3 == 4) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                        t[idx3] = a[i][j][2];
                        t[idx3 + 1] = a[i][j][3];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    fftn2.complexInverse(t, 2 * n2, scale);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        idx3 = 2 * n2 + 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                        a[i][j][2] = t[idx3];
                        a[i][j][3] = t[idx3 + 1];
                    }
                } else if (n3 == 2) {
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                    }
                    fftn2.complexInverse(t, 0, scale);
                    for (j = 0; j < n2; j++) {
                        idx2 = 2 * j;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                    }
                }
            }
        }
    }

    private void cdft3db_sub(int isgn, double[] a, boolean scale) {
        int i, j, k, idx0, idx1, idx2, idx3, idx4, idx5;

        if (isgn == -1) {
            if (n3 > 4) {
                for (j = 0; j < n2; j++) {
                    idx0 = j * rowStride;
                    for (k = 0; k < n3; k += 8) {
                        for (i = 0; i < n1; i++) {
                            idx1 = i * sliceStride + idx0 + k;
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftn1.complexForward(t, 0);
                        fftn1.complexForward(t, 2 * n1);
                        fftn1.complexForward(t, 4 * n1);
                        fftn1.complexForward(t, 6 * n1);
                        for (i = 0; i < n1; i++) {
                            idx1 = i * sliceStride + idx0 + k;
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                }
            } else if (n3 == 4) {
                for (j = 0; j < n2; j++) {
                    idx0 = j * rowStride;
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftn1.complexForward(t, 0);
                    fftn1.complexForward(t, 2 * n1);
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                }
            } else if (n3 == 2) {
                for (j = 0; j < n2; j++) {
                    idx0 = j * rowStride;
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftn1.complexForward(t, 0);
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            if (n3 > 4) {
                for (j = 0; j < n2; j++) {
                    idx0 = j * rowStride;
                    for (k = 0; k < n3; k += 8) {
                        for (i = 0; i < n1; i++) {
                            idx1 = i * sliceStride + idx0 + k;
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftn1.complexInverse(t, 0, scale);
                        fftn1.complexInverse(t, 2 * n1, scale);
                        fftn1.complexInverse(t, 4 * n1, scale);
                        fftn1.complexInverse(t, 6 * n1, scale);
                        for (i = 0; i < n1; i++) {
                            idx1 = i * sliceStride + idx0 + k;
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                }
            } else if (n3 == 4) {
                for (j = 0; j < n2; j++) {
                    idx0 = j * rowStride;
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftn1.complexInverse(t, 0, scale);
                    fftn1.complexInverse(t, 2 * n1, scale);
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                }
            } else if (n3 == 2) {
                for (j = 0; j < n2; j++) {
                    idx0 = j * rowStride;
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftn1.complexInverse(t, 0, scale);
                    for (i = 0; i < n1; i++) {
                        idx1 = i * sliceStride + idx0;
                        idx2 = 2 * i;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        }
    }

    private void cdft3db_sub(int isgn, double[][][] a, boolean scale) {
        int i, j, k, idx2, idx3, idx4, idx5;

        if (isgn == -1) {
            if (n3 > 4) {
                for (j = 0; j < n2; j++) {
                    for (k = 0; k < n3; k += 8) {
                        for (i = 0; i < n1; i++) {
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            t[idx2] = a[i][j][k];
                            t[idx2 + 1] = a[i][j][k + 1];
                            t[idx3] = a[i][j][k + 2];
                            t[idx3 + 1] = a[i][j][k + 3];
                            t[idx4] = a[i][j][k + 4];
                            t[idx4 + 1] = a[i][j][k + 5];
                            t[idx5] = a[i][j][k + 6];
                            t[idx5 + 1] = a[i][j][k + 7];
                        }
                        fftn1.complexForward(t, 0);
                        fftn1.complexForward(t, 2 * n1);
                        fftn1.complexForward(t, 4 * n1);
                        fftn1.complexForward(t, 6 * n1);
                        for (i = 0; i < n1; i++) {
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            a[i][j][k] = t[idx2];
                            a[i][j][k + 1] = t[idx2 + 1];
                            a[i][j][k + 2] = t[idx3];
                            a[i][j][k + 3] = t[idx3 + 1];
                            a[i][j][k + 4] = t[idx4];
                            a[i][j][k + 5] = t[idx4 + 1];
                            a[i][j][k + 6] = t[idx5];
                            a[i][j][k + 7] = t[idx5 + 1];
                        }
                    }
                }
            } else if (n3 == 4) {
                for (j = 0; j < n2; j++) {
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                        t[idx3] = a[i][j][2];
                        t[idx3 + 1] = a[i][j][3];
                    }
                    fftn1.complexForward(t, 0);
                    fftn1.complexForward(t, 2 * n1);
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                        a[i][j][2] = t[idx3];
                        a[i][j][3] = t[idx3 + 1];
                    }
                }
            } else if (n3 == 2) {
                for (j = 0; j < n2; j++) {
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                    }
                    fftn1.complexForward(t, 0);
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            if (n3 > 4) {
                for (j = 0; j < n2; j++) {
                    for (k = 0; k < n3; k += 8) {
                        for (i = 0; i < n1; i++) {
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            t[idx2] = a[i][j][k];
                            t[idx2 + 1] = a[i][j][k + 1];
                            t[idx3] = a[i][j][k + 2];
                            t[idx3 + 1] = a[i][j][k + 3];
                            t[idx4] = a[i][j][k + 4];
                            t[idx4 + 1] = a[i][j][k + 5];
                            t[idx5] = a[i][j][k + 6];
                            t[idx5 + 1] = a[i][j][k + 7];
                        }
                        fftn1.complexInverse(t, 0, scale);
                        fftn1.complexInverse(t, 2 * n1, scale);
                        fftn1.complexInverse(t, 4 * n1, scale);
                        fftn1.complexInverse(t, 6 * n1, scale);
                        for (i = 0; i < n1; i++) {
                            idx2 = 2 * i;
                            idx3 = 2 * n1 + 2 * i;
                            idx4 = idx3 + 2 * n1;
                            idx5 = idx4 + 2 * n1;
                            a[i][j][k] = t[idx2];
                            a[i][j][k + 1] = t[idx2 + 1];
                            a[i][j][k + 2] = t[idx3];
                            a[i][j][k + 3] = t[idx3 + 1];
                            a[i][j][k + 4] = t[idx4];
                            a[i][j][k + 5] = t[idx4 + 1];
                            a[i][j][k + 6] = t[idx5];
                            a[i][j][k + 7] = t[idx5 + 1];
                        }
                    }
                }
            } else if (n3 == 4) {
                for (j = 0; j < n2; j++) {
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                        t[idx3] = a[i][j][2];
                        t[idx3 + 1] = a[i][j][3];
                    }
                    fftn1.complexInverse(t, 0, scale);
                    fftn1.complexInverse(t, 2 * n1, scale);
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                        a[i][j][2] = t[idx3];
                        a[i][j][3] = t[idx3 + 1];
                    }
                }
            } else if (n3 == 2) {
                for (j = 0; j < n2; j++) {
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        t[idx2] = a[i][j][0];
                        t[idx2 + 1] = a[i][j][1];
                    }
                    fftn1.complexInverse(t, 0, scale);
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        a[i][j][0] = t[idx2];
                        a[i][j][1] = t[idx2 + 1];
                    }
                }
            }
        }
    }

    private void xdft3da_subth1(final int icr, final int isgn, final double[] a, final boolean scale) {
        int nthread;
        int nt, i;

        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread > n1) {
            nthread = n1;
        }
        nt = 8 * n2;
        if (n3 == 4) {
            nt >>= 1;
        } else if (n3 < 4) {
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j, k, idx0, idx1, idx2, idx3, idx4, idx5;

                    if (isgn == -1) {
                        for (i = n0; i < n1; i += nthread_f) {
                            idx0 = i * sliceStride;
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexForward(a, idx0 + j * rowStride);
                                }
                            } else {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realInverse(a, idx0 + j * rowStride, scale);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftn2.complexForward(t, startt);
                                    fftn2.complexForward(t, startt + 2 * n2);
                                    fftn2.complexForward(t, startt + 4 * n2);
                                    fftn2.complexForward(t, startt + 6 * n2);
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftn2.complexForward(t, startt);
                                fftn2.complexForward(t, startt + 2 * n2);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftn2.complexForward(t, startt);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (i = n0; i < n1; i += nthread_f) {
                            idx0 = i * sliceStride;
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexInverse(a, idx0 + j * rowStride, scale);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftn2.complexInverse(t, startt, scale);
                                    fftn2.complexInverse(t, startt + 2 * n2, scale);
                                    fftn2.complexInverse(t, startt + 4 * n2, scale);
                                    fftn2.complexInverse(t, startt + 6 * n2, scale);
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                fftn2.complexInverse(t, startt + 2 * n2, scale);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }
                            if (icr != 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realForward(a, idx0 + j * rowStride);
                                }
                            }
                        }
                    }
                }
            });
        }
        try {
            for (int j = 0; j < nthread; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void xdft3da_subth2(final int icr, final int isgn, final double[] a, final boolean scale) {
        int nthread;
        int nt, i;

        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread > n1) {
            nthread = n1;
        }
        nt = 8 * n2;
        if (n3 == 4) {
            nt >>= 1;
        } else if (n3 < 4) {
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j, k, idx0, idx1, idx2, idx3, idx4, idx5;

                    if (isgn == -1) {
                        for (i = n0; i < n1; i += nthread_f) {
                            idx0 = i * sliceStride;
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexForward(a, idx0 + j * rowStride);
                                }
                            } else {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realForward(a, idx0 + j * rowStride);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftn2.complexForward(t, startt);
                                    fftn2.complexForward(t, startt + 2 * n2);
                                    fftn2.complexForward(t, startt + 4 * n2);
                                    fftn2.complexForward(t, startt + 6 * n2);
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftn2.complexForward(t, startt);
                                fftn2.complexForward(t, startt + 2 * n2);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftn2.complexForward(t, startt);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (i = n0; i < n1; i += nthread_f) {
                            idx0 = i * sliceStride;
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexInverse(a, idx0 + j * rowStride, scale);
                                }
                            } else {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realInverse2(a, idx0 + j * rowStride, scale);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftn2.complexInverse(t, startt, scale);
                                    fftn2.complexInverse(t, startt + 2 * n2, scale);
                                    fftn2.complexInverse(t, startt + 4 * n2, scale);
                                    fftn2.complexInverse(t, startt + 6 * n2, scale);
                                    for (j = 0; j < n2; j++) {
                                        idx1 = idx0 + j * rowStride + k;
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                fftn2.complexInverse(t, startt + 2 * n2, scale);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                for (j = 0; j < n2; j++) {
                                    idx1 = idx0 + j * rowStride;
                                    idx2 = startt + 2 * j;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }
                        }
                    }
                }
            });
        }
        try {
            for (int j = 0; j < nthread; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void xdft3da_subth1(final int icr, final int isgn, final double[][][] a, final boolean scale) {
        int nthread;
        int nt, i;

        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread > n1) {
            nthread = n1;
        }
        nt = 8 * n2;
        if (n3 == 4) {
            nt >>= 1;
        } else if (n3 < 4) {
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j, k, idx2, idx3, idx4, idx5;

                    if (isgn == -1) {
                        for (i = n0; i < n1; i += nthread_f) {
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexForward(a[i][j]);
                                }
                            } else {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realInverse(a[i][j], 0, scale);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[i][j][k];
                                        t[idx2 + 1] = a[i][j][k + 1];
                                        t[idx3] = a[i][j][k + 2];
                                        t[idx3 + 1] = a[i][j][k + 3];
                                        t[idx4] = a[i][j][k + 4];
                                        t[idx4 + 1] = a[i][j][k + 5];
                                        t[idx5] = a[i][j][k + 6];
                                        t[idx5 + 1] = a[i][j][k + 7];
                                    }
                                    fftn2.complexForward(t, startt);
                                    fftn2.complexForward(t, startt + 2 * n2);
                                    fftn2.complexForward(t, startt + 4 * n2);
                                    fftn2.complexForward(t, startt + 6 * n2);
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[i][j][k] = t[idx2];
                                        a[i][j][k + 1] = t[idx2 + 1];
                                        a[i][j][k + 2] = t[idx3];
                                        a[i][j][k + 3] = t[idx3 + 1];
                                        a[i][j][k + 4] = t[idx4];
                                        a[i][j][k + 5] = t[idx4 + 1];
                                        a[i][j][k + 6] = t[idx5];
                                        a[i][j][k + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                    t[idx3] = a[i][j][2];
                                    t[idx3 + 1] = a[i][j][3];
                                }
                                fftn2.complexForward(t, startt);
                                fftn2.complexForward(t, startt + 2 * n2);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                    a[i][j][2] = t[idx3];
                                    a[i][j][3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                }
                                fftn2.complexForward(t, startt);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (i = n0; i < n1; i += nthread_f) {
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexInverse(a[i][j], scale);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[i][j][k];
                                        t[idx2 + 1] = a[i][j][k + 1];
                                        t[idx3] = a[i][j][k + 2];
                                        t[idx3 + 1] = a[i][j][k + 3];
                                        t[idx4] = a[i][j][k + 4];
                                        t[idx4 + 1] = a[i][j][k + 5];
                                        t[idx5] = a[i][j][k + 6];
                                        t[idx5 + 1] = a[i][j][k + 7];
                                    }
                                    fftn2.complexInverse(t, startt, scale);
                                    fftn2.complexInverse(t, startt + 2 * n2, scale);
                                    fftn2.complexInverse(t, startt + 4 * n2, scale);
                                    fftn2.complexInverse(t, startt + 6 * n2, scale);
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[i][j][k] = t[idx2];
                                        a[i][j][k + 1] = t[idx2 + 1];
                                        a[i][j][k + 2] = t[idx3];
                                        a[i][j][k + 3] = t[idx3 + 1];
                                        a[i][j][k + 4] = t[idx4];
                                        a[i][j][k + 5] = t[idx4 + 1];
                                        a[i][j][k + 6] = t[idx5];
                                        a[i][j][k + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                    t[idx3] = a[i][j][2];
                                    t[idx3 + 1] = a[i][j][3];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                fftn2.complexInverse(t, startt + 2 * n2, scale);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                    a[i][j][2] = t[idx3];
                                    a[i][j][3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                }
                            }
                            if (icr != 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realForward(a[i][j]);
                                }
                            }
                        }
                    }
                }
            });
        }
        try {
            for (int j = 0; j < nthread; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void xdft3da_subth2(final int icr, final int isgn, final double[][][] a, final boolean scale) {
        int nthread;
        int nt, i;

        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread > n1) {
            nthread = n1;
        }
        nt = 8 * n2;
        if (n3 == 4) {
            nt >>= 1;
        } else if (n3 < 4) {
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j, k, idx2, idx3, idx4, idx5;

                    if (isgn == -1) {
                        for (i = n0; i < n1; i += nthread_f) {
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexForward(a[i][j]);
                                }
                            } else {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realForward(a[i][j]);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[i][j][k];
                                        t[idx2 + 1] = a[i][j][k + 1];
                                        t[idx3] = a[i][j][k + 2];
                                        t[idx3 + 1] = a[i][j][k + 3];
                                        t[idx4] = a[i][j][k + 4];
                                        t[idx4 + 1] = a[i][j][k + 5];
                                        t[idx5] = a[i][j][k + 6];
                                        t[idx5 + 1] = a[i][j][k + 7];
                                    }
                                    fftn2.complexForward(t, startt);
                                    fftn2.complexForward(t, startt + 2 * n2);
                                    fftn2.complexForward(t, startt + 4 * n2);
                                    fftn2.complexForward(t, startt + 6 * n2);
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[i][j][k] = t[idx2];
                                        a[i][j][k + 1] = t[idx2 + 1];
                                        a[i][j][k + 2] = t[idx3];
                                        a[i][j][k + 3] = t[idx3 + 1];
                                        a[i][j][k + 4] = t[idx4];
                                        a[i][j][k + 5] = t[idx4 + 1];
                                        a[i][j][k + 6] = t[idx5];
                                        a[i][j][k + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                    t[idx3] = a[i][j][2];
                                    t[idx3 + 1] = a[i][j][3];
                                }
                                fftn2.complexForward(t, startt);
                                fftn2.complexForward(t, startt + 2 * n2);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                    a[i][j][2] = t[idx3];
                                    a[i][j][3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                }
                                fftn2.complexForward(t, startt);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (i = n0; i < n1; i += nthread_f) {
                            if (icr == 0) {
                                for (j = 0; j < n2; j++) {
                                    fftn3.complexInverse(a[i][j], scale);
                                }
                            } else {
                                for (j = 0; j < n2; j++) {
                                    fftn3.realInverse2(a[i][j], 0, scale);
                                }
                            }
                            if (n3 > 4) {
                                for (k = 0; k < n3; k += 8) {
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        t[idx2] = a[i][j][k];
                                        t[idx2 + 1] = a[i][j][k + 1];
                                        t[idx3] = a[i][j][k + 2];
                                        t[idx3 + 1] = a[i][j][k + 3];
                                        t[idx4] = a[i][j][k + 4];
                                        t[idx4 + 1] = a[i][j][k + 5];
                                        t[idx5] = a[i][j][k + 6];
                                        t[idx5 + 1] = a[i][j][k + 7];
                                    }
                                    fftn2.complexInverse(t, startt, scale);
                                    fftn2.complexInverse(t, startt + 2 * n2, scale);
                                    fftn2.complexInverse(t, startt + 4 * n2, scale);
                                    fftn2.complexInverse(t, startt + 6 * n2, scale);
                                    for (j = 0; j < n2; j++) {
                                        idx2 = startt + 2 * j;
                                        idx3 = startt + 2 * n2 + 2 * j;
                                        idx4 = idx3 + 2 * n2;
                                        idx5 = idx4 + 2 * n2;
                                        a[i][j][k] = t[idx2];
                                        a[i][j][k + 1] = t[idx2 + 1];
                                        a[i][j][k + 2] = t[idx3];
                                        a[i][j][k + 3] = t[idx3 + 1];
                                        a[i][j][k + 4] = t[idx4];
                                        a[i][j][k + 5] = t[idx4 + 1];
                                        a[i][j][k + 6] = t[idx5];
                                        a[i][j][k + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (n3 == 4) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                    t[idx3] = a[i][j][2];
                                    t[idx3 + 1] = a[i][j][3];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                fftn2.complexInverse(t, startt + 2 * n2, scale);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    idx3 = startt + 2 * n2 + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                    a[i][j][2] = t[idx3];
                                    a[i][j][3] = t[idx3 + 1];
                                }
                            } else if (n3 == 2) {
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                }
                                fftn2.complexInverse(t, startt, scale);
                                for (j = 0; j < n2; j++) {
                                    idx2 = startt + 2 * j;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                }
                            }
                        }
                    }
                }
            });
        }
        try {
            for (int j = 0; j < nthread; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void cdft3db_subth(final int isgn, final double[] a, final boolean scale) {
        int nthread;
        int nt, i;

        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread > n2) {
            nthread = n2;
        }
        nt = 8 * n1;
        if (n3 == 4) {
            nt >>= 1;
        } else if (n3 < 4) {
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                public void run() {
                    int i, j, k, idx0, idx1, idx2, idx3, idx4, idx5;

                    if (isgn == -1) {
                        if (n3 > 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                idx0 = j * rowStride;
                                for (k = 0; k < n3; k += 8) {
                                    for (i = 0; i < n1; i++) {
                                        idx1 = i * sliceStride + idx0 + k;
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftn1.complexForward(t, startt);
                                    fftn1.complexForward(t, startt + 2 * n1);
                                    fftn1.complexForward(t, startt + 4 * n1);
                                    fftn1.complexForward(t, startt + 6 * n1);
                                    for (i = 0; i < n1; i++) {
                                        idx1 = i * sliceStride + idx0 + k;
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            }
                        } else if (n3 == 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                idx0 = j * rowStride;
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftn1.complexForward(t, startt);
                                fftn1.complexForward(t, startt + 2 * n1);
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            }
                        } else if (n3 == 2) {
                            for (j = n0; j < n2; j += nthread_f) {
                                idx0 = j * rowStride;
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftn1.complexForward(t, startt);
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }
                        }
                    } else {
                        if (n3 > 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                idx0 = j * rowStride;
                                for (k = 0; k < n3; k += 8) {
                                    for (i = 0; i < n1; i++) {
                                        idx1 = i * sliceStride + idx0 + k;
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftn1.complexInverse(t, startt, scale);
                                    fftn1.complexInverse(t, startt + 2 * n1, scale);
                                    fftn1.complexInverse(t, startt + 4 * n1, scale);
                                    fftn1.complexInverse(t, startt + 6 * n1, scale);
                                    for (i = 0; i < n1; i++) {
                                        idx1 = i * sliceStride + idx0 + k;
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            }
                        } else if (n3 == 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                idx0 = j * rowStride;
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftn1.complexInverse(t, startt, scale);
                                fftn1.complexInverse(t, startt + 2 * n1, scale);
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            }
                        } else if (n3 == 2) {
                            for (j = n0; j < n2; j += nthread_f) {
                                idx0 = j * rowStride;
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftn1.complexInverse(t, startt, scale);
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * sliceStride + idx0;
                                    idx2 = startt + 2 * i;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }
                        }
                    }

                }
            });
        }
        try {
            for (int j = 0; j < nthread; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void cdft3db_subth(final int isgn, final double[][][] a, final boolean scale) {
        int nthread;
        int nt, i;

        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread > n2) {
            nthread = n2;
        }
        nt = 8 * n1;
        if (n3 == 4) {
            nt >>= 1;
        } else if (n3 < 4) {
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                public void run() {
                    int i, j, k, idx2, idx3, idx4, idx5;

                    if (isgn == -1) {
                        if (n3 > 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                for (k = 0; k < n3; k += 8) {
                                    for (i = 0; i < n1; i++) {
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        t[idx2] = a[i][j][k];
                                        t[idx2 + 1] = a[i][j][k + 1];
                                        t[idx3] = a[i][j][k + 2];
                                        t[idx3 + 1] = a[i][j][k + 3];
                                        t[idx4] = a[i][j][k + 4];
                                        t[idx4 + 1] = a[i][j][k + 5];
                                        t[idx5] = a[i][j][k + 6];
                                        t[idx5 + 1] = a[i][j][k + 7];
                                    }
                                    fftn1.complexForward(t, startt);
                                    fftn1.complexForward(t, startt + 2 * n1);
                                    fftn1.complexForward(t, startt + 4 * n1);
                                    fftn1.complexForward(t, startt + 6 * n1);
                                    for (i = 0; i < n1; i++) {
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        a[i][j][k] = t[idx2];
                                        a[i][j][k + 1] = t[idx2 + 1];
                                        a[i][j][k + 2] = t[idx3];
                                        a[i][j][k + 3] = t[idx3 + 1];
                                        a[i][j][k + 4] = t[idx4];
                                        a[i][j][k + 5] = t[idx4 + 1];
                                        a[i][j][k + 6] = t[idx5];
                                        a[i][j][k + 7] = t[idx5 + 1];
                                    }
                                }
                            }
                        } else if (n3 == 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                    t[idx3] = a[i][j][2];
                                    t[idx3 + 1] = a[i][j][3];
                                }
                                fftn1.complexForward(t, startt);
                                fftn1.complexForward(t, startt + 2 * n1);
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                    a[i][j][2] = t[idx3];
                                    a[i][j][3] = t[idx3 + 1];
                                }
                            }
                        } else if (n3 == 2) {
                            for (j = n0; j < n2; j += nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                }
                                fftn1.complexForward(t, startt);
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                }
                            }
                        }
                    } else {
                        if (n3 > 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                for (k = 0; k < n3; k += 8) {
                                    for (i = 0; i < n1; i++) {
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        t[idx2] = a[i][j][k];
                                        t[idx2 + 1] = a[i][j][k + 1];
                                        t[idx3] = a[i][j][k + 2];
                                        t[idx3 + 1] = a[i][j][k + 3];
                                        t[idx4] = a[i][j][k + 4];
                                        t[idx4 + 1] = a[i][j][k + 5];
                                        t[idx5] = a[i][j][k + 6];
                                        t[idx5 + 1] = a[i][j][k + 7];
                                    }
                                    fftn1.complexInverse(t, startt, scale);
                                    fftn1.complexInverse(t, startt + 2 * n1, scale);
                                    fftn1.complexInverse(t, startt + 4 * n1, scale);
                                    fftn1.complexInverse(t, startt + 6 * n1, scale);
                                    for (i = 0; i < n1; i++) {
                                        idx2 = startt + 2 * i;
                                        idx3 = startt + 2 * n1 + 2 * i;
                                        idx4 = idx3 + 2 * n1;
                                        idx5 = idx4 + 2 * n1;
                                        a[i][j][k] = t[idx2];
                                        a[i][j][k + 1] = t[idx2 + 1];
                                        a[i][j][k + 2] = t[idx3];
                                        a[i][j][k + 3] = t[idx3 + 1];
                                        a[i][j][k + 4] = t[idx4];
                                        a[i][j][k + 5] = t[idx4 + 1];
                                        a[i][j][k + 6] = t[idx5];
                                        a[i][j][k + 7] = t[idx5 + 1];
                                    }
                                }
                            }
                        } else if (n3 == 4) {
                            for (j = n0; j < n2; j += nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                    t[idx3] = a[i][j][2];
                                    t[idx3 + 1] = a[i][j][3];
                                }
                                fftn1.complexInverse(t, startt, scale);
                                fftn1.complexInverse(t, startt + 2 * n1, scale);
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                    a[i][j][2] = t[idx3];
                                    a[i][j][3] = t[idx3 + 1];
                                }
                            }
                        } else if (n3 == 2) {
                            for (j = n0; j < n2; j += nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    t[idx2] = a[i][j][0];
                                    t[idx2 + 1] = a[i][j][1];
                                }
                                fftn1.complexInverse(t, startt, scale);
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    a[i][j][0] = t[idx2];
                                    a[i][j][1] = t[idx2 + 1];
                                }
                            }
                        }
                    }

                }
            });
        }
        try {
            for (int j = 0; j < nthread; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void rdft3d_sub(int isgn, double[] a) {
        int n1h, n2h, i, j, k, l, idx1, idx2, idx3, idx4;
        double xi;

        n1h = n1 >> 1;
        n2h = n2 >> 1;
        if (isgn < 0) {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                idx1 = i * sliceStride;
                idx2 = j * sliceStride;
                idx3 = i * sliceStride + n2h * rowStride;
                idx4 = j * sliceStride + n2h * rowStride;
                xi = a[idx1] - a[idx2];
                a[idx1] += a[idx2];
                a[idx2] = xi;
                xi = a[idx2 + 1] - a[idx1 + 1];
                a[idx1 + 1] += a[idx2 + 1];
                a[idx2 + 1] = xi;
                xi = a[idx3] - a[idx4];
                a[idx3] += a[idx4];
                a[idx4] = xi;
                xi = a[idx4 + 1] - a[idx3 + 1];
                a[idx3 + 1] += a[idx4 + 1];
                a[idx4 + 1] = xi;
                for (k = 1; k < n2h; k++) {
                    l = n2 - k;
                    idx1 = i * sliceStride + k * rowStride;
                    idx2 = j * sliceStride + l * rowStride;
                    xi = a[idx1] - a[idx2];
                    a[idx1] += a[idx2];
                    a[idx2] = xi;
                    xi = a[idx2 + 1] - a[idx1 + 1];
                    a[idx1 + 1] += a[idx2 + 1];
                    a[idx2 + 1] = xi;
                    idx3 = j * sliceStride + k * rowStride;
                    idx4 = i * sliceStride + l * rowStride;
                    xi = a[idx3] - a[idx4];
                    a[idx3] += a[idx4];
                    a[idx4] = xi;
                    xi = a[idx4 + 1] - a[idx3 + 1];
                    a[idx3 + 1] += a[idx4 + 1];
                    a[idx4 + 1] = xi;
                }
            }
            for (k = 1; k < n2h; k++) {
                l = n2 - k;
                idx1 = k * rowStride;
                idx2 = l * rowStride;
                xi = a[idx1] - a[idx2];
                a[idx1] += a[idx2];
                a[idx2] = xi;
                xi = a[idx2 + 1] - a[idx1 + 1];
                a[idx1 + 1] += a[idx2 + 1];
                a[idx2 + 1] = xi;
                idx3 = n1h * sliceStride + k * rowStride;
                idx4 = n1h * sliceStride + l * rowStride;
                xi = a[idx3] - a[idx4];
                a[idx3] += a[idx4];
                a[idx4] = xi;
                xi = a[idx4 + 1] - a[idx3 + 1];
                a[idx3 + 1] += a[idx4 + 1];
                a[idx4 + 1] = xi;
            }
        } else {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                idx1 = j * sliceStride;
                idx2 = i * sliceStride;
                a[idx1] = 0.5f * (a[idx2] - a[idx1]);
                a[idx2] -= a[idx1];
                a[idx1 + 1] = 0.5f * (a[idx2 + 1] + a[idx1 + 1]);
                a[idx2 + 1] -= a[idx1 + 1];
                idx3 = j * sliceStride + n2h * rowStride;
                idx4 = i * sliceStride + n2h * rowStride;
                a[idx3] = 0.5f * (a[idx4] - a[idx3]);
                a[idx4] -= a[idx3];
                a[idx3 + 1] = 0.5f * (a[idx4 + 1] + a[idx3 + 1]);
                a[idx4 + 1] -= a[idx3 + 1];
                for (k = 1; k < n2h; k++) {
                    l = n2 - k;
                    idx1 = j * sliceStride + l * rowStride;
                    idx2 = i * sliceStride + k * rowStride;
                    a[idx1] = 0.5f * (a[idx2] - a[idx1]);
                    a[idx2] -= a[idx1];
                    a[idx1 + 1] = 0.5f * (a[idx2 + 1] + a[idx1 + 1]);
                    a[idx2 + 1] -= a[idx1 + 1];
                    idx3 = i * sliceStride + l * rowStride;
                    idx4 = j * sliceStride + k * rowStride;
                    a[idx3] = 0.5f * (a[idx4] - a[idx3]);
                    a[idx4] -= a[idx3];
                    a[idx3 + 1] = 0.5f * (a[idx4 + 1] + a[idx3 + 1]);
                    a[idx4 + 1] -= a[idx3 + 1];
                }
            }
            for (k = 1; k < n2h; k++) {
                l = n2 - k;
                idx1 = l * rowStride;
                idx2 = k * rowStride;
                a[idx1] = 0.5f * (a[idx2] - a[idx1]);
                a[idx2] -= a[idx1];
                a[idx1 + 1] = 0.5f * (a[idx2 + 1] + a[idx1 + 1]);
                a[idx2 + 1] -= a[idx1 + 1];
                idx3 = n1h * sliceStride + l * rowStride;
                idx4 = n1h * sliceStride + k * rowStride;
                a[idx3] = 0.5f * (a[idx4] - a[idx3]);
                a[idx4] -= a[idx3];
                a[idx3 + 1] = 0.5f * (a[idx4 + 1] + a[idx3 + 1]);
                a[idx4 + 1] -= a[idx3 + 1];
            }
        }
    }

    private void rdft3d_sub(int isgn, double[][][] a) {
        int n1h, n2h, i, j, k, l;
        double xi;

        n1h = n1 >> 1;
        n2h = n2 >> 1;
        if (isgn < 0) {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                xi = a[i][0][0] - a[j][0][0];
                a[i][0][0] += a[j][0][0];
                a[j][0][0] = xi;
                xi = a[j][0][1] - a[i][0][1];
                a[i][0][1] += a[j][0][1];
                a[j][0][1] = xi;
                xi = a[i][n2h][0] - a[j][n2h][0];
                a[i][n2h][0] += a[j][n2h][0];
                a[j][n2h][0] = xi;
                xi = a[j][n2h][1] - a[i][n2h][1];
                a[i][n2h][1] += a[j][n2h][1];
                a[j][n2h][1] = xi;
                for (k = 1; k < n2h; k++) {
                    l = n2 - k;
                    xi = a[i][k][0] - a[j][l][0];
                    a[i][k][0] += a[j][l][0];
                    a[j][l][0] = xi;
                    xi = a[j][l][1] - a[i][k][1];
                    a[i][k][1] += a[j][l][1];
                    a[j][l][1] = xi;
                    xi = a[j][k][0] - a[i][l][0];
                    a[j][k][0] += a[i][l][0];
                    a[i][l][0] = xi;
                    xi = a[i][l][1] - a[j][k][1];
                    a[j][k][1] += a[i][l][1];
                    a[i][l][1] = xi;
                }
            }
            for (k = 1; k < n2h; k++) {
                l = n2 - k;
                xi = a[0][k][0] - a[0][l][0];
                a[0][k][0] += a[0][l][0];
                a[0][l][0] = xi;
                xi = a[0][l][1] - a[0][k][1];
                a[0][k][1] += a[0][l][1];
                a[0][l][1] = xi;
                xi = a[n1h][k][0] - a[n1h][l][0];
                a[n1h][k][0] += a[n1h][l][0];
                a[n1h][l][0] = xi;
                xi = a[n1h][l][1] - a[n1h][k][1];
                a[n1h][k][1] += a[n1h][l][1];
                a[n1h][l][1] = xi;
            }
        } else {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                a[j][0][0] = 0.5f * (a[i][0][0] - a[j][0][0]);
                a[i][0][0] -= a[j][0][0];
                a[j][0][1] = 0.5f * (a[i][0][1] + a[j][0][1]);
                a[i][0][1] -= a[j][0][1];
                a[j][n2h][0] = 0.5f * (a[i][n2h][0] - a[j][n2h][0]);
                a[i][n2h][0] -= a[j][n2h][0];
                a[j][n2h][1] = 0.5f * (a[i][n2h][1] + a[j][n2h][1]);
                a[i][n2h][1] -= a[j][n2h][1];
                for (k = 1; k < n2h; k++) {
                    l = n2 - k;
                    a[j][l][0] = 0.5f * (a[i][k][0] - a[j][l][0]);
                    a[i][k][0] -= a[j][l][0];
                    a[j][l][1] = 0.5f * (a[i][k][1] + a[j][l][1]);
                    a[i][k][1] -= a[j][l][1];
                    a[i][l][0] = 0.5f * (a[j][k][0] - a[i][l][0]);
                    a[j][k][0] -= a[i][l][0];
                    a[i][l][1] = 0.5f * (a[j][k][1] + a[i][l][1]);
                    a[j][k][1] -= a[i][l][1];
                }
            }
            for (k = 1; k < n2h; k++) {
                l = n2 - k;
                a[0][l][0] = 0.5f * (a[0][k][0] - a[0][l][0]);
                a[0][k][0] -= a[0][l][0];
                a[0][l][1] = 0.5f * (a[0][k][1] + a[0][l][1]);
                a[0][k][1] -= a[0][l][1];
                a[n1h][l][0] = 0.5f * (a[n1h][k][0] - a[n1h][l][0]);
                a[n1h][k][0] -= a[n1h][l][0];
                a[n1h][l][1] = 0.5f * (a[n1h][k][1] + a[n1h][l][1]);
                a[n1h][k][1] -= a[n1h][l][1];
            }
        }
    }

    private void fillSymmetric(final double[][][] a) {
        final int twon3 = 2 * n3;
        final int n2d2 = n2 / 2;
        int n1d2 = n1 / 2;
        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if ((nthreads > 1) && useThreads && (n1 >= nthreads)) {
            Future[] futures = new Future[nthreads];
            int p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = (n1 - s) % n1;
                            for (int r = 0; r < n2; r++) {
                                int idx2 = (n2 - r) % n2;
                                for (int c = 1; c < n3; c += 2) {
                                    int idx3 = twon3 - c;
                                    a[idx1][idx2][idx3] = -a[s][r][c + 2];
                                    a[idx1][idx2][idx3 - 1] = a[s][r][c + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            // ---------------------------------------------

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = (n1 - s) % n1;
                            for (int r = 1; r < n2d2; r++) {
                                int idx2 = n2 - r;
                                a[idx1][r][n3] = a[s][idx2][1];
                                a[s][idx2][n3] = a[s][idx2][1];
                                a[idx1][r][n3 + 1] = -a[s][idx2][0];
                                a[s][idx2][n3 + 1] = a[s][idx2][0];
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx1 = (n1 - s) % n1;
                            for (int r = 1; r < n2d2; r++) {
                                int idx2 = n2 - r;
                                a[idx1][idx2][0] = a[s][r][0];
                                a[idx1][idx2][1] = -a[s][r][1];
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        } else {

            for (int s = 0; s < n1; s++) {
                int idx1 = (n1 - s) % n1;
                for (int r = 0; r < n2; r++) {
                    int idx2 = (n2 - r) % n2;
                    for (int c = 1; c < n3; c += 2) {
                        int idx3 = twon3 - c;
                        a[idx1][idx2][idx3] = -a[s][r][c + 2];
                        a[idx1][idx2][idx3 - 1] = a[s][r][c + 1];
                    }
                }
            }

            // ---------------------------------------------

            for (int s = 0; s < n1; s++) {
                int idx1 = (n1 - s) % n1;
                for (int r = 1; r < n2d2; r++) {
                    int idx2 = n2 - r;
                    a[idx1][r][n3] = a[s][idx2][1];
                    a[s][idx2][n3] = a[s][idx2][1];
                    a[idx1][r][n3 + 1] = -a[s][idx2][0];
                    a[s][idx2][n3 + 1] = a[s][idx2][0];
                }
            }

            for (int s = 0; s < n1; s++) {
                int idx1 = (n1 - s) % n1;
                for (int r = 1; r < n2d2; r++) {
                    int idx2 = n2 - r;
                    a[idx1][idx2][0] = a[s][r][0];
                    a[idx1][idx2][1] = -a[s][r][1];
                }
            }
        }

        // ----------------------------------------------------------

        for (int s = 1; s < n1d2; s++) {
            int idx1 = n1 - s;
            a[s][0][n3] = a[idx1][0][1];
            a[idx1][0][n3] = a[idx1][0][1];
            a[s][0][n3 + 1] = -a[idx1][0][0];
            a[idx1][0][n3 + 1] = a[idx1][0][0];
            a[s][n2d2][n3] = a[idx1][n2d2][1];
            a[idx1][n2d2][n3] = a[idx1][n2d2][1];
            a[s][n2d2][n3 + 1] = -a[idx1][n2d2][0];
            a[idx1][n2d2][n3 + 1] = a[idx1][n2d2][0];
            a[idx1][0][0] = a[s][0][0];
            a[idx1][0][1] = -a[s][0][1];
            a[idx1][n2d2][0] = a[s][n2d2][0];
            a[idx1][n2d2][1] = -a[s][n2d2][1];

        }
        // ----------------------------------------

        a[0][0][n3] = a[0][0][1];
        a[0][0][1] = 0;
        a[0][n2d2][n3] = a[0][n2d2][1];
        a[0][n2d2][1] = 0;
        a[n1d2][0][n3] = a[n1d2][0][1];
        a[n1d2][0][1] = 0;
        a[n1d2][n2d2][n3] = a[n1d2][n2d2][1];
        a[n1d2][n2d2][1] = 0;
        a[n1d2][0][n3 + 1] = 0;
        a[n1d2][n2d2][n3 + 1] = 0;
    }

    private void fillSymmetric(final double[] a) {
        final int twon3 = 2 * n3;
        final int n2d2 = n2 / 2;
        int n1d2 = n1 / 2;

        final int twoSliceStride = n2 * twon3;
        final int twoRowStride = twon3;

        int idx1, idx2, idx3, idx4, idx5, idx6;

        for (int s = (n1 - 1); s >= 1; s--) {
            idx3 = s * sliceStride;
            idx4 = 2 * idx3;
            for (int r = 0; r < n2; r++) {
                idx5 = r * rowStride;
                idx6 = 2 * idx5;
                for (int c = 0; c < n3; c += 2) {
                    idx1 = idx3 + idx5 + c;
                    idx2 = idx4 + idx6 + c;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                    idx1++;
                    idx2++;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                }
            }
        }

        for (int r = 1; r < n2; r++) {
            idx3 = (n2 - r) * rowStride;
            idx4 = (n2 - r) * twoRowStride;
            for (int c = 0; c < n3; c += 2) {
                idx1 = idx3 + c;
                idx2 = idx4 + c;
                a[idx2] = a[idx1];
                a[idx1] = 0;
                idx1++;
                idx2++;
                a[idx2] = a[idx1];
                a[idx1] = 0;
            }
        }

        int nthreads = ConcurrencyUtils.getNumberOfProcessors();
        if ((nthreads > 1) && useThreads && (n1 >= nthreads)) {
            Future[] futures = new Future[nthreads];
            int p = n1 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx3 = ((n1 - s) % n1) * twoSliceStride;
                            int idx5 = s * twoSliceStride;
                            for (int r = 0; r < n2; r++) {
                                int idx4 = ((n2 - r) % n2) * twoRowStride;
                                int idx6 = r * twoRowStride;
                                for (int c = 1; c < n3; c += 2) {
                                    int idx1 = idx3 + idx4 + twon3 - c;
                                    int idx2 = idx5 + idx6 + c;
                                    a[idx1] = -a[idx2 + 2];
                                    a[idx1 - 1] = a[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            // ---------------------------------------------

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx5 = ((n1 - s) % n1) * twoSliceStride;
                            int idx6 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = idx6 + (n2 - r) * twoRowStride;
                                int idx1 = idx5 + r * twoRowStride + n3;
                                int idx2 = idx4 + n3;
                                int idx3 = idx4 + 1;
                                a[idx1] = a[idx3];
                                a[idx2] = a[idx3];
                                a[idx1 + 1] = -a[idx4];
                                a[idx2 + 1] = a[idx4];

                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for (int l = 0; l < nthreads; l++) {
                final int startSlice = l * p;
                final int stopSlice;
                if (l == nthreads - 1) {
                    stopSlice = n1;
                } else {
                    stopSlice = startSlice + p;
                }
                futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        for (int s = startSlice; s < stopSlice; s++) {
                            int idx3 = ((n1 - s) % n1) * twoSliceStride;
                            int idx4 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx1 = idx3 + (n2 - r) * twoRowStride;
                                int idx2 = idx4 + r * twoRowStride;
                                a[idx1] = a[idx2];
                                a[idx1 + 1] = -a[idx2 + 1];

                            }
                        }
                    }
                });
            }
            try {
                for (int l = 0; l < nthreads; l++) {
                    futures[l].get();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {

            // -----------------------------------------------
            for (int s = 0; s < n1; s++) {
                idx3 = ((n1 - s) % n1) * twoSliceStride;
                idx5 = s * twoSliceStride;
                for (int r = 0; r < n2; r++) {
                    idx4 = ((n2 - r) % n2) * twoRowStride;
                    idx6 = r * twoRowStride;
                    for (int c = 1; c < n3; c += 2) {
                        idx1 = idx3 + idx4 + twon3 - c;
                        idx2 = idx5 + idx6 + c;
                        a[idx1] = -a[idx2 + 2];
                        a[idx1 - 1] = a[idx2 + 1];
                    }
                }
            }

            // ---------------------------------------------

            for (int s = 0; s < n1; s++) {
                idx5 = ((n1 - s) % n1) * twoSliceStride;
                idx6 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    idx4 = idx6 + (n2 - r) * twoRowStride;
                    idx1 = idx5 + r * twoRowStride + n3;
                    idx2 = idx4 + n3;
                    idx3 = idx4 + 1;
                    a[idx1] = a[idx3];
                    a[idx2] = a[idx3];
                    a[idx1 + 1] = -a[idx4];
                    a[idx2 + 1] = a[idx4];

                }
            }

            for (int s = 0; s < n1; s++) {
                idx3 = ((n1 - s) % n1) * twoSliceStride;
                idx4 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    idx1 = idx3 + (n2 - r) * twoRowStride;
                    idx2 = idx4 + r * twoRowStride;
                    a[idx1] = a[idx2];
                    a[idx1 + 1] = -a[idx2 + 1];

                }
            }
        }

        // ----------------------------------------------------------

        for (int s = 1; s < n1d2; s++) {
            idx1 = s * twoSliceStride;
            idx2 = (n1 - s) * twoSliceStride;
            idx3 = n2d2 * twoRowStride;
            idx4 = idx1 + idx3;
            idx5 = idx2 + idx3;
            a[idx1 + n3] = a[idx2 + 1];
            a[idx2 + n3] = a[idx2 + 1];
            a[idx1 + n3 + 1] = -a[idx2];
            a[idx2 + n3 + 1] = a[idx2];
            a[idx4 + n3] = a[idx5 + 1];
            a[idx5 + n3] = a[idx5 + 1];
            a[idx4 + n3 + 1] = -a[idx5];
            a[idx5 + n3 + 1] = a[idx5];
            a[idx2] = a[idx1];
            a[idx2 + 1] = -a[idx1 + 1];
            a[idx5] = a[idx4];
            a[idx5 + 1] = -a[idx4 + 1];

        }

        // ----------------------------------------

        a[n3] = a[1];
        a[1] = 0;
        idx1 = n2d2 * twoRowStride;
        idx2 = n1d2 * twoSliceStride;
        idx3 = idx1 + idx2;
        a[idx1 + n3] = a[idx1 + 1];
        a[idx1 + 1] = 0;
        a[idx2 + n3] = a[idx2 + 1];
        a[idx2 + 1] = 0;
        a[idx3 + n3] = a[idx3 + 1];
        a[idx3 + 1] = 0;
        a[idx2 + n3 + 1] = 0;
        a[idx3 + n3 + 1] = 0;
    }
}
