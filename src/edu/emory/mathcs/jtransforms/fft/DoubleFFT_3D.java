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
 * precision data. The sizes of all three dimensions must be power-of-two
 * numbers. This is a parallel implementation of split-radix algorithm optimized
 * for SMP systems. <br>
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

    private int[] ip;

    private double[] w;

    private double[] t;

    private DoubleFFT_1D fftn1, fftn2, fftn3;

    private int oldNthreads;

    private int nt;

    /**
     * Creates new instance of DoubleFFT_3D.
     * 
     * @param n1
     *            number of slices - must be a power-of-two number
     * @param n2
     *            number of rows - must be a power-of-two number
     * @param n3
     *            number of columns - must be a power-of-two number
     * 
     */
    public DoubleFFT_3D(int n1, int n2, int n3) {
        if (!ConcurrencyUtils.isPowerOf2(n1) || !ConcurrencyUtils.isPowerOf2(n2) || !ConcurrencyUtils.isPowerOf2(n3))
            throw new IllegalArgumentException("n1, n2 and n3 must be power of two numbers");
        if (n1 <= 1 || n2 <= 1 || n3 <= 1) {
            throw new IllegalArgumentException("n1, n2 and n3 must be greater than 1");
        }
        this.n1 = n1;
        this.n2 = n2;
        this.n3 = n3;
        this.sliceStride = n2 * n3;
        this.rowStride = n3;
        ip = new int[2 + (int) Math.ceil(Math.sqrt(Math.max(Math.max(n1, n2), n3)))];
        w = new double[(int) Math.ceil(Math.max(Math.max(Math.max(n1 / 2, n2 / 2), n3 / 2), Math.max(Math.max(n1 / 2, n2 / 2), n3 / 4) + n3 / 4))];
        fftn1 = new DoubleFFT_1D(n1, ip, w);
        fftn2 = new DoubleFFT_1D(n2, ip, w);
        fftn3 = new DoubleFFT_1D(n3, ip, w);
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
    public void complexForward(double[] a) {
        int n;
        int oldn3 = n3;
        n3 = 2 * n3;

        sliceStride = n2 * n3;
        rowStride = n3;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(0, -1, a, true);
            cdft3db_subth(-1, a, true);
        } else {
            xdft3da_sub2(0, -1, a, true);
            cdft3db_sub(-1, a, true);
        }
        n3 = oldn3;
        sliceStride = n2 * n3;
        rowStride = n3;
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
    public void complexForward(double[][][] a) {
        int n;
        int oldn3 = n3;
        n3 = 2 * n3;

        sliceStride = n2 * n3;
        rowStride = n3;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(0, -1, a, true);
            cdft3db_subth(-1, a, true);
        } else {
            xdft3da_sub2(0, -1, a, true);
            cdft3db_sub(-1, a, true);
        }
        n3 = oldn3;
        sliceStride = n2 * n3;
        rowStride = n3;
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
    public void complexInverse(double[] a, boolean scale) {
        int n;
        int oldn3 = n3;
        n3 = 2 * n3;
        sliceStride = n2 * n3;
        rowStride = n3;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(0, 1, a, scale);
            cdft3db_subth(1, a, scale);
        } else {
            xdft3da_sub2(0, 1, a, scale);
            cdft3db_sub(1, a, scale);
        }
        n3 = oldn3;
        sliceStride = n2 * n3;
        rowStride = n3;
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
    public void complexInverse(double[][][] a, boolean scale) {
        int n;
        int oldn3 = n3;
        n3 = 2 * n3;
        sliceStride = n2 * n3;
        rowStride = n3;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(0, 1, a, scale);
            cdft3db_subth(1, a, scale);
        } else {
            xdft3da_sub2(0, 1, a, scale);
            cdft3db_sub(1, a, scale);
        }
        n3 = oldn3;
        sliceStride = n2 * n3;
        rowStride = n3;
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . The data is stored in a 1D array addressed in slice-major, then
     * row-major, then column-major, in order of significance, i.e. element
     * (i,j,k) of 3-d array x[n1][n2][2*n3] is stored in a[i*sliceStride +
     * j*rowStride + k], where sliceStride = n2 * 2 * n3 and rowStride = 2 * n3.
     * The physical layout of the output data is as follows:
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
        int n, nw, nc;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth1(1, -1, a, true);
            cdft3db_subth(-1, a, true);
            rdft3d_sub(1, a);
        } else {
            xdft3da_sub1(1, -1, a, true);
            cdft3db_sub(-1, a, true);
            rdft3d_sub(1, a);
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . The data is stored in a 3D array. The physical layout of the output
     * data is as follows:
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
        int n, nw, nc;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth1(1, -1, a, true);
            cdft3db_subth(-1, a, true);
            rdft3d_sub(1, a);
        } else {
            xdft3da_sub1(1, -1, a, true);
            cdft3db_sub(-1, a, true);
            rdft3d_sub(1, a);
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real forward transform, i.e. you will get
     * the same result as from <code>complexForward</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1*n2*2*n3, with only the first n1*n2*n3
     * elements filled with real data. To get back the original data, use
     * <code>complexInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForwardFull(double[] a) {
        int n, nw, nc, k1, k2, k3, newn3;
        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(1, -1, a, true);
            cdft3db_subth(-1, a, true);
            rdft3d_sub(1, a);
        } else {
            xdft3da_sub2(1, -1, a, true);
            cdft3db_sub(-1, a, true);
            rdft3d_sub(1, a);
        }
        newn3 = 2 * n3;
        int n2d2 = n2 / 2;
        int n1d2 = n1 / 2;

        int newSliceStride = n2 * newn3;
        int newRowStride = newn3;
        int idx1, idx2, idx3, idx4, idx5;

        for (k1 = (n1 - 1); k1 >= 1; k1--) {
            for (k2 = 0; k2 < n2; k2++) {
                for (k3 = 0; k3 < n3; k3 += 2) {
                    idx1 = k1 * sliceStride + k2 * rowStride + k3;
                    idx2 = k1 * newSliceStride + k2 * newRowStride + k3;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                    idx1++;
                    idx2++;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                }
            }
        }

        for (k2 = 1; k2 < n2; k2++) {
            for (k3 = 0; k3 < n3; k3 += 2) {
                idx1 = (n2 - k2) * rowStride + k3;
                idx2 = (n2 - k2) * newRowStride + k3;
                a[idx2] = a[idx1];
                a[idx1] = 0;
                idx1++;
                idx2++;
                a[idx2] = a[idx1];
                a[idx1] = 0;
            }
        }

        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            fillSymmetric(a);
        } else {

            // -----------------------------------------------
            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 0; k2 < n2; k2++) {
                    for (k3 = 1; k3 < n3; k3 += 2) {
                        idx1 = ((n1 - k1) % n1) * newSliceStride + ((n2 - k2) % n2) * newRowStride + newn3 - k3;
                        idx2 = k1 * newSliceStride + k2 * newRowStride + k3;
                        a[idx1] = -a[idx2 + 2];
                        a[idx1 - 1] = a[idx2 + 1];
                    }
                }
            }

            // ---------------------------------------------

            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 1; k2 < n2d2; k2++) {
                    idx1 = ((n1 - k1) % n1) * newSliceStride + k2 * newRowStride + n3;
                    idx2 = k1 * newSliceStride + (n2 - k2) * newRowStride + n3;
                    idx3 = k1 * newSliceStride + (n2 - k2) * newRowStride + 1;
                    idx4 = k1 * newSliceStride + (n2 - k2) * newRowStride;
                    a[idx1] = a[idx3];
                    a[idx2] = a[idx3];
                    a[idx1 + 1] = -a[idx4];
                    a[idx2 + 1] = a[idx4];

                }
            }
        }
        for (k1 = 0; k1 < n1; k1++) {
            for (k2 = 1; k2 < n2d2; k2++) {
                idx1 = ((n1 - k1) % n1) * newSliceStride + (n2 - k2) * newRowStride;
                idx2 = k1 * newSliceStride + k2 * newRowStride;
                a[idx1] = a[idx2];
                a[idx1 + 1] = -a[idx2 + 1];

            }
        }

        // ----------------------------------------------------------

        for (k1 = 1; k1 < n1d2; k1++) {
            idx1 = k1 * newSliceStride;
            idx2 = (n1 - k1) * newSliceStride;
            idx4 = k1 * newSliceStride + n2d2 * newRowStride;
            idx5 = (n1 - k1) * newSliceStride + n2d2 * newRowStride;
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
        // }
        // ----------------------------------------

        a[n3] = a[1];
        a[1] = 0;
        idx1 = n2d2 * newRowStride;
        idx2 = n1d2 * newSliceStride;
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

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real forward transform, i.e. you will get
     * the same result as from <code>complexForward</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1 by n2 by 2*n3, with only the first n1
     * by n2 by n3 elements filled with real data. To get back the original
     * data, use <code>complexInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForwardFull(double[][][] a) {
        int n, nw, nc, k1, k2, k3, newn3;
        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(1, -1, a, true);
            cdft3db_subth(-1, a, true);
            rdft3d_sub(1, a);
        } else {
            xdft3da_sub2(1, -1, a, true);
            cdft3db_sub(-1, a, true);
            rdft3d_sub(1, a);
        }
        newn3 = 2 * n3;
        int n2d2 = n2 / 2;
        int n1d2 = n1 / 2;

        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            fillSymmetric(a);
        } else {

            // -----------------------------------------------
            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 0; k2 < n2; k2++) {
                    for (k3 = 1; k3 < n3; k3 += 2) {
                        a[(n1 - k1) % n1][(n2 - k2) % n2][newn3 - k3] = -a[k1][k2][k3 + 2];
                        a[(n1 - k1) % n1][(n2 - k2) % n2][newn3 - k3 - 1] = a[k1][k2][k3 + 1];
                    }
                }
            }

            // ---------------------------------------------

            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 1; k2 < n2d2; k2++) {

                    a[(n1 - k1) % n1][k2][n3] = a[k1][n2 - k2][1];
                    a[k1][n2 - k2][n3] = a[k1][n2 - k2][1];
                    a[(n1 - k1) % n1][k2][n3 + 1] = -a[k1][n2 - k2][0];
                    a[k1][n2 - k2][n3 + 1] = a[k1][n2 - k2][0];

                }
            }
        }
        for (k1 = 0; k1 < n1; k1++) {
            for (k2 = 1; k2 < n2d2; k2++) {

                a[(n1 - k1) % n1][n2 - k2][0] = a[k1][k2][0];
                a[(n1 - k1) % n1][n2 - k2][1] = -a[k1][k2][1];

            }
        }

        // ----------------------------------------------------------

        for (k1 = 1; k1 < n1d2; k1++) {

            a[k1][0][n3] = a[n1 - k1][0][1];
            a[n1 - k1][0][n3] = a[n1 - k1][0][1];
            a[k1][0][n3 + 1] = -a[n1 - k1][0][0];
            a[n1 - k1][0][n3 + 1] = a[n1 - k1][0][0];
            a[k1][n2d2][n3] = a[n1 - k1][n2d2][1];
            a[n1 - k1][n2d2][n3] = a[n1 - k1][n2d2][1];
            a[k1][n2d2][n3 + 1] = -a[n1 - k1][n2d2][0];
            a[n1 - k1][n2d2][n3 + 1] = a[n1 - k1][n2d2][0];
            a[n1 - k1][0][0] = a[k1][0][0];
            a[n1 - k1][0][1] = -a[k1][0][1];
            a[n1 - k1][n2d2][0] = a[k1][n2d2][0];
            a[n1 - k1][n2d2][1] = -a[k1][n2d2][1];

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

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . The data is stored in a 1D array addressed in slice-major, then
     * row-major, then column-major, in order of significance, i.e. element
     * (i,j,k) of 3-d array x[n1][n2][2*n3] is stored in a[i*sliceStride +
     * j*rowStride + k], where sliceStride = n2 * 2 * n3 and rowStride = 2 * n3.
     * The physical layout of the input data has to be as follows:
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
        int n, nw, nc;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            rdft3d_sub(-1, a);
            cdft3db_subth(1, a, scale);
            xdft3da_subth1(1, 1, a, scale);
        } else {
            rdft3d_sub(-1, a);
            cdft3db_sub(1, a, scale);
            xdft3da_sub1(1, 1, a, scale);
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . The data is stored in a 3D array. The physical layout of the input data
     * has to be as follows:
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
        int n, nw, nc;

        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            rdft3d_sub(-1, a);
            cdft3db_subth(1, a, scale);
            xdft3da_subth1(1, 1, a, scale);
        } else {
            rdft3d_sub(-1, a);
            cdft3db_sub(1, a, scale);
            xdft3da_sub1(1, 1, a, scale);
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real inverse transform, i.e. you will get
     * the same result as from <code>complexInverse</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1*n2*2*n3, with only the first n1*n2*n3
     * elements filled with real data.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverseFull(double[] a, boolean scale) {
        int n, nw, nc, k1, k2, k3, newn3;
        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(1, 1, a, scale);
            cdft3db_subth(1, a, scale);
            rdft3d_sub(1, a);
        } else {
            xdft3da_sub2(1, 1, a, scale);
            cdft3db_sub(1, a, scale);
            rdft3d_sub(1, a);
        }
        newn3 = 2 * n3;
        int n2d2 = n2 / 2;
        int n1d2 = n1 / 2;

        int newSliceStride = n2 * newn3;
        int newRowStride = newn3;
        int idx1, idx2, idx3, idx4, idx5;

        for (k1 = (n1 - 1); k1 >= 1; k1--) {
            for (k2 = 0; k2 < n2; k2++) {
                for (k3 = 0; k3 < n3; k3 += 2) {
                    idx1 = k1 * sliceStride + k2 * rowStride + k3;
                    idx2 = k1 * newSliceStride + k2 * newRowStride + k3;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                    idx1++;
                    idx2++;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                }
            }
        }

        for (k2 = 1; k2 < n2; k2++) {
            for (k3 = 0; k3 < n3; k3 += 2) {
                idx1 = (n2 - k2) * rowStride + k3;
                idx2 = (n2 - k2) * newRowStride + k3;
                a[idx2] = a[idx1];
                a[idx1] = 0;
                idx1++;
                idx2++;
                a[idx2] = a[idx1];
                a[idx1] = 0;
            }
        }

        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            fillSymmetric(a);
        } else {

            // -----------------------------------------------
            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 0; k2 < n2; k2++) {
                    for (k3 = 1; k3 < n3; k3 += 2) {
                        idx1 = ((n1 - k1) % n1) * newSliceStride + ((n2 - k2) % n2) * newRowStride + newn3 - k3;
                        idx2 = k1 * newSliceStride + k2 * newRowStride + k3;
                        a[idx1] = -a[idx2 + 2];
                        a[idx1 - 1] = a[idx2 + 1];
                    }
                }
            }

            // ---------------------------------------------

            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 1; k2 < n2d2; k2++) {
                    idx1 = ((n1 - k1) % n1) * newSliceStride + k2 * newRowStride + n3;
                    idx2 = k1 * newSliceStride + (n2 - k2) * newRowStride + n3;
                    idx3 = k1 * newSliceStride + (n2 - k2) * newRowStride + 1;
                    idx4 = k1 * newSliceStride + (n2 - k2) * newRowStride;
                    a[idx1] = a[idx3];
                    a[idx2] = a[idx3];
                    a[idx1 + 1] = -a[idx4];
                    a[idx2 + 1] = a[idx4];

                }
            }
        }
        for (k1 = 0; k1 < n1; k1++) {
            for (k2 = 1; k2 < n2d2; k2++) {
                idx1 = ((n1 - k1) % n1) * newSliceStride + (n2 - k2) * newRowStride;
                idx2 = k1 * newSliceStride + k2 * newRowStride;
                a[idx1] = a[idx2];
                a[idx1 + 1] = -a[idx2 + 1];

            }
        }

        // ----------------------------------------------------------

        for (k1 = 1; k1 < n1d2; k1++) {
            idx1 = k1 * newSliceStride;
            idx2 = (n1 - k1) * newSliceStride;
            idx4 = k1 * newSliceStride + n2d2 * newRowStride;
            idx5 = (n1 - k1) * newSliceStride + n2d2 * newRowStride;
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
        // }
        // ----------------------------------------

        a[n3] = a[1];
        a[1] = 0;
        idx1 = n2d2 * newRowStride;
        idx2 = n1d2 * newSliceStride;
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

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real inverse transform, i.e. you will get
     * the same result as from <code>complexInverse</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1 by n2 by 2*n3, with only the first n1
     * by n2 by n3 elements filled with real data.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverseFull(double[][][] a, boolean scale) {
        int n, nw, nc, k1, k2, k3, newn3;
        n = n1;
        if (n < n2) {
            n = n2;
        }
        n <<= 1;
        if (n < n3) {
            n = n3;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n3 > (nc << 2)) {
            nc = n3 >> 2;
            makect(nc, w, nw);
        }
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
        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            xdft3da_subth2(1, 1, a, scale);
            cdft3db_subth(1, a, scale);
            rdft3d_sub(1, a);
        } else {
            xdft3da_sub2(1, 1, a, scale);
            cdft3db_sub(1, a, scale);
            rdft3d_sub(1, a);
        }
        newn3 = 2 * n3;
        int n2d2 = n2 / 2;
        int n1d2 = n1 / 2;

        if ((nthreads > 1) && (n1 * n2 * n3 >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            fillSymmetric(a);
        } else {

            // -----------------------------------------------
            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 0; k2 < n2; k2++) {
                    for (k3 = 1; k3 < n3; k3 += 2) {
                        a[(n1 - k1) % n1][(n2 - k2) % n2][newn3 - k3] = -a[k1][k2][k3 + 2];
                        a[(n1 - k1) % n1][(n2 - k2) % n2][newn3 - k3 - 1] = a[k1][k2][k3 + 1];
                    }
                }
            }

            // ---------------------------------------------

            for (k1 = 0; k1 < n1; k1++) {
                for (k2 = 1; k2 < n2d2; k2++) {

                    a[(n1 - k1) % n1][k2][n3] = a[k1][n2 - k2][1];
                    a[k1][n2 - k2][n3] = a[k1][n2 - k2][1];
                    a[(n1 - k1) % n1][k2][n3 + 1] = -a[k1][n2 - k2][0];
                    a[k1][n2 - k2][n3 + 1] = a[k1][n2 - k2][0];

                }
            }
        }
        for (k1 = 0; k1 < n1; k1++) {
            for (k2 = 1; k2 < n2d2; k2++) {

                a[(n1 - k1) % n1][n2 - k2][0] = a[k1][k2][0];
                a[(n1 - k1) % n1][n2 - k2][1] = -a[k1][k2][1];

            }
        }

        // ----------------------------------------------------------

        for (k1 = 1; k1 < n1d2; k1++) {

            a[k1][0][n3] = a[n1 - k1][0][1];
            a[n1 - k1][0][n3] = a[n1 - k1][0][1];
            a[k1][0][n3 + 1] = -a[n1 - k1][0][0];
            a[n1 - k1][0][n3 + 1] = a[n1 - k1][0][0];
            a[k1][n2d2][n3] = a[n1 - k1][n2d2][1];
            a[n1 - k1][n2d2][n3] = a[n1 - k1][n2d2][1];
            a[k1][n2d2][n3 + 1] = -a[n1 - k1][n2d2][0];
            a[n1 - k1][n2d2][n3 + 1] = a[n1 - k1][n2d2][0];
            a[n1 - k1][0][0] = a[k1][0][0];
            a[n1 - k1][0][1] = -a[k1][0][1];
            a[n1 - k1][n2d2][0] = a[k1][n2d2][0];
            a[n1 - k1][n2d2][1] = -a[k1][n2d2][1];

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

    /* -------- initializing routines -------- */

    private void makewt(int nw) {
        int j, nwh, nw0, nw1;
        double delta, wn4r, wk1r, wk1i, wk3r, wk3i;

        ip[0] = nw;
        ip[1] = 1;
        if (nw > 2) {
            nwh = nw >> 1;
            delta = Math.atan(1.0) / nwh;
            wn4r = Math.cos(delta * nwh);
            w[0] = 1;
            w[1] = wn4r;
            if (nwh == 4) {
                w[2] = Math.cos(delta * 2);
                w[3] = Math.sin(delta * 2);
            } else if (nwh > 4) {
                makeipt(nw);
                w[2] = 0.5 / Math.cos(delta * 2);
                w[3] = 0.5 / Math.cos(delta * 6);
                for (j = 4; j < nwh; j += 4) {
                    w[j] = Math.cos(delta * j);
                    w[j + 1] = Math.sin(delta * j);
                    w[j + 2] = Math.cos(3 * delta * j);
                    w[j + 3] = -Math.sin(3 * delta * j);
                }
            }
            nw0 = 0;
            while (nwh > 2) {
                nw1 = nw0 + nwh;
                nwh >>= 1;
                w[nw1] = 1;
                w[nw1 + 1] = wn4r;
                if (nwh == 4) {
                    wk1r = w[nw0 + 4];
                    wk1i = w[nw0 + 5];
                    w[nw1 + 2] = wk1r;
                    w[nw1 + 3] = wk1i;
                } else if (nwh > 4) {
                    wk1r = w[nw0 + 4];
                    wk3r = w[nw0 + 6];
                    w[nw1 + 2] = 0.5 / wk1r;
                    w[nw1 + 3] = 0.5 / wk3r;
                    for (j = 4; j < nwh; j += 4) {
                        int idx1 = nw0 + 2 * j;
                        int idx2 = nw1 + j;
                        wk1r = w[idx1];
                        wk1i = w[idx1 + 1];
                        wk3r = w[idx1 + 2];
                        wk3i = w[idx1 + 3];
                        w[idx2] = wk1r;
                        w[idx2 + 1] = wk1i;
                        w[idx2 + 2] = wk3r;
                        w[idx2 + 3] = wk3i;
                    }
                }
                nw0 = nw1;
            }
        }
    }

    private void makeipt(int nw) {
        int j, l, m, m2, p, q;

        ip[2] = 0;
        ip[3] = 16;
        m = 2;
        for (l = nw; l > 32; l >>= 2) {
            m2 = m << 1;
            q = m2 << 3;
            for (j = m; j < m2; j++) {
                p = ip[j] << 2;
                ip[m + j] = p;
                ip[m2 + j] = p + q;
            }
            m = m2;
        }
    }

    private void makect(int nc, double[] c, int startc) {
        int j, nch;
        double delta;

        ip[1] = nc;
        if (nc > 1) {
            nch = nc >> 1;
            delta = Math.atan(1.0) / nch;
            c[startc] = Math.cos(delta * nch);
            c[startc + nch] = 0.5 * c[startc];
            for (j = 1; j < nch; j++) {
                c[startc + j] = 0.5 * Math.cos(delta * j);
                c[startc + nc - j] = 0.5 * Math.sin(delta * j);
            }
        }
    }

    /* -------- child routines -------- */

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

    private void fillSymmetric(final double[] a) {
        int np = ConcurrencyUtils.getNumberOfProcessors();
        Future[] futures = new Future[np];
        int l1k = n1 / np;
        final int newn3 = n3 * 2;
        final int newSliceStride = n2 * newn3;
        final int newRowStride = newn3;
        final int n2d2 = n2 / 2;

        for (int i = 0; i < np; i++) {
            final int l1offa, l1stopa;
            l1offa = i * l1k;
            l1stopa = l1offa + l1k;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                public void run() {
                    int k1, k2, k3, idx1, idx2, idx3;
                    // -----------------------------------------------
                    for (k1 = l1offa; k1 < l1stopa; k1++) {
                        for (k2 = 0; k2 < n2; k2++) {
                            for (k3 = 1; k3 < n3; k3 += 2) {
                                idx1 = ((n1 - k1) % n1) * newSliceStride + ((n2 - k2) % n2) * newRowStride + newn3 - k3;
                                idx2 = k1 * newSliceStride + k2 * newRowStride + k3;
                                a[idx1] = -a[idx2 + 2];
                                a[idx1 - 1] = a[idx2 + 1];
                            }
                        }
                    }

                    // ---------------------------------------------

                    for (k1 = l1offa; k1 < l1stopa; k1++) {
                        for (k2 = 1; k2 < n2d2; k2++) {
                            idx1 = ((n1 - k1) % n1) * newSliceStride + k2 * newRowStride + n3;
                            idx2 = k1 * newSliceStride + (n2 - k2) * newRowStride;
                            idx3 = k1 * newSliceStride + (n2 - k2) * newRowStride + n3;
                            a[idx1] = a[idx2 + 1];
                            a[idx3] = a[idx2 + 1];
                            a[idx1 + 1] = -a[idx2];
                            a[idx3 + 1] = a[idx2];

                        }
                    }
                }
            });
        }
        try {
            for (int j = 0; j < np; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void fillSymmetric(final double[][][] a) {
        int np = ConcurrencyUtils.getNumberOfProcessors();
        Future[] futures = new Future[np];
        int l1k = n1 / np;
        final int newn3 = n3 * 2;
        final int n2d2 = n2 / 2;

        for (int i = 0; i < np; i++) {
            final int l1offa, l1stopa;
            l1offa = i * l1k;
            l1stopa = l1offa + l1k;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                public void run() {
                    int k1, k2, k3;
                    // -----------------------------------------------
                    for (k1 = l1offa; k1 < l1stopa; k1++) {
                        for (k2 = 0; k2 < n2; k2++) {
                            for (k3 = 1; k3 < n3; k3 = k3 + 2) {
                                a[(n1 - k1) % n1][(n2 - k2) % n2][newn3 - k3] = -a[k1][k2][k3 + 2];
                                a[(n1 - k1) % n1][(n2 - k2) % n2][newn3 - k3 - 1] = a[k1][k2][k3 + 1];
                            }
                        }
                    }

                    // ---------------------------------------------

                    for (k1 = l1offa; k1 < l1stopa; k1++) {
                        for (k2 = 1; k2 < n2d2; k2++) {
                            a[(n1 - k1) % n1][k2][n3] = a[k1][n2 - k2][1];
                            a[k1][n2 - k2][n3] = a[k1][n2 - k2][1];
                            a[(n1 - k1) % n1][k2][n3 + 1] = -a[k1][n2 - k2][0];
                            a[k1][n2 - k2][n3 + 1] = a[k1][n2 - k2][0];
                        }
                    }
                }
            });
        }
        try {
            for (int j = 0; j < np; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
