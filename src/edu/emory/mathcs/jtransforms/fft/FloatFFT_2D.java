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
 * Computes 2D Discrete Fourier Transform (DFT) of complex and real, single
 * precision data. The sizes of both dimensions must be power-of-two numbers.
 * This is a parallel implementation of split-radix algorithm optimized for SMP
 * systems. <br>
 * <br>
 * This code is derived from General Purpose FFT Package written by Takuya Ooura
 * (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class FloatFFT_2D {

    private int n1;

    private int n2;

    private int[] ip;

    private float[] w;

    private float[] t;

    private FloatFFT_1D fftn2, fftn1;

    private int oldNthread;

    private int nt;

    /**
     * Creates new instance of FloatFFT_2D.
     * 
     * @param n1
     *            number of rows - must be a power-of-two number
     * @param n2
     *            number of columns - must be a power-of-two number
     */
    public FloatFFT_2D(int n1, int n2) {
        if (!ConcurrencyUtils.isPowerOf2(n1) || !ConcurrencyUtils.isPowerOf2(n2))
            throw new IllegalArgumentException("n1, n2 must be power of two numbers");
        if (n1 <= 1 || n2 <= 1) {
            throw new IllegalArgumentException("n1, n2 must be greater than 1");
        }
        this.n1 = n1;
        this.n2 = n2;
        ip = new int[2 + (int) Math.ceil(Math.sqrt(Math.max(n1, n2)))];
        this.w = new float[(int) Math.ceil(Math.max(Math.max(n1 / 2, n2 / 2), Math.max(n1 / 2, n2 / 4) + n2 / 4))];
        fftn2 = new FloatFFT_1D(n2, ip, w);
        fftn1 = new FloatFFT_1D(n1, ip, w);
        oldNthread = ConcurrencyUtils.getNumberOfProcessors();
        nt = 8 * oldNthread * n1;
        if (2 * n2 == 4 * oldNthread) {
            nt >>= 1;
        } else if (2 * n2 < 4 * oldNthread) {
            nt >>= 2;
        }
        t = new float[nt];
    }

    /**
     * Computes 2D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array in row-major order.
     * Complex number is stored as two float values in sequence: the real and
     * imaginary part, i.e. the input array must be of size n1*2*n2. The
     * physical layout of the input data is as follows:
     * 
     * <pre>
     * a[k1*2*n2+2*k2] = Re[k1][k2], 
     * a[k1*2*n2+2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2,
     * </pre>
     * 
     * @param a
     *            data to transform
     */
    public void complexForward(float[] a) {
        int nthread, n, i;
        int oldn2 = n2;
        n2 = 2 * n2;

        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * oldn2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(0, -1, a, true);
            cdft2d_subth(-1, a, true);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.complexForward(a, i * n2);
            }
            cdft2d_sub(-1, a, true);
        }
        n2 = oldn2;
    }

    /**
     * Computes 2D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 2D array. Complex data is
     * represented by 2 float values in sequence: the real and imaginary part,
     * i.e. the input array must be of size n1 by 2*n2. The physical layout of
     * the input data is as follows:
     * 
     * <pre>
     * a[k1][2*k2] = Re[k1][k2], 
     * a[k1][2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2,
     * </pre>
     * 
     * @param a
     *            data to transform
     */
    public void complexForward(float[][] a) {
        int nthread, n, i;
        int oldn2 = n2;
        n2 = 2 * n2;

        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * oldn2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(0, -1, a, true);
            cdft2d_subth(-1, a, true);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.complexForward(a[i]);
            }
            cdft2d_sub(-1, a, true);
        }
        n2 = oldn2;
    }

    /**
     * Computes 2D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array in row-major order.
     * Complex number is stored as two float values in sequence: the real and
     * imaginary part, i.e. the input array must be of size n1*2*n2. The
     * physical layout of the input data is as follows:
     * 
     * <pre>
     * a[k1*2*n2+2*k2] = Re[k1][k2], 
     * a[k1*2*n2+2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2,
     * </pre>
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void complexInverse(float[] a, boolean scale) {
        int nthread, n, i;
        int oldn2 = n2;
        n2 = 2 * n2;
        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * oldn2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(0, 1, a, scale);
            cdft2d_subth(1, a, scale);
        } else {

            for (i = 0; i < n1; i++) {
                fftn2.complexInverse(a, i * n2, scale);
            }
            cdft2d_sub(1, a, scale);
        }
        n2 = oldn2;
    }

    /**
     * Computes 2D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 2D array. Complex data is
     * represented by 2 float values in sequence: the real and imaginary part,
     * i.e. the input array must be of size n1 by 2*n2. The physical layout of
     * the input data is as follows:
     * 
     * <pre>
     * a[k1][2*k2] = Re[k1][k2], 
     * a[k1][2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2,
     * </pre>
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void complexInverse(float[][] a, boolean scale) {
        int nthread, n, i;
        int oldn2 = n2;
        n2 = 2 * n2;
        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        if (n > (ip[0] << 2)) {
            makewt(n >> 2);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * oldn2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(0, 1, a, scale);
            cdft2d_subth(1, a, scale);
        } else {

            for (i = 0; i < n1; i++) {
                fftn2.complexInverse(a[i], scale);
            }
            cdft2d_sub(1, a, scale);
        }
        n2 = oldn2;
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . The physical layout of the output data is as follows:
     * 
     * <pre>
     * a[k1*n2+2*k2] = Re[k1][k2] = Re[n1-k1][n2-k2], 
     * a[k1*n2+2*k2+1] = Im[k1][k2] = -Im[n1-k1][n2-k2], 
     *       0&lt;k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[2*k2] = Re[0][k2] = Re[0][n2-k2], 
     * a[2*k2+1] = Im[0][k2] = -Im[0][n2-k2], 
     *       0&lt;k2&lt;n2/2, 
     * a[k1*n2] = Re[k1][0] = Re[n1-k1][0], 
     * a[k1*n2+1] = Im[k1][0] = -Im[n1-k1][0], 
     * a[(n1-k1)*n2+1] = Re[k1][n2/2] = Re[n1-k1][n2/2], 
     * a[(n1-k1)*n2] = -Im[k1][n2/2] = Im[n1-k1][n2/2], 
     *       0&lt;k1&lt;n1/2, 
     * a[0] = Re[0][0], 
     * a[1] = Re[0][n2/2], 
     * a[(n1/2)*n2] = Re[n1/2][0], 
     * a[(n1/2)*n2+1] = Re[n1/2][n2/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForward(float[] a) {
        int nthread, n, nw, nc, i;

        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(1, 1, a, true);
            cdft2d_subth(-1, a, true);
            rdft2d_sub(1, a);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.realForward(a, i * n2);
            }
            cdft2d_sub(-1, a, true);
            rdft2d_sub(1, a);
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . The physical layout of the output data is as follows:
     * 
     * <pre>
     * a[k1][2*k2] = Re[k1][k2] = Re[n1-k1][n2-k2], 
     * a[k1][2*k2+1] = Im[k1][k2] = -Im[n1-k1][n2-k2], 
     *       0&lt;k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[0][2*k2] = Re[0][k2] = Re[0][n2-k2], 
     * a[0][2*k2+1] = Im[0][k2] = -Im[0][n2-k2], 
     *       0&lt;k2&lt;n2/2, 
     * a[k1][0] = Re[k1][0] = Re[n1-k1][0], 
     * a[k1][1] = Im[k1][0] = -Im[n1-k1][0], 
     * a[n1-k1][1] = Re[k1][n2/2] = Re[n1-k1][n2/2], 
     * a[n1-k1][0] = -Im[k1][n2/2] = Im[n1-k1][n2/2], 
     *       0&lt;k1&lt;n1/2, 
     * a[0][0] = Re[0][0], 
     * a[0][1] = Re[0][n2/2], 
     * a[n1/2][0] = Re[n1/2][0], 
     * a[n1/2][1] = Re[n1/2][n2/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForward(float[][] a) {
        int nthread, n, nw, nc, i;

        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(1, 1, a, true);
            cdft2d_subth(-1, a, true);
            rdft2d_sub(1, a);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.realForward(a[i]);
            }
            cdft2d_sub(-1, a, true);
            rdft2d_sub(1, a);
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real forward transform, i.e. you will get
     * the same result as from <code>complexForward</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1*2*n2, with only the first n1*n2
     * elements filled with real data. To get back the original data, use
     * <code>complexInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForwardFull(float[] a) {
        int nthread, n, nw, nc, i, newn2;
        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(1, 1, a, true);
            cdft2d_subth(-1, a, true);
            rdft2d_sub(1, a);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.realForward(a, i * n2);
            }
            cdft2d_sub(-1, a, true);
            rdft2d_sub(1, a);
        }
        newn2 = 2 * n2;
        int idx1, idx2, idx3;
        int n1d2 = n1 / 2;

        for (int k1 = (n1 - 1); k1 >= 1; k1--) {
            idx1 = k1 * n2;
            idx2 = 2 * idx1;
            for (int k2 = 0; k2 < n2; k2 += 2) {
                a[idx2 + k2] = a[idx1 + k2];
                a[idx1 + k2] = 0;
                a[idx2 + k2 + 1] = a[idx1 + k2 + 1];
                a[idx1 + k2 + 1] = 0;
            }
        }

        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            fillSymmetric(a);
        } else {
            for (int k1 = 1; k1 < n1d2; k1++) {
                idx2 = k1 * newn2;
                idx3 = (n1 - k1) * newn2;
                a[idx2 + n2] = a[idx3 + 1];
                a[idx2 + n2 + 1] = -a[idx3];
            }
            for (int k1 = 1; k1 < n1d2; k1++) {
                for (int k2 = n2 + 2; k2 < newn2; k2 += 2) {
                    idx2 = k1 * newn2;
                    idx3 = (n1 - k1) * newn2;
                    a[idx2 + k2] = a[idx3 + newn2 - k2];
                    a[idx2 + k2 + 1] = -a[idx3 + newn2 - k2 + 1];

                }
            }
            for (int k1 = 0; k1 <= n1 / 2; k1++) {
                for (int k2 = 0; k2 < newn2; k2 += 2) {
                    idx2 = k1 * newn2 + k2;
                    idx3 = ((n1 - k1) % n1) * newn2 + (newn2 - k2) % newn2;
                    a[idx3] = a[idx2];
                    a[idx3 + 1] = -a[idx2 + 1];
                }
            }
        }
        a[n2] = -a[1];
        a[1] = 0;
        a[n1d2 * newn2 + n2] = -a[n1d2 * newn2 + 1];
        a[n1d2 * newn2 + 1] = 0;
        a[n1d2 * newn2 + n2 + 1] = 0;
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real forward transform, i.e. you will get
     * the same result as from <code>complexForward</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1 by 2*n2, with only the first n1 by n2
     * elements filled with real data. To get back the original data, use
     * <code>complexInverse</code> on the output of this method.
     * 
     * @param a
     *            data to transform
     */
    public void realForwardFull(float[][] a) {
        int nthread, n, nw, nc, i, newn2;
        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth1(1, 1, a, true);
            cdft2d_subth(-1, a, true);
            rdft2d_sub(1, a);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.realForward(a[i]);
            }
            cdft2d_sub(-1, a, true);
            rdft2d_sub(1, a);
        }
        newn2 = 2 * n2;
        int n1d2 = n1 / 2;

        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            fillSymmetric(a);
        } else {
            for (int k1 = 1; k1 < n1d2; k1++) {
                a[k1][n2] = a[n1 - k1][1];
                a[k1][n2 + 1] = -a[n1 - k1][0];
            }
            for (int k1 = 1; k1 < n1d2; k1++) {
                for (int k2 = n2 + 2; k2 < newn2; k2 += 2) {
                    a[k1][k2] = a[n1 - k1][newn2 - k2];
                    a[k1][k2 + 1] = -a[n1 - k1][newn2 - k2 + 1];

                }
            }

            for (int k1 = 0; k1 <= n1 / 2; k1++) {
                for (int k2 = 0; k2 < newn2; k2 += 2) {
                    a[(n1 - k1) % n1][(newn2 - k2) % newn2] = a[k1][k2];
                    a[(n1 - k1) % n1][(newn2 - k2) % newn2 + 1] = -a[k1][k2 + 1];
                }
            }
        }

        a[0][n2] = -a[0][1];
        a[0][1] = 0;
        a[n1d2][n2] = -a[n1d2][1];
        a[n1d2][1] = 0;
        a[n1d2][n2 + 1] = 0;
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * a[k1*n2+2*k2] = Re[k1][k2] = Re[n1-k1][n2-k2], 
     * a[k1*n2+2*k2+1] = Im[k1][k2] = -Im[n1-k1][n2-k2], 
     *       0&lt;k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[2*k2] = Re[0][k2] = Re[0][n2-k2], 
     * a[2*k2+1] = Im[0][k2] = -Im[0][n2-k2], 
     *       0&lt;k2&lt;n2/2, 
     * a[k1*n2] = Re[k1][0] = Re[n1-k1][0], 
     * a[k1*n2+1] = Im[k1][0] = -Im[n1-k1][0], 
     * a[(n1-k1)*n2+1] = Re[k1][n2/2] = Re[n1-k1][n2/2], 
     * a[(n1-k1)*n2] = -Im[k1][n2/2] = Im[n1-k1][n2/2], 
     *       0&lt;k1&lt;n1/2, 
     * a[0] = Re[0][0], 
     * a[1] = Re[0][n2/2], 
     * a[(n1/2)*n2] = Re[n1/2][0], 
     * a[(n1/2)*n2+1] = Re[n1/2][n2/2]
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
    public void realInverse(float[] a, boolean scale) {
        int nthread, n, nw, nc, i;

        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            rdft2d_sub(-1, a);
            cdft2d_subth(1, a, scale);
            xdft2d0_subth1(1, -1, a, scale);
        } else {
            rdft2d_sub(-1, a);
            cdft2d_sub(1, a, scale);
            for (i = 0; i < n1; i++) {
                fftn2.realInverse(a, i * n2, scale);
            }
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * a[k1][2*k2] = Re[k1][k2] = Re[n1-k1][n2-k2], 
     * a[k1][2*k2+1] = Im[k1][k2] = -Im[n1-k1][n2-k2], 
     *       0&lt;k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * a[0][2*k2] = Re[0][k2] = Re[0][n2-k2], 
     * a[0][2*k2+1] = Im[0][k2] = -Im[0][n2-k2], 
     *       0&lt;k2&lt;n2/2, 
     * a[k1][0] = Re[k1][0] = Re[n1-k1][0], 
     * a[k1][1] = Im[k1][0] = -Im[n1-k1][0], 
     * a[n1-k1][1] = Re[k1][n2/2] = Re[n1-k1][n2/2], 
     * a[n1-k1][0] = -Im[k1][n2/2] = Im[n1-k1][n2/2], 
     *       0&lt;k1&lt;n1/2, 
     * a[0][0] = Re[0][0], 
     * a[0][1] = Re[0][n2/2], 
     * a[n1/2][0] = Re[n1/2][0], 
     * a[n1/2][1] = Re[n1/2][n2/2]
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
    public void realInverse(float[][] a, boolean scale) {
        int nthread, n, nw, nc, i;

        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            rdft2d_sub(-1, a);
            cdft2d_subth(1, a, scale);
            xdft2d0_subth1(1, -1, a, scale);
        } else {
            rdft2d_sub(-1, a);
            cdft2d_sub(1, a, scale);
            for (i = 0; i < n1; i++) {
                fftn2.realInverse(a[i], scale);
            }
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real inverse transform, i.e. you will get
     * the same result as from <code>complexInverse</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1*2*n2, with only the first n1*n2
     * elements filled with real data.
     * 
     * @param a
     *            data to transform
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverseFull(float[] a, boolean scale) {
        int nthread, n, nw, nc, i, newn2;
        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth2(1, -1, a, scale);
            cdft2d_subth(1, a, scale);
            rdft2d_sub(1, a);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.realInverse2(a, i * n2, scale);
            }
            cdft2d_sub(1, a, scale);
            rdft2d_sub(1, a);
        }
        newn2 = 2 * n2;
        int idx1, idx2, idx3;
        int n1d2 = n1 / 2;

        for (int k1 = (n1 - 1); k1 >= 1; k1--) {
            idx1 = k1 * n2;
            idx2 = 2 * idx1;
            for (int k2 = 0; k2 < n2; k2 += 2) {
                a[idx2 + k2] = a[idx1 + k2];
                a[idx1 + k2] = 0;
                a[idx2 + k2 + 1] = a[idx1 + k2 + 1];
                a[idx1 + k2 + 1] = 0;
            }
        }

        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            fillSymmetric(a);
        } else {
            for (int k1 = 1; k1 < n1d2; k1++) {
                idx2 = k1 * newn2;
                idx3 = (n1 - k1) * newn2;
                a[idx2 + n2] = a[idx3 + 1];
                a[idx2 + n2 + 1] = -a[idx3];
            }
            for (int k1 = 1; k1 < n1d2; k1++) {
                for (int k2 = n2 + 2; k2 < newn2; k2 += 2) {
                    idx2 = k1 * newn2;
                    idx3 = (n1 - k1) * newn2;
                    a[idx2 + k2] = a[idx3 + newn2 - k2];
                    a[idx2 + k2 + 1] = -a[idx3 + newn2 - k2 + 1];

                }
            }
            for (int k1 = 0; k1 <= n1 / 2; k1++) {
                for (int k2 = 0; k2 < newn2; k2 += 2) {
                    idx2 = k1 * newn2 + k2;
                    idx3 = ((n1 - k1) % n1) * newn2 + (newn2 - k2) % newn2;
                    a[idx3] = a[idx2];
                    a[idx3 + 1] = -a[idx2 + 1];
                }
            }
        }
        a[n2] = -a[1];
        a[1] = 0;
        a[n1d2 * newn2 + n2] = -a[n1d2 * newn2 + 1];
        a[n1d2 * newn2 + 1] = 0;
        a[n1d2 * newn2 + n2 + 1] = 0;
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes the full real inverse transform, i.e. you will get
     * the same result as from <code>complexInverse</code> called with all
     * imaginary part equal 0. Because the result is stored in <code>a</code>,
     * the input array must be of size n1 by 2*n2, with only the first n1 by n2
     * elements filled with real data.
     * 
     * @param a
     *            data to transform
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void realInverseFull(float[][] a, boolean scale) {
        int nthread, n, nw, nc, i, newn2;
        n = n1 << 1;
        if (n < n2) {
            n = n2;
        }
        nw = ip[0];
        if (n > (nw << 2)) {
            nw = n >> 2;
            makewt(nw);
        }
        nc = ip[1];
        if (n2 > (nc << 2)) {
            nc = n2 >> 2;
            makect(nc, w, nw);
        }
        nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (nthread != oldNthread) {
            nt = 8 * nthread * n1;
            if (n2 == 4 * nthread) {
                nt >>= 1;
            } else if (n2 < 4 * nthread) {
                nt >>= 2;
            }
            t = new float[nt];
            oldNthread = nthread;
        }
        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            xdft2d0_subth2(1, -1, a, scale);
            cdft2d_subth(1, a, scale);
            rdft2d_sub(1, a);
        } else {
            for (i = 0; i < n1; i++) {
                fftn2.realInverse2(a[i], 0, scale);
            }
            cdft2d_sub(1, a, scale);
            rdft2d_sub(1, a);
        }
        newn2 = 2 * n2;
        int n1d2 = n1 / 2;

        if ((nthread > 1) && (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            fillSymmetric(a);
        } else {
            for (int k1 = 1; k1 < n1d2; k1++) {
                a[k1][n2] = a[n1 - k1][1];
                a[k1][n2 + 1] = -a[n1 - k1][0];
            }
            for (int k1 = 1; k1 < n1d2; k1++) {
                for (int k2 = n2 + 2; k2 < newn2; k2 += 2) {
                    a[k1][k2] = a[n1 - k1][newn2 - k2];
                    a[k1][k2 + 1] = -a[n1 - k1][newn2 - k2 + 1];

                }
            }

            for (int k1 = 0; k1 <= n1 / 2; k1++) {
                for (int k2 = 0; k2 < newn2; k2 += 2) {
                    a[(n1 - k1) % n1][(newn2 - k2) % newn2] = a[k1][k2];
                    a[(n1 - k1) % n1][(newn2 - k2) % newn2 + 1] = -a[k1][k2 + 1];
                }
            }
        }

        a[0][n2] = -a[0][1];
        a[0][1] = 0;
        a[n1d2][n2] = -a[n1d2][1];
        a[n1d2][1] = 0;
        a[n1d2][n2 + 1] = 0;
    }

    /* -------- initializing routines -------- */

    private void makewt(int nw) {
        int j, nwh, nw0, nw1;
        float delta, wn4r, wk1r, wk1i, wk3r, wk3i;

        ip[0] = nw;
        ip[1] = 1;
        if (nw > 2) {
            nwh = nw >> 1;
            delta = (float) (Math.atan(1.0) / nwh);
            wn4r = (float) Math.cos(delta * nwh);
            w[0] = 1;
            w[1] = wn4r;
            if (nwh == 4) {
                w[2] = (float) Math.cos(delta * 2);
                w[3] = (float) Math.sin(delta * 2);
            } else if (nwh > 4) {
                makeipt(nw);
                w[2] = (float) (0.5 / Math.cos(delta * 2));
                w[3] = (float) (0.5 / Math.cos(delta * 6));
                for (j = 4; j < nwh; j += 4) {
                    w[j] = (float) Math.cos(delta * j);
                    w[j + 1] = (float) Math.sin(delta * j);
                    w[j + 2] = (float) Math.cos(3 * delta * j);
                    w[j + 3] = (float) -Math.sin(3 * delta * j);
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
                    w[nw1 + 2] = (float) (0.5 / wk1r);
                    w[nw1 + 3] = (float) (0.5 / wk3r);
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

    private void makect(int nc, float[] c, int startc) {
        int j, nch;
        float delta;

        ip[1] = nc;
        if (nc > 1) {
            nch = nc >> 1;
            delta = (float) Math.atan(1.0) / nch;
            c[startc] = (float) Math.cos(delta * nch);
            c[startc + nch] = 0.5f * c[startc];
            for (j = 1; j < nch; j++) {
                c[startc + j] = (float) (0.5 * Math.cos(delta * j));
                c[startc + nc - j] = (float) (0.5 * Math.sin(delta * j));
            }
        }
    }

    /* -------- child routines -------- */

    private void rdft2d_sub(int isgn, float[] a) {
        int n1h, i, j;
        float xi;
        int idx1, idx2;

        n1h = n1 >> 1;
        if (isgn < 0) {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                idx1 = i * n2;
                idx2 = j * n2;
                xi = a[idx1] - a[idx2];
                a[idx1] += a[idx2];
                a[idx2] = xi;
                xi = a[idx2 + 1] - a[idx1 + 1];
                a[idx1 + 1] += a[idx2 + 1];
                a[idx2 + 1] = xi;
            }
        } else {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                idx1 = i * n2;
                idx2 = j * n2;
                a[idx2] = 0.5f * (a[idx1] - a[idx2]);
                a[idx1] -= a[idx2];
                a[idx2 + 1] = 0.5f * (a[idx1 + 1] + a[idx2 + 1]);
                a[idx1 + 1] -= a[idx2 + 1];
            }
        }
    }

    private void rdft2d_sub(int isgn, float[][] a) {
        int n1h, i, j;
        float xi;

        n1h = n1 >> 1;
        if (isgn < 0) {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                xi = a[i][0] - a[j][0];
                a[i][0] += a[j][0];
                a[j][0] = xi;
                xi = a[j][1] - a[i][1];
                a[i][1] += a[j][1];
                a[j][1] = xi;
            }
        } else {
            for (i = 1; i < n1h; i++) {
                j = n1 - i;
                a[j][0] = 0.5f * (a[i][0] - a[j][0]);
                a[i][0] -= a[j][0];
                a[j][1] = 0.5f * (a[i][1] + a[j][1]);
                a[i][1] -= a[j][1];
            }
        }
    }

    private void cdft2d_sub(int isgn, float[] a, boolean scale) {
        int i, j;
        int idx1, idx2, idx3, idx4, idx5;
        if (isgn == -1) {
            if (n2 > 4) {
                for (j = 0; j < n2; j += 8) {
                    for (i = 0; i < n1; i++) {
                        idx1 = i * n2 + j;
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
                        idx1 = i * n2 + j;
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
            } else if (n2 == 4) {
                for (i = 0; i < n1; i++) {
                    idx1 = i * n2;
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
                    idx1 = i * n2;
                    idx2 = 2 * i;
                    idx3 = 2 * n1 + 2 * i;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                    a[idx1 + 2] = t[idx3];
                    a[idx1 + 3] = t[idx3 + 1];
                }
            } else if (n2 == 2) {
                for (i = 0; i < n1; i++) {
                    idx1 = i * n2;
                    idx2 = 2 * i;
                    t[idx2] = a[idx1];
                    t[idx2 + 1] = a[idx1 + 1];
                }
                fftn1.complexForward(t, 0);
                for (i = 0; i < n1; i++) {
                    idx1 = i * n2;
                    idx2 = 2 * i;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                }
            }
        } else {
            if (n2 > 4) {
                for (j = 0; j < n2; j += 8) {
                    for (i = 0; i < n1; i++) {
                        idx1 = i * n2 + j;
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
                        idx1 = i * n2 + j;
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
            } else if (n2 == 4) {
                for (i = 0; i < n1; i++) {
                    idx1 = i * n2;
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
                    idx1 = i * n2;
                    idx2 = 2 * i;
                    idx3 = 2 * n1 + 2 * i;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                    a[idx1 + 2] = t[idx3];
                    a[idx1 + 3] = t[idx3 + 1];
                }
            } else if (n2 == 2) {
                for (i = 0; i < n1; i++) {
                    idx1 = i * n2;
                    idx2 = 2 * i;
                    t[idx2] = a[idx1];
                    t[idx2 + 1] = a[idx1 + 1];
                }
                fftn1.complexInverse(t, 0, scale);
                for (i = 0; i < n1; i++) {
                    idx1 = i * n2;
                    idx2 = 2 * i;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                }
            }
        }
    }

    private void cdft2d_sub(int isgn, float[][] a, boolean scale) {
        int i, j;
        int idx2, idx3, idx4, idx5;
        if (isgn == -1) {
            if (n2 > 4) {
                for (j = 0; j < n2; j += 8) {
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        idx4 = idx3 + 2 * n1;
                        idx5 = idx4 + 2 * n1;
                        t[idx2] = a[i][j];
                        t[idx2 + 1] = a[i][j + 1];
                        t[idx3] = a[i][j + 2];
                        t[idx3 + 1] = a[i][j + 3];
                        t[idx4] = a[i][j + 4];
                        t[idx4 + 1] = a[i][j + 5];
                        t[idx5] = a[i][j + 6];
                        t[idx5 + 1] = a[i][j + 7];
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
                        a[i][j] = t[idx2];
                        a[i][j + 1] = t[idx2 + 1];
                        a[i][j + 2] = t[idx3];
                        a[i][j + 3] = t[idx3 + 1];
                        a[i][j + 4] = t[idx4];
                        a[i][j + 5] = t[idx4 + 1];
                        a[i][j + 6] = t[idx5];
                        a[i][j + 7] = t[idx5 + 1];
                    }
                }
            } else if (n2 == 4) {
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    idx3 = 2 * n1 + 2 * i;
                    t[idx2] = a[i][0];
                    t[idx2 + 1] = a[i][1];
                    t[idx3] = a[i][2];
                    t[idx3 + 1] = a[i][3];
                }
                fftn1.complexForward(t, 0);
                fftn1.complexForward(t, 2 * n1);
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    idx3 = 2 * n1 + 2 * i;
                    a[i][0] = t[idx2];
                    a[i][1] = t[idx2 + 1];
                    a[i][2] = t[idx3];
                    a[i][3] = t[idx3 + 1];
                }
            } else if (n2 == 2) {
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    t[idx2] = a[i][0];
                    t[idx2 + 1] = a[i][1];
                }
                fftn1.complexForward(t, 0);
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    a[i][0] = t[idx2];
                    a[i][1] = t[idx2 + 1];
                }
            }
        } else {
            if (n2 > 4) {
                for (j = 0; j < n2; j += 8) {
                    for (i = 0; i < n1; i++) {
                        idx2 = 2 * i;
                        idx3 = 2 * n1 + 2 * i;
                        idx4 = idx3 + 2 * n1;
                        idx5 = idx4 + 2 * n1;
                        t[idx2] = a[i][j];
                        t[idx2 + 1] = a[i][j + 1];
                        t[idx3] = a[i][j + 2];
                        t[idx3 + 1] = a[i][j + 3];
                        t[idx4] = a[i][j + 4];
                        t[idx4 + 1] = a[i][j + 5];
                        t[idx5] = a[i][j + 6];
                        t[idx5 + 1] = a[i][j + 7];
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
                        a[i][j] = t[idx2];
                        a[i][j + 1] = t[idx2 + 1];
                        a[i][j + 2] = t[idx3];
                        a[i][j + 3] = t[idx3 + 1];
                        a[i][j + 4] = t[idx4];
                        a[i][j + 5] = t[idx4 + 1];
                        a[i][j + 6] = t[idx5];
                        a[i][j + 7] = t[idx5 + 1];
                    }
                }
            } else if (n2 == 4) {
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    idx3 = 2 * n1 + 2 * i;
                    t[idx2] = a[i][0];
                    t[idx2 + 1] = a[i][1];
                    t[idx3] = a[i][2];
                    t[idx3 + 1] = a[i][3];
                }
                fftn1.complexInverse(t, 0, scale);
                fftn1.complexInverse(t, 2 * n1, scale);
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    idx3 = 2 * n1 + 2 * i;
                    a[i][0] = t[idx2];
                    a[i][1] = t[idx2 + 1];
                    a[i][2] = t[idx3];
                    a[i][3] = t[idx3 + 1];
                }
            } else if (n2 == 2) {
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    t[idx2] = a[i][0];
                    t[idx2 + 1] = a[i][1];
                }
                fftn1.complexInverse(t, 0, scale);
                for (i = 0; i < n1; i++) {
                    idx2 = 2 * i;
                    a[i][0] = t[idx2];
                    a[i][1] = t[idx2 + 1];
                }
            }
        }
    }

    private void xdft2d0_subth1(final int icr, final int isgn, final float[] a, final boolean scale) {
        final int nthread;
        int i;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if (np > n1) {
            nthread = n1;
        } else {
            nthread = np;
        }
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i;
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexForward(a, i * n2);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexInverse(a, i * n2, scale);
                            }
                        }
                    } else {
                        if (isgn == 1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realForward(a, i * n2);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realInverse(a, i * n2, scale);
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

    private void xdft2d0_subth2(final int icr, final int isgn, final float[] a, final boolean scale) {
        final int nthread;
        int i;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if (np > n1) {
            nthread = n1;
        } else {
            nthread = np;
        }
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i;
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexForward(a, i * n2);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexInverse(a, i * n2, scale);
                            }
                        }
                    } else {
                        if (isgn == 1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realForward(a, i * n2);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realInverse2(a, i * n2, scale);
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

    private void xdft2d0_subth1(final int icr, final int isgn, final float[][] a, final boolean scale) {
        final int nthread;
        int i;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if (np > n1) {
            nthread = n1;
        } else {
            nthread = np;
        }
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i;
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexForward(a[i]);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexInverse(a[i], scale);
                            }
                        }
                    } else {
                        if (isgn == 1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realForward(a[i]);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realInverse(a[i], scale);
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

    private void xdft2d0_subth2(final int icr, final int isgn, final float[][] a, final boolean scale) {
        final int nthread;
        int i;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if (np > n1) {
            nthread = n1;
        } else {
            nthread = np;
        }
        Future[] futures = new Future[nthread];
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i;
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexForward(a[i]);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.complexInverse(a[i], scale);
                            }
                        }
                    } else {
                        if (isgn == 1) {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realForward(a[i]);
                            }
                        } else {
                            for (i = n0; i < n1; i += nthread) {
                                fftn2.realInverse2(a[i], 0, scale);
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

    private void cdft2d_subth(final int isgn, final float[] a, final boolean scale) {
        int nthread;
        int nt, i;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        nthread = np;
        nt = 8 * n1;
        if (n2 == 4 * np) {
            nt >>= 1;
        } else if (n2 < 4 * np) {
            nthread = n2 >> 1;
            nt >>= 2;
        }
        Future[] futures = new Future[nthread];
        final int nthread_f = nthread;
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j;
                    int idx1, idx2, idx3, idx4, idx5;
                    if (isgn == -1) {
                        if (n2 > 4 * nthread_f) {
                            for (j = 8 * n0; j < n2; j += 8 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * n2 + j;
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
                                    idx1 = i * n2 + j;
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
                        } else if (n2 == 4 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx1 = i * n2 + 4 * n0;
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
                                idx1 = i * n2 + 4 * n0;
                                idx2 = startt + 2 * i;
                                idx3 = startt + 2 * n1 + 2 * i;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
                                a[idx1 + 2] = t[idx3];
                                a[idx1 + 3] = t[idx3 + 1];
                            }
                        } else if (n2 == 2 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx1 = i * n2 + 2 * n0;
                                idx2 = startt + 2 * i;
                                t[idx2] = a[idx1];
                                t[idx2 + 1] = a[idx1 + 1];
                            }
                            fftn1.complexForward(t, startt);
                            for (i = 0; i < n1; i++) {
                                idx1 = i * n2 + 2 * n0;
                                idx2 = startt + 2 * i;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
                            }
                        }
                    } else {
                        if (n2 > 4 * nthread_f) {
                            for (j = 8 * n0; j < n2; j += 8 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * n2 + j;
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
                                    idx1 = i * n2 + j;
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
                        } else if (n2 == 4 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx1 = i * n2 + 4 * n0;
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
                                idx1 = i * n2 + 4 * n0;
                                idx2 = startt + 2 * i;
                                idx3 = startt + 2 * n1 + 2 * i;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
                                a[idx1 + 2] = t[idx3];
                                a[idx1 + 3] = t[idx3 + 1];
                            }
                        } else if (n2 == 2 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx1 = i * n2 + 2 * n0;
                                idx2 = startt + 2 * i;
                                t[idx2] = a[idx1];
                                t[idx2 + 1] = a[idx1 + 1];
                            }
                            fftn1.complexInverse(t, startt, scale);
                            for (i = 0; i < n1; i++) {
                                idx1 = i * n2 + 2 * n0;
                                idx2 = startt + 2 * i;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
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

    private void cdft2d_subth(final int isgn, final float[][] a, final boolean scale) {
        int nthread;
        int nt, i;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        nthread = np;
        nt = 8 * n1;
        if (n2 == 4 * np) {
            nt >>= 1;
        } else if (n2 < 4 * np) {
            nthread = n2 >> 1;
            nt >>= 2;
        }
        Future[] futures = new Future[nthread];
        final int nthread_f = nthread;
        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j;
                    int idx2, idx3, idx4, idx5;
                    if (isgn == -1) {
                        if (n2 > 4 * nthread_f) {
                            for (j = 8 * n0; j < n2; j += 8 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    idx4 = idx3 + 2 * n1;
                                    idx5 = idx4 + 2 * n1;
                                    t[idx2] = a[i][j];
                                    t[idx2 + 1] = a[i][j + 1];
                                    t[idx3] = a[i][j + 2];
                                    t[idx3 + 1] = a[i][j + 3];
                                    t[idx4] = a[i][j + 4];
                                    t[idx4 + 1] = a[i][j + 5];
                                    t[idx5] = a[i][j + 6];
                                    t[idx5 + 1] = a[i][j + 7];
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
                                    a[i][j] = t[idx2];
                                    a[i][j + 1] = t[idx2 + 1];
                                    a[i][j + 2] = t[idx3];
                                    a[i][j + 3] = t[idx3 + 1];
                                    a[i][j + 4] = t[idx4];
                                    a[i][j + 5] = t[idx4 + 1];
                                    a[i][j + 6] = t[idx5];
                                    a[i][j + 7] = t[idx5 + 1];
                                }
                            }
                        } else if (n2 == 4 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                idx3 = startt + 2 * n1 + 2 * i;
                                t[idx2] = a[i][4 * n0];
                                t[idx2 + 1] = a[i][4 * n0 + 1];
                                t[idx3] = a[i][4 * n0 + 2];
                                t[idx3 + 1] = a[i][4 * n0 + 3];
                            }
                            fftn1.complexForward(t, startt);
                            fftn1.complexForward(t, startt + 2 * n1);
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                idx3 = startt + 2 * n1 + 2 * i;
                                a[i][4 * n0] = t[idx2];
                                a[i][4 * n0 + 1] = t[idx2 + 1];
                                a[i][4 * n0 + 2] = t[idx3];
                                a[i][4 * n0 + 3] = t[idx3 + 1];
                            }
                        } else if (n2 == 2 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                t[idx2] = a[i][2 * n0];
                                t[idx2 + 1] = a[i][2 * n0 + 1];
                            }
                            fftn1.complexForward(t, startt);
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                a[i][2 * n0] = t[idx2];
                                a[i][2 * n0 + 1] = t[idx2 + 1];
                            }
                        }
                    } else {
                        if (n2 > 4 * nthread_f) {
                            for (j = 8 * n0; j < n2; j += 8 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + 2 * i;
                                    idx3 = startt + 2 * n1 + 2 * i;
                                    idx4 = idx3 + 2 * n1;
                                    idx5 = idx4 + 2 * n1;
                                    t[idx2] = a[i][j];
                                    t[idx2 + 1] = a[i][j + 1];
                                    t[idx3] = a[i][j + 2];
                                    t[idx3 + 1] = a[i][j + 3];
                                    t[idx4] = a[i][j + 4];
                                    t[idx4 + 1] = a[i][j + 5];
                                    t[idx5] = a[i][j + 6];
                                    t[idx5 + 1] = a[i][j + 7];
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
                                    a[i][j] = t[idx2];
                                    a[i][j + 1] = t[idx2 + 1];
                                    a[i][j + 2] = t[idx3];
                                    a[i][j + 3] = t[idx3 + 1];
                                    a[i][j + 4] = t[idx4];
                                    a[i][j + 5] = t[idx4 + 1];
                                    a[i][j + 6] = t[idx5];
                                    a[i][j + 7] = t[idx5 + 1];
                                }
                            }
                        } else if (n2 == 4 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                idx3 = startt + 2 * n1 + 2 * i;
                                t[idx2] = a[i][4 * n0];
                                t[idx2 + 1] = a[i][4 * n0 + 1];
                                t[idx3] = a[i][4 * n0 + 2];
                                t[idx3 + 1] = a[i][4 * n0 + 3];
                            }
                            fftn1.complexInverse(t, startt, scale);
                            fftn1.complexInverse(t, startt + 2 * n1, scale);
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                idx3 = startt + 2 * n1 + 2 * i;
                                a[i][4 * n0] = t[idx2];
                                a[i][4 * n0 + 1] = t[idx2 + 1];
                                a[i][4 * n0 + 2] = t[idx3];
                                a[i][4 * n0 + 3] = t[idx3 + 1];
                            }
                        } else if (n2 == 2 * nthread_f) {
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                t[idx2] = a[i][2 * n0];
                                t[idx2 + 1] = a[i][2 * n0 + 1];
                            }
                            fftn1.complexInverse(t, startt, scale);
                            for (i = 0; i < n1; i++) {
                                idx2 = startt + 2 * i;
                                a[i][2 * n0] = t[idx2];
                                a[i][2 * n0 + 1] = t[idx2 + 1];
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

    private void fillSymmetric(final float[] a) {
        int np = ConcurrencyUtils.getNumberOfProcessors();
        Future[] futures = new Future[np];
        int n1d2 = n1 / 2;
        int l1k = n1d2 / np;
        final int newn2 = 2 * n2;
        for (int i = 0; i < np; i++) {
            final int l1offa, l1stopa, l2offa, l2stopa;
            if (i == 0)
                l1offa = i * l1k + 1;
            else {
                l1offa = i * l1k;
            }
            l1stopa = i * l1k + l1k;
            l2offa = i * l1k;
            if (i == np - 1) {
                l2stopa = i * l1k + l1k + 1;
            } else {
                l2stopa = i * l1k + l1k;
            }
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int idx1, idx2;

                    for (int k1 = l1offa; k1 < l1stopa; k1++) {
                        idx1 = k1 * newn2;
                        idx2 = (n1 - k1) * newn2;
                        a[idx1 + n2] = a[idx2 + 1];
                        a[idx1 + n2 + 1] = -a[idx2];
                    }
                    for (int k1 = l1offa; k1 < l1stopa; k1++) {
                        for (int k2 = n2 + 2; k2 < newn2; k2 = k2 + 2) {
                            idx1 = k1 * newn2;
                            idx2 = (n1 - k1) * newn2 + newn2 - k2;
                            a[idx1 + k2] = a[idx2];
                            a[idx1 + k2 + 1] = -a[idx2 + 1];

                        }
                    }
                    for (int k1 = l2offa; k1 < l2stopa; k1++) {
                        for (int k2 = 0; k2 < newn2; k2 = k2 + 2) {
                            idx1 = ((n1 - k1) % n1) * newn2 + (newn2 - k2) % newn2;
                            idx2 = k1 * newn2 + k2;
                            a[idx1] = a[idx2];
                            a[idx1 + 1] = -a[idx2 + 1];
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

    private void fillSymmetric(final float[][] a) {
        int np = ConcurrencyUtils.getNumberOfProcessors();
        Future[] futures = new Future[np];
        int n1d2 = n1 / 2;
        int l1k = n1d2 / np;
        final int newn2 = 2 * n2;
        for (int i = 0; i < np; i++) {
            final int l1offa, l1stopa, l2offa, l2stopa;
            if (i == 0)
                l1offa = i * l1k + 1;
            else {
                l1offa = i * l1k;
            }
            l1stopa = i * l1k + l1k;
            l2offa = i * l1k;
            if (i == np - 1) {
                l2stopa = i * l1k + l1k + 1;
            } else {
                l2stopa = i * l1k + l1k;
            }
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {

                    for (int k1 = l1offa; k1 < l1stopa; k1++) {
                        a[k1][n2] = a[n1 - k1][1];
                        a[k1][n2 + 1] = -a[n1 - k1][0];
                    }
                    for (int k1 = l1offa; k1 < l1stopa; k1++) {
                        for (int k2 = n2 + 2; k2 < newn2; k2 = k2 + 2) {
                            a[k1][k2] = a[n1 - k1][newn2 - k2];
                            a[k1][k2 + 1] = -a[n1 - k1][newn2 - k2 + 1];

                        }
                    }
                    for (int k1 = l2offa; k1 < l2stopa; k1++) {
                        for (int k2 = 0; k2 < newn2; k2 = k2 + 2) {
                            a[(n1 - k1) % n1][(newn2 - k2) % newn2] = a[k1][k2];
                            a[(n1 - k1) % n1][(newn2 - k2) % newn2 + 1] = -a[k1][k2 + 1];
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
