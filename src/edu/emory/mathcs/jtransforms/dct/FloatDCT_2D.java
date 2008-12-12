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

package edu.emory.mathcs.jtransforms.dct;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Computes 2D Discrete Cosine Transform (DCT) of single precision data. The
 * sizes of both dimensions can be arbitrary numbers. This is a parallel
 * implementation of split-radix and mixed-radix algorithms optimized for SMP
 * systems. <br>
 * <br>
 * This code is derived from General Purpose FFT Package written by Takuya Ooura
 * (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class FloatDCT_2D {

    private int n1;

    private int n2;

    private float[] t;

    private FloatDCT_1D dctn2, dctn1;

    private int nt;

    private int oldNthread;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of FloatDCT_2D.
     * 
     * @param n1
     *            number of rows
     * @param n2
     *            number of columns
     */
    public FloatDCT_2D(int n1, int n2) {
        if (n1 <= 1 || n2 <= 1) {
            throw new IllegalArgumentException("n1, n2 must be greater than 1");
        }
        this.n1 = n1;
        this.n2 = n2;
        if (n1 * n2 >= ConcurrencyUtils.getThreadsBeginN_2D()) {
            this.useThreads = true;
        }
        if (ConcurrencyUtils.isPowerOf2(n1) && ConcurrencyUtils.isPowerOf2(n2)) {
            isPowerOfTwo = true;
            oldNthread = ConcurrencyUtils.getNumberOfProcessors();
            nt = 4 * oldNthread * n1;
            if (n2 == 2 * oldNthread) {
                nt >>= 1;
            } else if (n2 < 2 * oldNthread) {
                nt >>= 2;
            }
            t = new float[nt];
        }
        dctn2 = new FloatDCT_1D(n2);
        if (n2 == n1) {
            dctn1 = dctn2;
        } else {
            dctn1 = new FloatDCT_1D(n1);
        }
    }

    /**
     * Computes 2D forward DCT (DCT-II) leaving the result in <code>a</code>.
     * The data is stored in 1D array in row-major order.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void forward(final float[] a, final boolean scale) {
        int nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (isPowerOfTwo) {
            if (nthread != oldNthread) {
                nt = 4 * nthread * n1;
                if (n2 == 2 * nthread) {
                    nt >>= 1;
                } else if (n2 < 2 * nthread) {
                    nt >>= 2;
                }
                t = new float[nt];
                oldNthread = nthread;
            }
            if ((nthread > 1) && useThreads) {
                ddxt2d_subth(-1, a, scale);
                ddxt2d0_subth(-1, a, scale);
            } else {
                ddxt2d_sub(-1, a, scale);
                for (int i = 0; i < n1; i++) {
                    dctn2.forward(a, i * n2, scale);
                }
            }
        } else {
            if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread)) {
                Future[] futures = new Future[nthread];
                int p = n1 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthread - 1) {
                        stopRow = n1;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int i = startRow; i < stopRow; i++) {
                                dctn2.forward(a, i * n2, scale);
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                p = n2 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startCol = l * p;
                    final int stopCol;
                    if (l == nthread - 1) {
                        stopCol = n2;
                    } else {
                        stopCol = startCol + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            float[] temp = new float[n1];
                            for (int c = startCol; c < stopCol; c++) {
                                for (int r = 0; r < n1; r++) {
                                    temp[r] = a[r * n2 + c];
                                }
                                dctn1.forward(temp, scale);
                                for (int r = 0; r < n1; r++) {
                                    a[r * n2 + c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                for (int i = 0; i < n1; i++) {
                    dctn2.forward(a, i * n2, scale);
                }
                float[] temp = new float[n1];
                for (int c = 0; c < n2; c++) {
                    for (int r = 0; r < n1; r++) {
                        temp[r] = a[r * n2 + c];
                    }
                    dctn1.forward(temp, scale);
                    for (int r = 0; r < n1; r++) {
                        a[r * n2 + c] = temp[r];
                    }
                }
            }
        }
    }

    /**
     * Computes 2D forward DCT (DCT-II) leaving the result in <code>a</code>.
     * The data is stored in 2D array.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void forward(final float[][] a, final boolean scale) {
        int nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (isPowerOfTwo) {
            if (nthread != oldNthread) {
                nt = 4 * nthread * n1;
                if (n2 == 2 * nthread) {
                    nt >>= 1;
                } else if (n2 < 2 * nthread) {
                    nt >>= 2;
                }
                t = new float[nt];
                oldNthread = nthread;
            }
            if ((nthread > 1) && useThreads) {
                ddxt2d_subth(-1, a, scale);
                ddxt2d0_subth(-1, a, scale);
            } else {
                ddxt2d_sub(-1, a, scale);
                for (int i = 0; i < n1; i++) {
                    dctn2.forward(a[i], scale);
                }
            }
        } else {
            if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread)) {
                Future[] futures = new Future[nthread];
                int p = n1 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthread - 1) {
                        stopRow = n1;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int i = startRow; i < stopRow; i++) {
                                dctn2.forward(a[i], scale);
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                p = n2 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startCol = l * p;
                    final int stopCol;
                    if (l == nthread - 1) {
                        stopCol = n2;
                    } else {
                        stopCol = startCol + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            float[] temp = new float[n1];
                            for (int c = startCol; c < stopCol; c++) {
                                for (int r = 0; r < n1; r++) {
                                    temp[r] = a[r][c];
                                }
                                dctn1.forward(temp, scale);
                                for (int r = 0; r < n1; r++) {
                                    a[r][c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                for (int i = 0; i < n1; i++) {
                    dctn2.forward(a[i], scale);
                }
                float[] temp = new float[n1];
                for (int c = 0; c < n2; c++) {
                    for (int r = 0; r < n1; r++) {
                        temp[r] = a[r][c];
                    }
                    dctn1.forward(temp, scale);
                    for (int r = 0; r < n1; r++) {
                        a[r][c] = temp[r];
                    }
                }
            }
        }
    }

    /**
     * Computes 2D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     * The data is stored in 1D array in row-major order.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void inverse(final float[] a, final boolean scale) {
        int nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (isPowerOfTwo) {
            if (nthread != oldNthread) {
                nt = 4 * nthread * n1;
                if (n2 == 2 * nthread) {
                    nt >>= 1;
                } else if (n2 < 2 * nthread) {
                    nt >>= 2;
                }
                t = new float[nt];
                oldNthread = nthread;
            }
            if ((nthread > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (int i = 0; i < n1; i++) {
                    dctn2.inverse(a, i * n2, scale);
                }
            }
        } else {
            if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread)) {
                Future[] futures = new Future[nthread];
                int p = n1 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthread - 1) {
                        stopRow = n1;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int i = startRow; i < stopRow; i++) {
                                dctn2.inverse(a, i * n2, scale);
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                p = n2 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startCol = l * p;
                    final int stopCol;
                    if (l == nthread - 1) {
                        stopCol = n2;
                    } else {
                        stopCol = startCol + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            float[] temp = new float[n1];
                            for (int c = startCol; c < stopCol; c++) {
                                for (int r = 0; r < n1; r++) {
                                    temp[r] = a[r * n2 + c];
                                }
                                dctn1.inverse(temp, scale);
                                for (int r = 0; r < n1; r++) {
                                    a[r * n2 + c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                for (int i = 0; i < n1; i++) {
                    dctn2.inverse(a, i * n2, scale);
                }
                float[] temp = new float[n1];
                for (int c = 0; c < n2; c++) {
                    for (int r = 0; r < n1; r++) {
                        temp[r] = a[r * n2 + c];
                    }
                    dctn1.inverse(temp, scale);
                    for (int r = 0; r < n1; r++) {
                        a[r * n2 + c] = temp[r];
                    }
                }
            }
        }
    }

    /**
     * Computes 2D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     * The data is stored in 2D array.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void inverse(final float[][] a, final boolean scale) {
        int nthread = ConcurrencyUtils.getNumberOfProcessors();
        if (isPowerOfTwo) {
            if (nthread != oldNthread) {
                nt = 4 * nthread * n1;
                if (n2 == 2 * nthread) {
                    nt >>= 1;
                } else if (n2 < 2 * nthread) {
                    nt >>= 2;
                }
                t = new float[nt];
                oldNthread = nthread;
            }
            if ((nthread > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (int i = 0; i < n1; i++) {
                    dctn2.inverse(a[i], scale);
                }
            }
        } else {
            if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread)) {
                Future[] futures = new Future[nthread];
                int p = n1 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthread - 1) {
                        stopRow = n1;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            for (int i = startRow; i < stopRow; i++) {
                                dctn2.inverse(a[i], scale);
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                p = n2 / nthread;
                for (int l = 0; l < nthread; l++) {
                    final int startCol = l * p;
                    final int stopCol;
                    if (l == nthread - 1) {
                        stopCol = n2;
                    } else {
                        stopCol = startCol + p;
                    }
                    futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                        public void run() {
                            float[] temp = new float[n1];
                            for (int c = startCol; c < stopCol; c++) {
                                for (int r = 0; r < n1; r++) {
                                    temp[r] = a[r][c];
                                }
                                dctn1.inverse(temp, scale);
                                for (int r = 0; r < n1; r++) {
                                    a[r][c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    for (int l = 0; l < nthread; l++) {
                        futures[l].get();
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                for (int i = 0; i < n1; i++) {
                    dctn2.inverse(a[i], scale);
                }
                float[] temp = new float[n1];
                for (int c = 0; c < n2; c++) {
                    for (int r = 0; r < n1; r++) {
                        temp[r] = a[r][c];
                    }
                    dctn1.inverse(temp, scale);
                    for (int r = 0; r < n1; r++) {
                        a[r][c] = temp[r];
                    }
                }
            }
        }
    }

    /* -------- child routines -------- */

    private void ddxt2d_subth(final int isgn, final float[] a, final boolean scale) {
        int nthread, nt, i;

        int np = ConcurrencyUtils.getNumberOfProcessors();
        nthread = np;
        nt = 4 * n1;
        if (n2 == 2 * np) {
            nt >>= 1;
        } else if (n2 < 2 * np) {
            nthread = n2;
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];

        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j, idx1, idx2;
                    if (n2 > 2 * nthread_f) {
                        if (isgn == -1) {
                            for (j = 4 * n0; j < n2; j += 4 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * n2 + j;
                                    idx2 = startt + n1 + i;
                                    t[startt + i] = a[idx1];
                                    t[idx2] = a[idx1 + 1];
                                    t[idx2 + n1] = a[idx1 + 2];
                                    t[idx2 + 2 * n1] = a[idx1 + 3];
                                }
                                dctn1.forward(t, startt, scale);
                                dctn1.forward(t, startt + n1, scale);
                                dctn1.forward(t, startt + 2 * n1, scale);
                                dctn1.forward(t, startt + 3 * n1, scale);
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * n2 + j;
                                    idx2 = startt + n1 + i;
                                    a[idx1] = t[startt + i];
                                    a[idx1 + 1] = t[idx2];
                                    a[idx1 + 2] = t[idx2 + n1];
                                    a[idx1 + 3] = t[idx2 + 2 * n1];
                                }
                            }
                        } else {
                            for (j = 4 * n0; j < n2; j += 4 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * n2 + j;
                                    idx2 = startt + n1 + i;
                                    t[startt + i] = a[idx1];
                                    t[idx2] = a[idx1 + 1];
                                    t[idx2 + n1] = a[idx1 + 2];
                                    t[idx2 + 2 * n1] = a[idx1 + 3];
                                }
                                dctn1.inverse(t, startt, scale);
                                dctn1.inverse(t, startt + n1, scale);
                                dctn1.inverse(t, startt + 2 * n1, scale);
                                dctn1.inverse(t, startt + 3 * n1, scale);
                                for (i = 0; i < n1; i++) {
                                    idx1 = i * n2 + j;
                                    idx2 = startt + n1 + i;
                                    a[idx1] = t[startt + i];
                                    a[idx1 + 1] = t[idx2];
                                    a[idx1 + 2] = t[idx2 + n1];
                                    a[idx1 + 3] = t[idx2 + 2 * n1];
                                }
                            }
                        }
                    } else if (n2 == 2 * nthread_f) {
                        for (i = 0; i < n1; i++) {
                            idx1 = i * n2 + 2 * n0;
                            idx2 = startt + i;
                            t[idx2] = a[idx1];
                            t[idx2 + n1] = a[idx1 + 1];
                        }
                        if (isgn == -1) {
                            dctn1.forward(t, startt, scale);
                            dctn1.forward(t, startt + n1, scale);
                        } else {
                            dctn1.inverse(t, startt, scale);
                            dctn1.inverse(t, startt + n1, scale);
                        }
                        for (i = 0; i < n1; i++) {
                            idx1 = i * n2 + 2 * n0;
                            idx2 = startt + i;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + n1];
                        }
                    } else if (n2 == nthread_f) {
                        for (i = 0; i < n1; i++) {
                            t[startt + i] = a[i * n2 + n0];
                        }
                        if (isgn == -1) {
                            dctn1.forward(t, startt, scale);
                        } else {
                            dctn1.inverse(t, startt, scale);
                        }
                        for (i = 0; i < n1; i++) {
                            a[i * n2 + n0] = t[startt + i];
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

    private void ddxt2d_subth(final int isgn, final float[][] a, final boolean scale) {
        int nthread, nt, i;

        int np = ConcurrencyUtils.getNumberOfProcessors();
        nthread = np;
        nt = 4 * n1;
        if (n2 == 2 * np) {
            nt >>= 1;
        } else if (n2 < 2 * np) {
            nthread = n2;
            nt >>= 2;
        }
        final int nthread_f = nthread;
        Future[] futures = new Future[nthread];

        for (i = 0; i < nthread; i++) {
            final int n0 = i;
            final int startt = nt * i;
            futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    int i, j, idx2;
                    if (n2 > 2 * nthread_f) {
                        if (isgn == -1) {
                            for (j = 4 * n0; j < n2; j += 4 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + n1 + i;
                                    t[startt + i] = a[i][j];
                                    t[idx2] = a[i][j + 1];
                                    t[idx2 + n1] = a[i][j + 2];
                                    t[idx2 + 2 * n1] = a[i][j + 3];
                                }
                                dctn1.forward(t, startt, scale);
                                dctn1.forward(t, startt + n1, scale);
                                dctn1.forward(t, startt + 2 * n1, scale);
                                dctn1.forward(t, startt + 3 * n1, scale);
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + n1 + i;
                                    a[i][j] = t[startt + i];
                                    a[i][j + 1] = t[idx2];
                                    a[i][j + 2] = t[idx2 + n1];
                                    a[i][j + 3] = t[idx2 + 2 * n1];
                                }
                            }
                        } else {
                            for (j = 4 * n0; j < n2; j += 4 * nthread_f) {
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + n1 + i;
                                    t[startt + i] = a[i][j];
                                    t[idx2] = a[i][j + 1];
                                    t[idx2 + n1] = a[i][j + 2];
                                    t[idx2 + 2 * n1] = a[i][j + 3];
                                }
                                dctn1.inverse(t, startt, scale);
                                dctn1.inverse(t, startt + n1, scale);
                                dctn1.inverse(t, startt + 2 * n1, scale);
                                dctn1.inverse(t, startt + 3 * n1, scale);
                                for (i = 0; i < n1; i++) {
                                    idx2 = startt + n1 + i;
                                    a[i][j] = t[startt + i];
                                    a[i][j + 1] = t[idx2];
                                    a[i][j + 2] = t[idx2 + n1];
                                    a[i][j + 3] = t[idx2 + 2 * n1];
                                }
                            }
                        }
                    } else if (n2 == 2 * nthread_f) {
                        for (i = 0; i < n1; i++) {
                            idx2 = startt + i;
                            t[idx2] = a[i][2 * n0];
                            t[idx2 + n1] = a[i][2 * n0 + 1];
                        }
                        if (isgn == -1) {
                            dctn1.forward(t, startt, scale);
                            dctn1.forward(t, startt + n1, scale);
                        } else {
                            dctn1.inverse(t, startt, scale);
                            dctn1.inverse(t, startt + n1, scale);
                        }
                        for (i = 0; i < n1; i++) {
                            idx2 = startt + i;
                            a[i][2 * n0] = t[idx2];
                            a[i][2 * n0 + 1] = t[idx2 + n1];
                        }
                    } else if (n2 == nthread_f) {
                        for (i = 0; i < n1; i++) {
                            t[startt + i] = a[i][n0];
                        }
                        if (isgn == -1) {
                            dctn1.forward(t, startt, scale);
                        } else {
                            dctn1.inverse(t, startt, scale);
                        }
                        for (i = 0; i < n1; i++) {
                            a[i][n0] = t[startt + i];
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

    private void ddxt2d0_subth(final int isgn, final float[] a, final boolean scale) {
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
                    if (isgn == -1) {
                        for (int i = n0; i < n1; i += nthread) {
                            dctn2.forward(a, i * n2, scale);
                        }
                    } else {
                        for (int i = n0; i < n1; i += nthread) {
                            dctn2.inverse(a, i * n2, scale);
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

    private void ddxt2d0_subth(final int isgn, final float[][] a, final boolean scale) {
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
                    if (isgn == -1) {
                        for (int i = n0; i < n1; i += nthread) {
                            dctn2.forward(a[i], scale);
                        }
                    } else {
                        for (int i = n0; i < n1; i += nthread) {
                            dctn2.inverse(a[i], scale);
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

    private void ddxt2d_sub(int isgn, float[] a, boolean scale) {
        int i, j, idx1, idx2;

        if (n2 > 2) {
            if (isgn == -1) {
                for (j = 0; j < n2; j += 4) {
                    for (i = 0; i < n1; i++) {
                        idx1 = i * n2 + j;
                        idx2 = n1 + i;
                        t[i] = a[idx1];
                        t[idx2] = a[idx1 + 1];
                        t[idx2 + n1] = a[idx1 + 2];
                        t[idx2 + 2 * n1] = a[idx1 + 3];
                    }
                    dctn1.forward(t, 0, scale);
                    dctn1.forward(t, n1, scale);
                    dctn1.forward(t, 2 * n1, scale);
                    dctn1.forward(t, 3 * n1, scale);
                    for (i = 0; i < n1; i++) {
                        idx1 = i * n2 + j;
                        idx2 = n1 + i;
                        a[idx1] = t[i];
                        a[idx1 + 1] = t[idx2];
                        a[idx1 + 2] = t[idx2 + n1];
                        a[idx1 + 3] = t[idx2 + 2 * n1];
                    }
                }
            } else {
                for (j = 0; j < n2; j += 4) {
                    for (i = 0; i < n1; i++) {
                        idx1 = i * n2 + j;
                        idx2 = n1 + i;
                        t[i] = a[idx1];
                        t[idx2] = a[idx1 + 1];
                        t[idx2 + n1] = a[idx1 + 2];
                        t[idx2 + 2 * n1] = a[idx1 + 3];
                    }
                    dctn1.inverse(t, 0, scale);
                    dctn1.inverse(t, n1, scale);
                    dctn1.inverse(t, 2 * n1, scale);
                    dctn1.inverse(t, 3 * n1, scale);
                    for (i = 0; i < n1; i++) {
                        idx1 = i * n2 + j;
                        idx2 = n1 + i;
                        a[idx1] = t[i];
                        a[idx1 + 1] = t[idx2];
                        a[idx1 + 2] = t[idx2 + n1];
                        a[idx1 + 3] = t[idx2 + 2 * n1];
                    }
                }
            }
        } else if (n2 == 2) {
            for (i = 0; i < n1; i++) {
                idx1 = i * n2;
                t[i] = a[idx1];
                t[n1 + i] = a[idx1 + 1];
            }
            if (isgn == -1) {
                dctn1.forward(t, 0, scale);
                dctn1.forward(t, n1, scale);
            } else {
                dctn1.inverse(t, 0, scale);
                dctn1.inverse(t, n1, scale);
            }
            for (i = 0; i < n1; i++) {
                idx1 = i * n2;
                a[idx1] = t[i];
                a[idx1 + 1] = t[n1 + i];
            }
        }
    }

    private void ddxt2d_sub(int isgn, float[][] a, boolean scale) {
        int i, j, idx2;

        if (n2 > 2) {
            if (isgn == -1) {
                for (j = 0; j < n2; j += 4) {
                    for (i = 0; i < n1; i++) {
                        idx2 = n1 + i;
                        t[i] = a[i][j];
                        t[idx2] = a[i][j + 1];
                        t[idx2 + n1] = a[i][j + 2];
                        t[idx2 + 2 * n1] = a[i][j + 3];
                    }
                    dctn1.forward(t, 0, scale);
                    dctn1.forward(t, n1, scale);
                    dctn1.forward(t, 2 * n1, scale);
                    dctn1.forward(t, 3 * n1, scale);
                    for (i = 0; i < n1; i++) {
                        idx2 = n1 + i;
                        a[i][j] = t[i];
                        a[i][j + 1] = t[idx2];
                        a[i][j + 2] = t[idx2 + n1];
                        a[i][j + 3] = t[idx2 + 2 * n1];
                    }
                }
            } else {
                for (j = 0; j < n2; j += 4) {
                    for (i = 0; i < n1; i++) {
                        idx2 = n1 + i;
                        t[i] = a[i][j];
                        t[idx2] = a[i][j + 1];
                        t[idx2 + n1] = a[i][j + 2];
                        t[idx2 + 2 * n1] = a[i][j + 3];
                    }
                    dctn1.inverse(t, 0, scale);
                    dctn1.inverse(t, n1, scale);
                    dctn1.inverse(t, 2 * n1, scale);
                    dctn1.inverse(t, 3 * n1, scale);
                    for (i = 0; i < n1; i++) {
                        idx2 = n1 + i;
                        a[i][j] = t[i];
                        a[i][j + 1] = t[idx2];
                        a[i][j + 2] = t[idx2 + n1];
                        a[i][j + 3] = t[idx2 + 2 * n1];
                    }
                }
            }
        } else if (n2 == 2) {
            for (i = 0; i < n1; i++) {
                t[i] = a[i][0];
                t[n1 + i] = a[i][1];
            }
            if (isgn == -1) {
                dctn1.forward(t, 0, scale);
                dctn1.forward(t, n1, scale);
            } else {
                dctn1.inverse(t, 0, scale);
                dctn1.inverse(t, n1, scale);
            }
            for (i = 0; i < n1; i++) {
                a[i][0] = t[i];
                a[i][1] = t[n1 + i];
            }
        }
    }
}
