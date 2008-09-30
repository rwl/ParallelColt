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

package edu.emory.mathcs.jtransforms.dst;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import edu.emory.mathcs.jtransforms.dct.FloatDCT_1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Computes 1D Discrete Sine Transform (DST) of single precision data. The size
 * of data must be a power-of-two number. It uses DCT algorithm. This is a
 * parallel implementation optimized for SMP systems. <br>
 * <br>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class FloatDST_1D {

    private int n;

    private FloatDCT_1D dct;

    /**
     * Creates new instance of FloatDST_1D.
     * 
     * @param n
     *            size of data - must be a power-of-two number
     */
    public FloatDST_1D(int n) {
        this.n = n;
        dct = new FloatDCT_1D(n);
    }

    /**
     * Creates new instance of FloatDST_1D.
     * 
     * @param n
     *            size of data - must be a power-of-two number
     * @param ip
     *            work area for bit reversal, length >=
     *            2+(1<<(int)(log(n/2+0.5)/log(2))/2)
     * @param w
     *            cos/sin table, length = n*5/4
     */
    public FloatDST_1D(int n, int[] ip, float[] w) {
        this.n = n;
        dct = new FloatDCT_1D(n, ip, w);
    }

    /**
     * Computes 1D forward DST (DST-II) leaving the result in <code>a</code>.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void forward(float[] a, boolean scale) {
        forward(a, 0, scale);
    }

    /**
     * Computes 1D forward DST (DST-II) leaving the result in <code>a</code>.
     * 
     * @param a
     *            data to transform
     * @param offa
     *            index of the first element in array <code>a</code>
     * @param scale
     *            if true then scaling is performed
     */
    public void forward(final float[] a, final int offa, boolean scale) {
        if (n == 1)
            return;
        float tmp;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (n > ConcurrencyUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final int k = n / np;
            Future[] futures = new Future[np];
            for (int j = 0; j < np; j++) {
                final int loc_offa = offa + j * k + 1;
                final int loc_stopa;
                if (j == np - 1) {
                    loc_stopa = n;
                } else {
                    loc_stopa = loc_offa + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        if (n >= 4) {
                            for (int i = loc_offa; i < loc_stopa; i += 4) {
                                a[i] = -a[i];
                                a[i + 2] = -a[i + 2];
                            }
                        } else {
                            for (int i = loc_offa; i < loc_stopa; i += 2) {
                                a[i] = -a[i];
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
        } else {
            int startidx = 1 + offa;
            int stopidx = offa + n;
            if (n >= 4) {
                for (int i = startidx; i < stopidx; i += 4) {
                    a[i] = -a[i];
                    a[i + 2] = -a[i + 2];
                }
            } else {
                for (int i = startidx; i < stopidx; i += 2) {
                    a[i] = -a[i];
                }
            }
        }
        dct.forward(a, offa, scale);
        if ((np > 1) && (n > ConcurrencyUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final int k = n / 2 / np;
            Future[] futures = new Future[np];
            for (int j = 0; j < np; j++) {
                final int loc_offa = offa + j * k;
                final int loc_stopa;
                if (j == np - 1) {
                    loc_stopa = n / 2;
                } else {
                    loc_stopa = loc_offa + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        float tmp;
                        int idx0 = offa + n - 1;
                        int idx1;
                        if (n / 2 >= 4) {
                            for (int i = loc_offa; i < loc_stopa; i += 4) {
                                tmp = a[i];
                                idx1 = idx0 - i;
                                a[i] = a[idx1];
                                a[idx1] = tmp;
                                tmp = a[i + 1];
                                idx1 = idx0 - i - 1;
                                a[i + 1] = a[idx1];
                                a[idx1] = tmp;
                                tmp = a[i + 2];
                                idx1 = idx0 - i - 2;
                                a[i + 2] = a[idx1];
                                a[idx1] = tmp;
                                tmp = a[i + 3];
                                idx1 = idx0 - i - 3;
                                a[i + 3] = a[idx1];
                                a[idx1] = tmp;
                            }
                        } else {
                            for (int i = loc_offa; i < loc_stopa; i++) {
                                tmp = a[i];
                                idx1 = idx0 - i;
                                a[i] = a[idx1];
                                a[idx1] = tmp;
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
        } else {
            int idx0 = offa + n - 1;
            if (n / 2 >= 4) {
                for (int i = 0; i < n / 2; i += 4) {
                    tmp = a[offa + i];
                    a[offa + i] = a[idx0 - i];
                    a[idx0 - i] = tmp;
                    tmp = a[offa + i + 1];
                    a[offa + i + 1] = a[idx0 - i - 1];
                    a[idx0 - i - 1] = tmp;
                    tmp = a[offa + i + 2];
                    a[offa + i + 2] = a[idx0 - i - 2];
                    a[idx0 - i - 2] = tmp;
                    tmp = a[offa + i + 3];
                    a[offa + i + 3] = a[idx0 - i - 3];
                    a[idx0 - i - 3] = tmp;
                }
            } else {
                for (int i = 0; i < n / 2; i++) {
                    tmp = a[offa + i];
                    a[offa + i] = a[idx0 - i];
                    a[idx0 - i] = tmp;
                }
            }
        }
    }

    /**
     * Computes 1D inverse DST (DST-III) leaving the result in <code>a</code>.
     * 
     * @param a
     *            data to transform
     * @param scale
     *            if true then scaling is performed
     */
    public void inverse(float[] a, boolean scale) {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D inverse DST (DST-III) leaving the result in <code>a</code>.
     * 
     * @param a
     *            data to transform
     * @param offa
     *            index of the first element in array <code>a</code>
     * @param scale
     *            if true then scaling is performed
     */
    public void inverse(final float[] a, final int offa, boolean scale) {
        if (n == 1)
            return;
        float tmp;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (n > ConcurrencyUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final int k = n / 2 / np;
            Future[] futures = new Future[np];
            for (int j = 0; j < np; j++) {
                final int loc_offa = offa + j * k;
                final int loc_stopa;
                if (j == np - 1) {
                    loc_stopa = n / 2;
                } else {
                    loc_stopa = loc_offa + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        float tmp;
                        int idx0 = offa + n - 1;
                        int idx1;
                        if (n / 2 >= 4) {
                            for (int i = loc_offa; i < loc_stopa; i += 4) {
                                tmp = a[i];
                                idx1 = idx0 - i;
                                a[i] = a[idx1];
                                a[idx1] = tmp;
                                tmp = a[i + 1];
                                idx1 = idx0 - i - 1;
                                a[i + 1] = a[idx1];
                                a[idx1] = tmp;
                                tmp = a[i + 2];
                                idx1 = idx0 - i - 2;
                                a[i + 2] = a[idx1];
                                a[idx1] = tmp;
                                tmp = a[i + 3];
                                idx1 = idx0 - i - 3;
                                a[i + 3] = a[idx1];
                                a[idx1] = tmp;
                            }
                        } else {
                            for (int i = loc_offa; i < loc_stopa; i++) {
                                tmp = a[i];
                                idx1 = idx0 - i;
                                a[i] = a[idx1];
                                a[idx1] = tmp;
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
        } else {
            int idx0 = offa + n - 1;
            if (n / 2 >= 4) {
                for (int i = 0; i < n / 2; i += 4) {
                    tmp = a[offa + i];
                    a[offa + i] = a[idx0 - i];
                    a[idx0 - i] = tmp;
                    tmp = a[offa + i + 1];
                    a[offa + i + 1] = a[idx0 - i - 1];
                    a[idx0 - i - 1] = tmp;
                    tmp = a[offa + i + 2];
                    a[offa + i + 2] = a[idx0 - i - 2];
                    a[idx0 - i - 2] = tmp;
                    tmp = a[offa + i + 3];
                    a[offa + i + 3] = a[idx0 - i - 3];
                    a[idx0 - i - 3] = tmp;
                }
            } else {
                for (int i = 0; i < n / 2; i++) {
                    tmp = a[offa + i];
                    a[offa + i] = a[idx0 - i];
                    a[idx0 - i] = tmp;
                }
            }
        }
        dct.inverse(a, offa, scale);
        if ((np > 1) && (n > ConcurrencyUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final int k = n / np;
            Future[] futures = new Future[np];
            for (int j = 0; j < np; j++) {
                final int loc_offa = offa + j * k + 1;
                final int loc_stopa;
                if (j == np - 1) {
                    loc_stopa = n;
                } else {
                    loc_stopa = loc_offa + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                    public void run() {
                        if (n >= 4) {
                            for (int i = loc_offa; i < loc_stopa; i += 4) {
                                a[i] = -a[i];
                                a[i + 2] = -a[i + 2];
                            }
                        } else {
                            for (int i = loc_offa; i < loc_stopa; i += 2) {
                                a[i] = -a[i];
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
        } else {
            int startidx = 1 + offa;
            int stopidx = offa + n;
            if (n >= 4) {
                for (int i = startidx; i < stopidx; i += 4) {
                    a[i] = -a[i];
                    a[i + 2] = -a[i + 2];
                }
            } else {
                for (int i = startidx; i < stopidx; i += 2) {
                    a[i] = -a[i];
                }
            }
        }
    }
}
