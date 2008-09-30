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

package edu.emory.mathcs.jtransforms.dht;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Computes 1D Discrete Hartley Transform (DHT) of real, single precision data.
 * The size of data must be a power-of-two number. It uses FFT algorithm. This
 * is a parallel implementation optimized for SMP systems.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FloatDHT_1D {
    private int n;
    private FloatFFT_1D fft;

    /**
     * Creates new instance of FloatDHT_1D.
     * 
     * @param n
     *            size of data - must be a power-of-two number
     */
    public FloatDHT_1D(int n) {
        this.n = n;
        fft = new FloatFFT_1D(n);
    }

    /**
     * Creates new instance of FloatDHT_1D.
     * 
     * @param n
     *            size of data - must be a power-of-two number
     * @param ip
     *            work area for bit reversal, length >=
     *            2+(1<<(int)(log(n+0.5)/log(2))/2)
     * @param w
     *            cos/sin table, length = n/2
     */
    public FloatDHT_1D(int n, int[] ip, float[] w) {

        this.n = n;
        fft = new FloatFFT_1D(n, ip, w);
    }

    /**
     * Computes 1D real, forward DHT leaving the result in <code>a</code>.
     * 
     * @param a
     *            data to transform
     */
    public void forward(float[] a) {
        forward(a, 0);
    }

    /**
     * Computes 1D real, forward DHT leaving the result in <code>a</code>.
     * 
     * @param a
     *            data to transform
     * @param offa
     *            index of the first element in array <code>a</code>
     */
    public void forward(final float[] a, final int offa) {
        if (n == 1)
            return;
        fft.realForward(a, offa);
        final float[] b = new float[n];
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (n > ConcurrencyUtils.getThreadsBeginN_1D_FFT_4Threads())) {
        	Future[] futures = new Future[np];
			int k = n / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int length;
				if (j == np - 1) {
					length = n - startidx;
				} else {
					length = k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						System.arraycopy(a, offa + startidx, b, startidx, length);
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
        else {
        	System.arraycopy(a, offa, b, 0, n);
        }
        int nd2 = n / 2;
        if ((np > 1) && (nd2 > ConcurrencyUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final int k1 = nd2 / np;
            Future[] futures = new Future[np];
            for (int i = 0; i < np; i++) {
                final int startidx = 1 + i * k1;
                final int stopidx;
                if (i == np - 1) {
                    stopidx = nd2;
                } else {
                    stopidx = startidx + k1;
                }
                futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        int idx1, idx2;
                        for (int i = startidx; i < stopidx; i++) {
                            idx1 = 2 * i;
                            idx2 = idx1 + 1;
                            a[offa + i] = b[idx1] - b[idx2];
                            a[offa + n - i] = b[idx1] + b[idx2];
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
            int idx1, idx2;
            for (int i = 1; i < nd2; i++) {
                idx1 = 2 * i;
                idx2 = idx1 + 1;
                a[offa + i] = b[idx1] - b[idx2];
                a[offa + n - i] = b[idx1] + b[idx2];
            }
        }
        a[offa + nd2] = b[1];

    }

    /**
     * Computes 1D real, inverse DHT leaving the result in <code>a</code>.
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
     * Computes 1D real, inverse DHT leaving the result in <code>a</code>.
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
        forward(a, offa);
        if (scale) {
            scale(n, a, offa, false);
        }
    }

    private void scale(final float m, final float[] a, int offa, boolean complex) {
        int locn;
        if (complex) {
            locn = 2 * n;
        } else {
            locn = n;
        }
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (locn >= ConcurrencyUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final int k = locn / np;
            Future[] futures = new Future[np];
            for (int i = 0; i < np; i++) {
                final int idx1 = offa + i * k;
                futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int i = idx1; i < idx1 + k; i += 2) {
                            a[i] /= m;
                            a[i + 1] /= m;
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
            for (int i = offa; i < offa + locn; i += 2) {
                a[i] /= m;
                a[i + 1] /= m;
            }

        }
    }
}
