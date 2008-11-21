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
 * Computes 2D Discrete Fourier Transform (DFT) of complex and real, double
 * precision data. The sizes of both dimensions can be arbitrary numbers. This
 * is a parallel implementation of split-radix and mixed-radix algorithms
 * optimized for SMP systems. <br>
 * <br>
 * This code is derived from General Purpose FFT Package written by Takuya Ooura
 * (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DoubleFFT_2D {

	private int n1;

	private int n2;

	private double[] t;

	private DoubleFFT_1D fftn2, fftn1;

	private int oldNthread;

	private int nt;

	private boolean isPowerOfTwo = false;

	private boolean useThreads = false;

	/**
	 * Creates new instance of DoubleFFT_2D.
	 * 
	 * @param n1
	 *            number of rows
	 * @param n2
	 *            number of columns
	 */
	public DoubleFFT_2D(int n1, int n2) {
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
			nt = 8 * oldNthread * n1;
			if (2 * n2 == 4 * oldNthread) {
				nt >>= 1;
			} else if (2 * n2 < 4 * oldNthread) {
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
	}

	/**
	 * Computes 2D forward DFT of complex data leaving the result in
	 * <code>a</code>. The data is stored in 1D array in row-major order.
	 * Complex number is stored as two double values in sequence: the real and
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
	public void complexForward(final double[] a) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			int oldn2 = n2;
			n2 = 2 * n2;
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(0, -1, a, true);
				cdft2d_subth(-1, a, true);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.complexForward(a, i * n2);
				}
				cdft2d_sub(-1, a, true);
			}
			n2 = oldn2;
		} else {
			final int rowStride = 2 * n2;
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
								fftn2.complexForward(a, i * rowStride);
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
							double[] temp = new double[2 * n1];
							for (int c = startCol; c < stopCol; c++) {
								int idx0 = 2 * c;
								for (int r = 0; r < n1; r++) {
									int idx1 = 2 * r;
									int idx2 = r * rowStride + idx0;
									temp[idx1] = a[idx2];
									temp[idx1 + 1] = a[idx2 + 1];
								}
								fftn1.complexForward(temp);
								for (int r = 0; r < n1; r++) {
									int idx1 = 2 * r;
									int idx2 = r * rowStride + idx0;
									a[idx2] = temp[idx1];
									a[idx2 + 1] = temp[idx1 + 1];
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
					fftn2.complexForward(a, i * rowStride);
				}
				double[] temp = new double[2 * n1];
				for (int c = 0; c < n2; c++) {
					int idx0 = 2 * c;
					for (int r = 0; r < n1; r++) {
						int idx1 = 2 * r;
						int idx2 = r * rowStride + idx0;
						temp[idx1] = a[idx2];
						temp[idx1 + 1] = a[idx2 + 1];
					}
					fftn1.complexForward(temp);
					for (int r = 0; r < n1; r++) {
						int idx1 = 2 * r;
						int idx2 = r * rowStride + idx0;
						a[idx2] = temp[idx1];
						a[idx2 + 1] = temp[idx1 + 1];
					}
				}
			}
		}
	}

	/**
	 * Computes 2D forward DFT of complex data leaving the result in
	 * <code>a</code>. The data is stored in 2D array. Complex data is
	 * represented by 2 double values in sequence: the real and imaginary part,
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
	public void complexForward(final double[][] a) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			int oldn2 = n2;
			n2 = 2 * n2;
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(0, -1, a, true);
				cdft2d_subth(-1, a, true);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.complexForward(a[i]);
				}
				cdft2d_sub(-1, a, true);
			}
			n2 = oldn2;
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
								fftn2.complexForward(a[i]);
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
							double[] temp = new double[2 * n1];
							for (int c = startCol; c < stopCol; c++) {
								int idx1 = 2 * c;
								for (int r = 0; r < n1; r++) {
									int idx2 = 2 * r;
									temp[idx2] = a[r][idx1];
									temp[idx2 + 1] = a[r][idx1 + 1];
								}
								fftn1.complexForward(temp);
								for (int r = 0; r < n1; r++) {
									int idx2 = 2 * r;
									a[r][idx1] = temp[idx2];
									a[r][idx1 + 1] = temp[idx2 + 1];
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
					fftn2.complexForward(a[i]);
				}
				double[] temp = new double[2 * n1];
				for (int c = 0; c < n2; c++) {
					int idx1 = 2 * c;
					for (int r = 0; r < n1; r++) {
						int idx2 = 2 * r;
						temp[idx2] = a[r][idx1];
						temp[idx2 + 1] = a[r][idx1 + 1];
					}
					fftn1.complexForward(temp);
					for (int r = 0; r < n1; r++) {
						int idx2 = 2 * r;
						a[r][idx1] = temp[idx2];
						a[r][idx1 + 1] = temp[idx2 + 1];
					}
				}
			}
		}
	}

	/**
	 * Computes 2D inverse DFT of complex data leaving the result in
	 * <code>a</code>. The data is stored in 1D array in row-major order.
	 * Complex number is stored as two double values in sequence: the real and
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
	public void complexInverse(final double[] a, final boolean scale) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			int oldn2 = n2;
			n2 = 2 * n2;
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(0, 1, a, scale);
				cdft2d_subth(1, a, scale);
			} else {

				for (int i = 0; i < n1; i++) {
					fftn2.complexInverse(a, i * n2, scale);
				}
				cdft2d_sub(1, a, scale);
			}
			n2 = oldn2;
		} else {
			final int rowspan = 2 * n2;
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
								fftn2.complexInverse(a, i * rowspan, scale);
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
							double[] temp = new double[2 * n1];
							for (int c = startCol; c < stopCol; c++) {
								int idx1 = 2 * c;
								for (int r = 0; r < n1; r++) {
									int idx2 = 2 * r;
									int idx3 = r * rowspan + idx1;
									temp[idx2] = a[idx3];
									temp[idx2 + 1] = a[idx3 + 1];
								}
								fftn1.complexInverse(temp, scale);
								for (int r = 0; r < n1; r++) {
									int idx2 = 2 * r;
									int idx3 = r * rowspan + idx1;
									a[idx3] = temp[idx2];
									a[idx3 + 1] = temp[idx2 + 1];
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
					fftn2.complexInverse(a, i * rowspan, scale);
				}
				double[] temp = new double[2 * n1];
				for (int c = 0; c < n2; c++) {
					int idx1 = 2 * c;
					for (int r = 0; r < n1; r++) {
						int idx2 = 2 * r;
						int idx3 = r * rowspan + idx1;
						temp[idx2] = a[idx3];
						temp[idx2 + 1] = a[idx3 + 1];
					}
					fftn1.complexInverse(temp, scale);
					for (int r = 0; r < n1; r++) {
						int idx2 = 2 * r;
						int idx3 = r * rowspan + idx1;
						a[idx3] = temp[idx2];
						a[idx3 + 1] = temp[idx2 + 1];
					}
				}
			}
		}
	}

	/**
	 * Computes 2D inverse DFT of complex data leaving the result in
	 * <code>a</code>. The data is stored in 2D array. Complex data is
	 * represented by 2 double values in sequence: the real and imaginary part,
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
	public void complexInverse(final double[][] a, final boolean scale) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			int oldn2 = n2;
			n2 = 2 * n2;

			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(0, 1, a, scale);
				cdft2d_subth(1, a, scale);
			} else {

				for (int i = 0; i < n1; i++) {
					fftn2.complexInverse(a[i], scale);
				}
				cdft2d_sub(1, a, scale);
			}
			n2 = oldn2;
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
								fftn2.complexInverse(a[i], scale);
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
							double[] temp = new double[2 * n1];
							for (int c = startCol; c < stopCol; c++) {
								int idx1 = 2 * c;
								for (int r = 0; r < n1; r++) {
									int idx2 = 2 * r;
									temp[idx2] = a[r][idx1];
									temp[idx2 + 1] = a[r][idx1 + 1];
								}
								fftn1.complexInverse(temp, scale);
								for (int r = 0; r < n1; r++) {
									int idx2 = 2 * r;
									a[r][idx1] = temp[idx2];
									a[r][idx1 + 1] = temp[idx2 + 1];
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
					fftn2.complexInverse(a[i], scale);
				}
				double[] temp = new double[2 * n1];
				for (int c = 0; c < n2; c++) {
					int idx1 = 2 * c;
					for (int r = 0; r < n1; r++) {
						int idx2 = 2 * r;
						temp[idx2] = a[r][idx1];
						temp[idx2 + 1] = a[r][idx1 + 1];
					}
					fftn1.complexInverse(temp, scale);
					for (int r = 0; r < n1; r++) {
						int idx2 = 2 * r;
						a[r][idx1] = temp[idx2];
						a[r][idx1 + 1] = temp[idx2 + 1];
					}
				}
			}
		}
	}

	/**
	 * Computes 2D forward DFT of real data leaving the result in <code>a</code>
	 * . This method only works when the sizes of both dimensions are
	 * power-of-two numbers. The physical layout of the output data is as
	 * follows:
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
	public void realForward(double[] a) {
		if (isPowerOfTwo == false) {
			throw new IllegalArgumentException("n1 and n2 must be power of two numbers");
		} else {
			int nthread;

			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(1, 1, a, true);
				cdft2d_subth(-1, a, true);
				rdft2d_sub(1, a);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.realForward(a, i * n2);
				}
				cdft2d_sub(-1, a, true);
				rdft2d_sub(1, a);
			}
		}
	}

	/**
	 * Computes 2D forward DFT of real data leaving the result in <code>a</code>
	 * . This method only works when the sizes of both dimensions are
	 * power-of-two numbers. The physical layout of the output data is as
	 * follows:
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
	public void realForward(double[][] a) {
		if (isPowerOfTwo == false) {
			throw new IllegalArgumentException("n1 and n2 must be power of two numbers");
		} else {
			int nthread;

			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(1, 1, a, true);
				cdft2d_subth(-1, a, true);
				rdft2d_sub(1, a);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.realForward(a[i]);
				}
				cdft2d_sub(-1, a, true);
				rdft2d_sub(1, a);
			}
		}
	}

	/**
	 * Computes 2D forward DFT of real data leaving the result in <code>a</code>
	 * . This method computes full real forward transform, i.e. you will get the
	 * same result as from <code>complexForward</code> called with all imaginary
	 * part equal 0. Because the result is stored in <code>a</code>, the input
	 * array must be of size n1*2*n2, with only the first n1*n2 elements filled
	 * with real data. To get back the original data, use
	 * <code>complexInverse</code> on the output of this method.
	 * 
	 * @param a
	 *            data to transform
	 */
	public void realForwardFull(double[] a) {
		if (isPowerOfTwo) {
			int nthread;

			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(1, 1, a, true);
				cdft2d_subth(-1, a, true);
				rdft2d_sub(1, a);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.realForward(a, i * n2);
				}
				cdft2d_sub(-1, a, true);
				rdft2d_sub(1, a);
			}
			fillSymmetric(a);
		} else {
			mixedRadixRealForwardFull(a);
		}
	}

	/**
	 * Computes 2D forward DFT of real data leaving the result in <code>a</code>
	 * . This method computes full real forward transform, i.e. you will get the
	 * same result as from <code>complexForward</code> called with all imaginary
	 * part equal 0. Because the result is stored in <code>a</code>, the input
	 * array must be of size n1 by 2*n2, with only the first n1 by n2 elements
	 * filled with real data. To get back the original data, use
	 * <code>complexInverse</code> on the output of this method.
	 * 
	 * @param a
	 *            data to transform
	 */
	public void realForwardFull(double[][] a) {
		if (isPowerOfTwo) {
			int nthread;

			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth1(1, 1, a, true);
				cdft2d_subth(-1, a, true);
				rdft2d_sub(1, a);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.realForward(a[i]);
				}
				cdft2d_sub(-1, a, true);
				rdft2d_sub(1, a);
			}
			fillSymmetric(a);
		} else {
			mixedRadixRealForwardFull(a);
		}
	}

	/**
	 * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
	 * . This method only works when the sizes of both dimensions are
	 * power-of-two numbers. The physical layout of the input data has to be as
	 * follows:
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
	public void realInverse(double[] a, boolean scale) {
		if (isPowerOfTwo == false) {
			throw new IllegalArgumentException("n1 and n2 must be power of two numbers");
		} else {
			int nthread;
			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				rdft2d_sub(-1, a);
				cdft2d_subth(1, a, scale);
				xdft2d0_subth1(1, -1, a, scale);
			} else {
				rdft2d_sub(-1, a);
				cdft2d_sub(1, a, scale);
				for (int i = 0; i < n1; i++) {
					fftn2.realInverse(a, i * n2, scale);
				}
			}
		}
	}

	/**
	 * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
	 * . This method only works when the sizes of both dimensions are
	 * power-of-two numbers. The physical layout of the input data has to be as
	 * follows:
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
	public void realInverse(double[][] a, boolean scale) {
		if (isPowerOfTwo == false) {
			throw new IllegalArgumentException("n1 and n2 must be power of two numbers");
		} else {
			int nthread;

			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				rdft2d_sub(-1, a);
				cdft2d_subth(1, a, scale);
				xdft2d0_subth1(1, -1, a, scale);
			} else {
				rdft2d_sub(-1, a);
				cdft2d_sub(1, a, scale);
				for (int i = 0; i < n1; i++) {
					fftn2.realInverse(a[i], scale);
				}
			}
		}
	}

	/**
	 * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
	 * . This method computes full real inverse transform, i.e. you will get the
	 * same result as from <code>complexInverse</code> called with all imaginary
	 * part equal 0. Because the result is stored in <code>a</code>, the input
	 * array must be of size n1*2*n2, with only the first n1*n2 elements filled
	 * with real data.
	 * 
	 * @param a
	 *            data to transform
	 * 
	 * @param scale
	 *            if true then scaling is performed
	 */
	public void realInverseFull(double[] a, boolean scale) {
		if (isPowerOfTwo) {
			int nthread;

			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth2(1, -1, a, scale);
				cdft2d_subth(1, a, scale);
				rdft2d_sub(1, a);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.realInverse2(a, i * n2, scale);
				}
				cdft2d_sub(1, a, scale);
				rdft2d_sub(1, a);
			}
			fillSymmetric(a);
		} else {
			mixedRadixRealInverseFull(a, scale);
		}
	}

	/**
	 * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
	 * . This method computes full real inverse transform, i.e. you will get the
	 * same result as from <code>complexInverse</code> called with all imaginary
	 * part equal 0. Because the result is stored in <code>a</code>, the input
	 * array must be of size n1 by 2*n2, with only the first n1 by n2 elements
	 * filled with real data.
	 * 
	 * @param a
	 *            data to transform
	 * 
	 * @param scale
	 *            if true then scaling is performed
	 */
	public void realInverseFull(double[][] a, boolean scale) {
		if (isPowerOfTwo) {
			int nthread;

			nthread = ConcurrencyUtils.getNumberOfProcessors();
			if (nthread != oldNthread) {
				nt = 8 * nthread * n1;
				if (n2 == 4 * nthread) {
					nt >>= 1;
				} else if (n2 < 4 * nthread) {
					nt >>= 2;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				xdft2d0_subth2(1, -1, a, scale);
				cdft2d_subth(1, a, scale);
				rdft2d_sub(1, a);
			} else {
				for (int i = 0; i < n1; i++) {
					fftn2.realInverse2(a[i], 0, scale);
				}
				cdft2d_sub(1, a, scale);
				rdft2d_sub(1, a);
			}
			fillSymmetric(a);
		} else {
			mixedRadixRealInverseFull(a, scale);
		}
	}

	/* -------- child routines -------- */

	private void mixedRadixRealForwardFull(final double[][] a) {
		final int n2d2 = n2 / 2 + 1;
		final double[][] temp = new double[n2d2][2 * n1];

		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2d2 >= nthread)) {
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
							fftn2.realForward(a[i]);
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

			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r][0]; //first column is always real
			}
			fftn1.realForwardFull(temp[0]);

			p = n2d2 / nthread;
			for (int l = 0; l < nthread; l++) {
				final int startCol = 1 + l * p;
				final int stopCol;
				if (l == nthread - 1) {
					stopCol = n2d2 - 1;
				} else {
					stopCol = startCol + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int c = startCol; c < stopCol; c++) {
							int idx2 = 2 * c;
							for (int r = 0; r < n1; r++) {
								int idx1 = 2 * r;
								temp[c][idx1] = a[r][idx2];
								temp[c][idx1 + 1] = a[r][idx2 + 1];
							}
							fftn1.complexForward(temp[c]);
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

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r][1];
					//imaginary part = 0;
				}
				fftn1.realForwardFull(temp[n2d2 - 1]);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = n2d2 - 1;
					temp[idx2][idx1] = a[r][2 * idx2];
					temp[idx2][idx1 + 1] = a[r][1];
				}
				fftn1.complexForward(temp[n2d2 - 1]);

			}

			p = n1 / nthread;
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
						for (int r = startRow; r < stopRow; r++) {
							int idx1 = 2 * r;
							for (int c = 0; c < n2d2; c++) {
								int idx2 = 2 * c;
								a[r][idx2] = temp[c][idx1];
								a[r][idx2 + 1] = temp[c][idx1 + 1];
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

			for (int l = 0; l < nthread; l++) {
				final int startRow = 1 + l * p;
				final int stopRow;
				if (l == nthread - 1) {
					stopRow = n1;
				} else {
					stopRow = startRow + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int r = startRow; r < stopRow; r++) {
							int idx3 = n1 - r;
							for (int c = n2d2; c < n2; c++) {
								int idx1 = 2 * c;
								int idx2 = 2 * (n2 - c);
								a[0][idx1] = a[0][idx2];
								a[0][idx1 + 1] = -a[0][idx2 + 1];
								a[r][idx1] = a[idx3][idx2];
								a[r][idx1 + 1] = -a[idx3][idx2 + 1];
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
				fftn2.realForward(a[i]);
			}

			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r][0]; //first column is always real
			}
			fftn1.realForwardFull(temp[0]);

			for (int c = 1; c < n2d2 - 1; c++) {
				int idx2 = 2 * c;
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					temp[c][idx1] = a[r][idx2];
					temp[c][idx1 + 1] = a[r][idx2 + 1];
				}
				fftn1.complexForward(temp[c]);
			}

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r][1];
					//imaginary part = 0;
				}
				fftn1.realForwardFull(temp[n2d2 - 1]);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = n2d2 - 1;
					temp[idx2][idx1] = a[r][2 * idx2];
					temp[idx2][idx1 + 1] = a[r][1];
				}
				fftn1.complexForward(temp[n2d2 - 1]);

			}

			for (int r = 0; r < n1; r++) {
				int idx1 = 2 * r;
				for (int c = 0; c < n2d2; c++) {
					int idx2 = 2 * c;
					a[r][idx2] = temp[c][idx1];
					a[r][idx2 + 1] = temp[c][idx1 + 1];
				}
			}

			//fill symmetric
			for (int r = 1; r < n1; r++) {
				int idx3 = n1 - r;
				for (int c = n2d2; c < n2; c++) {
					int idx1 = 2 * c;
					int idx2 = 2 * (n2 - c);
					a[0][idx1] = a[0][idx2];
					a[0][idx1 + 1] = -a[0][idx2 + 1];
					a[r][idx1] = a[idx3][idx2];
					a[r][idx1 + 1] = -a[idx3][idx2 + 1];
				}
			}
		}
	}

	private void mixedRadixRealForwardFull(final double[] a) {
		final int rowStride = 2 * n2;
		final int n2d2 = n2 / 2 + 1;
		final double[][] temp = new double[n2d2][2 * n1];

		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2d2 >= nthread)) {
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
							fftn2.realForward(a, i * n2);
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

			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r * n2]; //first column is always real
			}
			fftn1.realForwardFull(temp[0]);

			p = n2d2 / nthread;
			for (int l = 0; l < nthread; l++) {
				final int startCol = 1 + l * p;
				final int stopCol;
				if (l == nthread - 1) {
					stopCol = n2d2 - 1;
				} else {
					stopCol = startCol + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int c = startCol; c < stopCol; c++) {
							int idx0 = 2 * c;
							for (int r = 0; r < n1; r++) {
								int idx1 = 2 * r;
								int idx2 = r * n2 + idx0;
								temp[c][idx1] = a[idx2];
								temp[c][idx1 + 1] = a[idx2 + 1];
							}
							fftn1.complexForward(temp[c]);
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

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r * n2 + 1];
					//imaginary part = 0;
				}
				fftn1.realForwardFull(temp[n2d2 - 1]);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = r * n2;
					int idx3 = n2d2 - 1;
					temp[idx3][idx1] = a[idx2 + 2 * idx3];
					temp[idx3][idx1 + 1] = a[idx2 + 1];
				}
				fftn1.complexForward(temp[n2d2 - 1]);
			}

			p = n1 / nthread;
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
						for (int r = startRow; r < stopRow; r++) {
							int idx1 = 2 * r;
							for (int c = 0; c < n2d2; c++) {
								int idx0 = 2 * c;
								int idx2 = r * rowStride + idx0;
								a[idx2] = temp[c][idx1];
								a[idx2 + 1] = temp[c][idx1 + 1];
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

			for (int l = 0; l < nthread; l++) {
				final int startRow = 1 + l * p;
				final int stopRow;
				if (l == nthread - 1) {
					stopRow = n1;
				} else {
					stopRow = startRow + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int r = startRow; r < stopRow; r++) {
							int idx5 = r * rowStride;
							int idx6 = (n1 - r + 1) * rowStride;
							for (int c = n2d2; c < n2; c++) {
								int idx1 = 2 * c;
								int idx2 = 2 * (n2 - c);
								a[idx1] = a[idx2];
								a[idx1 + 1] = -a[idx2 + 1];
								int idx3 = idx5 + idx1;
								int idx4 = idx6 - idx1;
								a[idx3] = a[idx4];
								a[idx3 + 1] = -a[idx4 + 1];
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
				fftn2.realForward(a, i * n2);
			}
			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r * n2]; //first column is always real
			}
			fftn1.realForwardFull(temp[0]);

			for (int c = 1; c < n2d2 - 1; c++) {
				int idx0 = 2 * c;
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = r * n2 + idx0;
					temp[c][idx1] = a[idx2];
					temp[c][idx1 + 1] = a[idx2 + 1];
				}
				fftn1.complexForward(temp[c]);
			}

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r * n2 + 1];
					//imaginary part = 0;
				}
				fftn1.realForwardFull(temp[n2d2 - 1]);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = r * n2;
					int idx3 = n2d2 - 1;
					temp[idx3][idx1] = a[idx2 + 2 * idx3];
					temp[idx3][idx1 + 1] = a[idx2 + 1];
				}
				fftn1.complexForward(temp[n2d2 - 1]);
			}

			for (int r = 0; r < n1; r++) {
				int idx1 = 2 * r;
				for (int c = 0; c < n2d2; c++) {
					int idx0 = 2 * c;
					int idx2 = r * rowStride + idx0;
					a[idx2] = temp[c][idx1];
					a[idx2 + 1] = temp[c][idx1 + 1];
				}
			}

			//fill symmetric
			for (int r = 1; r < n1; r++) {
				int idx5 = r * rowStride;
				int idx6 = (n1 - r + 1) * rowStride;
				for (int c = n2d2; c < n2; c++) {
					int idx1 = 2 * c;
					int idx2 = 2 * (n2 - c);
					a[idx1] = a[idx2];
					a[idx1 + 1] = -a[idx2 + 1];
					int idx3 = idx5 + idx1;
					int idx4 = idx6 - idx1;
					a[idx3] = a[idx4];
					a[idx3 + 1] = -a[idx4 + 1];
				}
			}
		}
	}

	private void mixedRadixRealInverseFull(final double[][] a, final boolean scale) {
		final int n2d2 = n2 / 2 + 1;
		final double[][] temp = new double[n2d2][2 * n1];

		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2d2 >= nthread)) {
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
							fftn2.realInverse2(a[i], 0, scale);
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

			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r][0]; //first column is always real
			}
			fftn1.realInverseFull(temp[0], scale);

			p = n2d2 / nthread;
			for (int l = 0; l < nthread; l++) {
				final int startCol = 1 + l * p;
				final int stopCol;
				if (l == nthread - 1) {
					stopCol = n2d2 - 1;
				} else {
					stopCol = startCol + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int c = startCol; c < stopCol; c++) {
							int idx2 = 2 * c;
							for (int r = 0; r < n1; r++) {
								int idx1 = 2 * r;
								temp[c][idx1] = a[r][idx2];
								temp[c][idx1 + 1] = a[r][idx2 + 1];
							}
							fftn1.complexInverse(temp[c], scale);
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

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r][1];
					//imaginary part = 0;
				}
				fftn1.realInverseFull(temp[n2d2 - 1], scale);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = n2d2 - 1;
					temp[idx2][idx1] = a[r][2 * idx2];
					temp[idx2][idx1 + 1] = a[r][1];
				}
				fftn1.complexInverse(temp[n2d2 - 1], scale);

			}

			p = n1 / nthread;
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
						for (int r = startRow; r < stopRow; r++) {
							int idx1 = 2 * r;
							for (int c = 0; c < n2d2; c++) {
								int idx2 = 2 * c;
								a[r][idx2] = temp[c][idx1];
								a[r][idx2 + 1] = temp[c][idx1 + 1];
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

			for (int l = 0; l < nthread; l++) {
				final int startRow = 1 + l * p;
				final int stopRow;
				if (l == nthread - 1) {
					stopRow = n1;
				} else {
					stopRow = startRow + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int r = startRow; r < stopRow; r++) {
							int idx3 = n1 - r;
							for (int c = n2d2; c < n2; c++) {
								int idx1 = 2 * c;
								int idx2 = 2 * (n2 - c);
								a[0][idx1] = a[0][idx2];
								a[0][idx1 + 1] = -a[0][idx2 + 1];
								a[r][idx1] = a[idx3][idx2];
								a[r][idx1 + 1] = -a[idx3][idx2 + 1];
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
				fftn2.realInverse2(a[i], 0, scale);
			}

			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r][0]; //first column is always real
			}
			fftn1.realInverseFull(temp[0], scale);

			for (int c = 1; c < n2d2 - 1; c++) {
				int idx2 = 2 * c;
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					temp[c][idx1] = a[r][idx2];
					temp[c][idx1 + 1] = a[r][idx2 + 1];
				}
				fftn1.complexInverse(temp[c], scale);
			}

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r][1];
					//imaginary part = 0;
				}
				fftn1.realInverseFull(temp[n2d2 - 1], scale);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = n2d2 - 1;
					temp[idx2][idx1] = a[r][2 * idx2];
					temp[idx2][idx1 + 1] = a[r][1];
				}
				fftn1.complexInverse(temp[n2d2 - 1], scale);

			}

			for (int r = 0; r < n1; r++) {
				int idx1 = 2 * r;
				for (int c = 0; c < n2d2; c++) {
					int idx2 = 2 * c;
					a[r][idx2] = temp[c][idx1];
					a[r][idx2 + 1] = temp[c][idx1 + 1];
				}
			}

			//fill symmetric
			for (int r = 1; r < n1; r++) {
				int idx3 = n1 - r;
				for (int c = n2d2; c < n2; c++) {
					int idx1 = 2 * c;
					int idx2 = 2 * (n2 - c);
					a[0][idx1] = a[0][idx2];
					a[0][idx1 + 1] = -a[0][idx2 + 1];
					a[r][idx1] = a[idx3][idx2];
					a[r][idx1 + 1] = -a[idx3][idx2 + 1];
				}
			}
		}
	}

	private void mixedRadixRealInverseFull(final double[] a, final boolean scale) {
		final int rowStride = 2 * n2;
		final int n2d2 = n2 / 2 + 1;
		final double[][] temp = new double[n2d2][2 * n1];

		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2d2 >= nthread)) {
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
							fftn2.realInverse2(a, i * n2, scale);
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

			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r * n2]; //first column is always real
			}
			fftn1.realInverseFull(temp[0], scale);

			p = n2d2 / nthread;
			for (int l = 0; l < nthread; l++) {
				final int startCol = 1 + l * p;
				final int stopCol;
				if (l == nthread - 1) {
					stopCol = n2d2 - 1;
				} else {
					stopCol = startCol + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int c = startCol; c < stopCol; c++) {
							int idx0 = 2 * c;
							for (int r = 0; r < n1; r++) {
								int idx1 = 2 * r;
								int idx2 = r * n2 + idx0;
								temp[c][idx1] = a[idx2];
								temp[c][idx1 + 1] = a[idx2 + 1];
							}
							fftn1.complexInverse(temp[c], scale);
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

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r * n2 + 1];
					//imaginary part = 0;
				}
				fftn1.realInverseFull(temp[n2d2 - 1], scale);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = r * n2;
					int idx3 = n2d2 - 1;
					temp[idx3][idx1] = a[idx2 + 2 * idx3];
					temp[idx3][idx1 + 1] = a[idx2 + 1];
				}
				fftn1.complexInverse(temp[n2d2 - 1], scale);
			}

			p = n1 / nthread;
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
						for (int r = startRow; r < stopRow; r++) {
							int idx1 = 2 * r;
							for (int c = 0; c < n2d2; c++) {
								int idx0 = 2 * c;
								int idx2 = r * rowStride + idx0;
								a[idx2] = temp[c][idx1];
								a[idx2 + 1] = temp[c][idx1 + 1];
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

			for (int l = 0; l < nthread; l++) {
				final int startRow = 1 + l * p;
				final int stopRow;
				if (l == nthread - 1) {
					stopRow = n1;
				} else {
					stopRow = startRow + p;
				}
				futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						for (int r = startRow; r < stopRow; r++) {
							int idx5 = r * rowStride;
							int idx6 = (n1 - r + 1) * rowStride;
							for (int c = n2d2; c < n2; c++) {
								int idx1 = 2 * c;
								int idx2 = 2 * (n2 - c);
								a[idx1] = a[idx2];
								a[idx1 + 1] = -a[idx2 + 1];
								int idx3 = idx5 + idx1;
								int idx4 = idx6 - idx1;
								a[idx3] = a[idx4];
								a[idx3 + 1] = -a[idx4 + 1];
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
				fftn2.realInverse2(a, i * n2, scale);
			}
			for (int r = 0; r < n1; r++) {
				temp[0][r] = a[r * n2]; //first column is always real
			}
			fftn1.realInverseFull(temp[0], scale);

			for (int c = 1; c < n2d2 - 1; c++) {
				int idx0 = 2 * c;
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = r * n2 + idx0;
					temp[c][idx1] = a[idx2];
					temp[c][idx1 + 1] = a[idx2 + 1];
				}
				fftn1.complexInverse(temp[c], scale);
			}

			if ((n2 % 2) == 0) {
				for (int r = 0; r < n1; r++) {
					temp[n2d2 - 1][r] = a[r * n2 + 1];
					//imaginary part = 0;
				}
				fftn1.realInverseFull(temp[n2d2 - 1], scale);

			} else {
				for (int r = 0; r < n1; r++) {
					int idx1 = 2 * r;
					int idx2 = r * n2;
					int idx3 = n2d2 - 1;
					temp[idx3][idx1] = a[idx2 + 2 * idx3];
					temp[idx3][idx1 + 1] = a[idx2 + 1];
				}
				fftn1.complexInverse(temp[n2d2 - 1], scale);
			}

			for (int r = 0; r < n1; r++) {
				int idx1 = 2 * r;
				for (int c = 0; c < n2d2; c++) {
					int idx0 = 2 * c;
					int idx2 = r * rowStride + idx0;
					a[idx2] = temp[c][idx1];
					a[idx2 + 1] = temp[c][idx1 + 1];
				}
			}

			//fill symmetric
			for (int r = 1; r < n1; r++) {
				int idx5 = r * rowStride;
				int idx6 = (n1 - r + 1) * rowStride;
				for (int c = n2d2; c < n2; c++) {
					int idx1 = 2 * c;
					int idx2 = 2 * (n2 - c);
					a[idx1] = a[idx2];
					a[idx1 + 1] = -a[idx2 + 1];
					int idx3 = idx5 + idx1;
					int idx4 = idx6 - idx1;
					a[idx3] = a[idx4];
					a[idx3 + 1] = -a[idx4 + 1];
				}
			}
		}
	}

	private void rdft2d_sub(int isgn, double[] a) {
		int n1h, i, j;
		double xi;
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

	private void rdft2d_sub(int isgn, double[][] a) {
		int n1h, i, j;
		double xi;

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

	private void cdft2d_sub(int isgn, double[] a, boolean scale) {
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

	private void cdft2d_sub(int isgn, double[][] a, boolean scale) {
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

	private void xdft2d0_subth1(final int icr, final int isgn, final double[] a, final boolean scale) {
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

	private void xdft2d0_subth2(final int icr, final int isgn, final double[] a, final boolean scale) {
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

	private void xdft2d0_subth1(final int icr, final int isgn, final double[][] a, final boolean scale) {
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

	private void xdft2d0_subth2(final int icr, final int isgn, final double[][] a, final boolean scale) {
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

	private void cdft2d_subth(final int isgn, final double[] a, final boolean scale) {
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

	private void cdft2d_subth(final int isgn, final double[][] a, final boolean scale) {
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

	private void fillSymmetric(final double[] a) {
		final int twon2 = 2 * n2;
		int idx1, idx2, idx3, idx4;
		int n1d2 = n1 / 2;

		for (int r = (n1 - 1); r >= 1; r--) {
			idx1 = r * n2;
			idx2 = 2 * idx1;
			for (int c = 0; c < n2; c += 2) {
				a[idx2 + c] = a[idx1 + c];
				a[idx1 + c] = 0;
				a[idx2 + c + 1] = a[idx1 + c + 1];
				a[idx1 + c + 1] = 0;
			}
		}
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if ((nthread > 1) && useThreads && (n1d2 >= nthread)) {
			Future[] futures = new Future[nthread];
			int l1k = n1d2 / nthread;
			final int newn2 = 2 * n2;
			for (int i = 0; i < nthread; i++) {
				final int l1offa, l1stopa, l2offa, l2stopa;
				if (i == 0)
					l1offa = i * l1k + 1;
				else {
					l1offa = i * l1k;
				}
				l1stopa = i * l1k + l1k;
				l2offa = i * l1k;
				if (i == nthread - 1) {
					l2stopa = i * l1k + l1k + 1;
				} else {
					l2stopa = i * l1k + l1k;
				}
				futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx1, idx2, idx3, idx4;

						for (int r = l1offa; r < l1stopa; r++) {
							idx1 = r * newn2;
							idx2 = (n1 - r) * newn2;
							idx3 = idx1 + n2;
							a[idx3] = a[idx2 + 1];
							a[idx3 + 1] = -a[idx2];
						}
						for (int r = l1offa; r < l1stopa; r++) {
							idx1 = r * newn2;
							idx3 = (n1 - r + 1) * newn2;
							for (int c = n2 + 2; c < newn2; c += 2) {
								idx2 = idx3 - c;
								idx4 = idx1 + c;
								a[idx4] = a[idx2];
								a[idx4 + 1] = -a[idx2 + 1];

							}
						}
						for (int r = l2offa; r < l2stopa; r++) {
							idx3 = ((n1 - r) % n1) * newn2;
							idx4 = r * newn2;
							for (int c = 0; c < newn2; c += 2) {
								idx1 = idx3 + (newn2 - c) % newn2;
								idx2 = idx4 + c;
								a[idx1] = a[idx2];
								a[idx1 + 1] = -a[idx2 + 1];
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

		} else {

			for (int r = 1; r < n1d2; r++) {
				idx2 = r * twon2;
				idx3 = (n1 - r) * twon2;
				a[idx2 + n2] = a[idx3 + 1];
				a[idx2 + n2 + 1] = -a[idx3];
			}

			for (int r = 1; r < n1d2; r++) {
				idx2 = r * twon2;
				idx3 = (n1 - r + 1) * twon2;
				for (int c = n2 + 2; c < twon2; c += 2) {
					a[idx2 + c] = a[idx3 - c];
					a[idx2 + c + 1] = -a[idx3 - c + 1];

				}
			}
			for (int r = 0; r <= n1 / 2; r++) {
				idx1 = r * twon2;
				idx4 = ((n1 - r) % n1) * twon2;
				for (int c = 0; c < twon2; c += 2) {
					idx2 = idx1 + c;
					idx3 = idx4 + (twon2 - c) % twon2;
					a[idx3] = a[idx2];
					a[idx3 + 1] = -a[idx2 + 1];
				}
			}
		}
		a[n2] = -a[1];
		a[1] = 0;
		idx1 = n1d2 * twon2;
		a[idx1 + n2] = -a[idx1 + 1];
		a[idx1 + 1] = 0;
		a[idx1 + n2 + 1] = 0;
	}

	private void fillSymmetric(final double[][] a) {
		final int newn2 = 2 * n2;
		int n1d2 = n1 / 2;

		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if ((nthread > 1) && useThreads && (n1d2 >= nthread)) {
			Future[] futures = new Future[nthread];
			int l1k = n1d2 / nthread;
			for (int i = 0; i < nthread; i++) {
				final int l1offa, l1stopa, l2offa, l2stopa;
				if (i == 0)
					l1offa = i * l1k + 1;
				else {
					l1offa = i * l1k;
				}
				l1stopa = i * l1k + l1k;
				l2offa = i * l1k;
				if (i == nthread - 1) {
					l2stopa = i * l1k + l1k + 1;
				} else {
					l2stopa = i * l1k + l1k;
				}
				futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx1, idx2;
						for (int r = l1offa; r < l1stopa; r++) {
							idx1 = n1 - r;
							a[r][n2] = a[idx1][1];
							a[r][n2 + 1] = -a[idx1][0];
						}
						for (int r = l1offa; r < l1stopa; r++) {
							idx1 = n1 - r;
							for (int c = n2 + 2; c < newn2; c += 2) {
								idx2 = newn2 - c;
								a[r][c] = a[idx1][idx2];
								a[r][c + 1] = -a[idx1][idx2 + 1];

							}
						}
						for (int r = l2offa; r < l2stopa; r++) {
							idx1 = (n1 - r) % n1;
							for (int c = 0; c < newn2; c = c + 2) {
								idx2 = (newn2 - c) % newn2;
								a[idx1][idx2] = a[r][c];
								a[idx1][idx2 + 1] = -a[r][c + 1];
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
		} else {

			for (int r = 1; r < n1d2; r++) {
				int idx1 = n1 - r;
				a[r][n2] = a[idx1][1];
				a[r][n2 + 1] = -a[idx1][0];
			}
			for (int r = 1; r < n1d2; r++) {
				int idx1 = n1 - r;
				for (int c = n2 + 2; c < newn2; c += 2) {
					int idx2 = newn2 - c;
					a[r][c] = a[idx1][idx2];
					a[r][c + 1] = -a[idx1][idx2 + 1];
				}
			}
			for (int r = 0; r <= n1 / 2; r++) {
				int idx1 = (n1 - r) % n1;
				for (int c = 0; c < newn2; c += 2) {
					int idx2 = (newn2 - c) % newn2;
					a[idx1][idx2] = a[r][c];
					a[idx1][idx2 + 1] = -a[r][c + 1];
				}
			}
		}
		a[0][n2] = -a[0][1];
		a[0][1] = 0;
		a[n1d2][n2] = -a[n1d2][1];
		a[n1d2][1] = 0;
		a[n1d2][n2 + 1] = 0;
	}
}
