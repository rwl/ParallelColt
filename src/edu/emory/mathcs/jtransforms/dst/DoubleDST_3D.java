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

import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Computes 3D Discrete Sine Transform (DST) of double precision data. The sizes
 * of all three dimensions can be arbitrary numbers. This is a parallel
 * implementation optimized for SMP systems.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DoubleDST_3D {

	private int n1;

	private int n2;

	private int n3;

	private int sliceStride;

	private int rowStride;

	private double[] t;

	private DoubleDST_1D dstn1, dstn2, dstn3;

	private int oldNthread;

	private int nt;

	private boolean isPowerOfTwo = false;

	private boolean useThreads = false;

	/**
	 * Creates new instance of DoubleDST_3D.
	 * 
	 * @param n1
	 *            number of slices
	 * @param n2
	 *            number of rows
	 * @param n3
	 *            number of columns
	 */
	public DoubleDST_3D(int n1, int n2, int n3) {
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
			oldNthread = ConcurrencyUtils.getNumberOfProcessors();
			nt = n1;
			if (nt < n2) {
				nt = n2;
			}
			nt *= 4;
			if (oldNthread > 1) {
				nt *= oldNthread;
			}
			if (n3 == 2) {
				nt >>= 1;
			}
			t = new double[nt];
		}
		dstn1 = new DoubleDST_1D(n1);
		if (n1 == n2) {
			dstn2 = dstn1;
		} else {
			dstn2 = new DoubleDST_1D(n2);
		}
		if (n1 == n3) {
			dstn3 = dstn1;
		} else if (n2 == n3) {
			dstn3 = dstn2;
		} else {
			dstn3 = new DoubleDST_1D(n3);
		}
	}

	/**
	 * Computes the 3D forward DST (DST-II) leaving the result in <code>a</code>
	 * . The data is stored in 1D array addressed in slice-major, then
	 * row-major, then column-major, in order of significance, i.e. the element
	 * (i,j,k) of 3D array x[n1][n2][n3] is stored in a[i*sliceStride +
	 * j*rowStride + k], where sliceStride = n2 * n3 and rowStride = n3.
	 * 
	 * @param a
	 *            data to transform
	 * @param scale
	 *            if true then scaling is performed
	 */
	public void forward(final double[] a, final boolean scale) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			if (nthread != oldNthread) {
				nt = n1;
				if (nt < n2) {
					nt = n2;
				}
				nt *= 4;
				if (nthread > 1) {
					nt *= nthread;
				}
				if (n3 == 2) {
					nt >>= 1;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				ddxt3da_subth(-1, a, scale);
				ddxt3db_subth(-1, a, scale);
			} else {
				ddxt3da_sub(-1, a, scale);
				ddxt3db_sub(-1, a, scale);
			}
		} else {
			if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread) && (n3 >= nthread)) {
				Future[] futures = new Future[nthread];
				int p = n1 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							for (int s = startSlice; s < stopSlice; s++) {
								int idx1 = s * sliceStride;
								for (int r = 0; r < n2; r++) {
									dstn3.forward(a, idx1 + r * rowStride, scale);
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
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n2];
							for (int s = startSlice; s < stopSlice; s++) {
								int idx1 = s * sliceStride;
								for (int c = 0; c < n3; c++) {
									for (int r = 0; r < n2; r++) {
										int idx3 = idx1 + r * rowStride + c;
										int idx4 = 2 * r;
										temp[r] = a[idx3];
									}
									dstn2.forward(temp, scale);
									for (int r = 0; r < n2; r++) {
										int idx3 = idx1 + r * rowStride + c;
										a[idx3] = temp[r];
									}
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

				p = n2 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startRow = l * p;
					final int stopRow;
					if (l == nthread - 1) {
						stopRow = n2;
					} else {
						stopRow = startRow + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n1];
							for (int r = startRow; r < stopRow; r++) {
								int idx1 = r * rowStride;
								for (int c = 0; c < n3; c++) {
									for (int s = 0; s < n1; s++) {
										int idx3 = s * sliceStride + idx1 + c;
										temp[s] = a[idx3];
									}
									dstn1.forward(temp, scale);
									for (int s = 0; s < n1; s++) {
										int idx3 = s * sliceStride + idx1 + c;
										a[idx3] = temp[s];
									}
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
				for (int s = 0; s < n1; s++) {
					int idx1 = s * sliceStride;
					for (int r = 0; r < n2; r++) {
						dstn3.forward(a, idx1 + r * rowStride, scale);
					}
				}
				double[] temp = new double[n2];
				for (int s = 0; s < n1; s++) {
					int idx1 = s * sliceStride;
					for (int c = 0; c < n3; c++) {
						for (int r = 0; r < n2; r++) {
							int idx3 = idx1 + r * rowStride + c;
							temp[r] = a[idx3];
						}
						dstn2.forward(temp, scale);
						for (int r = 0; r < n2; r++) {
							int idx3 = idx1 + r * rowStride + c;
							a[idx3] = temp[r];
						}
					}
				}
				temp = new double[n1];
				for (int r = 0; r < n2; r++) {
					int idx1 = r * rowStride;
					for (int c = 0; c < n3; c++) {
						for (int s = 0; s < n1; s++) {
							int idx3 = s * sliceStride + idx1 + c;
							temp[s] = a[idx3];
						}
						dstn1.forward(temp, scale);
						for (int s = 0; s < n1; s++) {
							int idx3 = s * sliceStride + idx1 + c;
							a[idx3] = temp[s];
						}
					}
				}
			}
		}
	}

	/**
	 * Computes the 3D forward DST (DST-II) leaving the result in <code>a</code>
	 * . The data is stored in 3D array.
	 * 
	 * @param a
	 *            data to transform
	 * @param scale
	 *            if true then scaling is performed
	 */
	public void forward(final double[][][] a, final boolean scale) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			if (nthread != oldNthread) {
				nt = n1;
				if (nt < n2) {
					nt = n2;
				}
				nt *= 4;
				if (nthread > 1) {
					nt *= nthread;
				}
				if (n3 == 2) {
					nt >>= 1;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				ddxt3da_subth(-1, a, scale);
				ddxt3db_subth(-1, a, scale);
			} else {
				ddxt3da_sub(-1, a, scale);
				ddxt3db_sub(-1, a, scale);
			}
		} else {
			if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread) && (n3 >= nthread)) {
				Future[] futures = new Future[nthread];
				int p = n1 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							for (int s = startSlice; s < stopSlice; s++) {
								for (int r = 0; r < n2; r++) {
									dstn3.forward(a[s][r], scale);
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
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n2];
							for (int s = startSlice; s < stopSlice; s++) {
								for (int c = 0; c < n3; c++) {
									for (int r = 0; r < n2; r++) {
										temp[r] = a[s][r][c];
									}
									dstn2.forward(temp, scale);
									for (int r = 0; r < n2; r++) {
										a[s][r][c] = temp[r];
									}
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

				p = n2 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startRow = l * p;
					final int stopRow;
					if (l == nthread - 1) {
						stopRow = n2;
					} else {
						stopRow = startRow + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n1];
							for (int r = startRow; r < stopRow; r++) {
								for (int c = 0; c < n3; c++) {
									for (int s = 0; s < n1; s++) {
										temp[s] = a[s][r][c];
									}
									dstn1.forward(temp, scale);
									for (int s = 0; s < n1; s++) {
										a[s][r][c] = temp[s];
									}
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
				for (int s = 0; s < n1; s++) {
					for (int r = 0; r < n2; r++) {
						dstn3.forward(a[s][r], scale);
					}
				}
				double[] temp = new double[n2];
				for (int s = 0; s < n1; s++) {
					for (int c = 0; c < n3; c++) {
						for (int r = 0; r < n2; r++) {
							temp[r] = a[s][r][c];
						}
						dstn2.forward(temp, scale);
						for (int r = 0; r < n2; r++) {
							a[s][r][c] = temp[r];
						}
					}
				}
				temp = new double[n1];
				for (int r = 0; r < n2; r++) {
					for (int c = 0; c < n3; c++) {
						for (int s = 0; s < n1; s++) {
							temp[s] = a[s][r][c];
						}
						dstn1.forward(temp, scale);
						for (int s = 0; s < n1; s++) {
							a[s][r][c] = temp[s];
						}
					}
				}
			}
		}
	}

	/**
	 * Computes the 3D inverse DST (DST-III) leaving the result in
	 * <code>a</code>. The data is stored in 1D array addressed in slice-major,
	 * then row-major, then column-major, in order of significance, i.e. the
	 * element (i,j,k) of 3D array x[n1][n2][n3] is stored in a[i*sliceStride +
	 * j*rowStride + k], where sliceStride = n2 * n3 and rowStride = n3.
	 * 
	 * @param a
	 *            data to transform
	 * @param scale
	 *            if true then scaling is performed
	 */
	public void inverse(final double[] a, final boolean scale) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			if (nthread != oldNthread) {
				nt = n1;
				if (nt < n2) {
					nt = n2;
				}
				nt *= 4;
				if (nthread > 1) {
					nt *= nthread;
				}
				if (n3 == 2) {
					nt >>= 1;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				ddxt3da_subth(1, a, scale);
				ddxt3db_subth(1, a, scale);
			} else {
				ddxt3da_sub(1, a, scale);
				ddxt3db_sub(1, a, scale);
			}
		} else {
			if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread) && (n3 >= nthread)) {
				Future[] futures = new Future[nthread];
				int p = n1 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							for (int s = startSlice; s < stopSlice; s++) {
								int idx1 = s * sliceStride;
								for (int r = 0; r < n2; r++) {
									dstn3.inverse(a, idx1 + r * rowStride, scale);
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
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n2];
							for (int s = startSlice; s < stopSlice; s++) {
								int idx1 = s * sliceStride;
								for (int c = 0; c < n3; c++) {
									for (int r = 0; r < n2; r++) {
										int idx3 = idx1 + r * rowStride + c;
										temp[r] = a[idx3];
									}
									dstn2.inverse(temp, scale);
									for (int r = 0; r < n2; r++) {
										int idx3 = idx1 + r * rowStride + c;
										a[idx3] = temp[r];
									}
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

				p = n2 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startRow = l * p;
					final int stopRow;
					if (l == nthread - 1) {
						stopRow = n2;
					} else {
						stopRow = startRow + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n1];
							for (int r = startRow; r < stopRow; r++) {
								int idx1 = r * rowStride;
								for (int c = 0; c < n3; c++) {
									for (int s = 0; s < n1; s++) {
										int idx3 = s * sliceStride + idx1 + c;
										temp[s] = a[idx3];
									}
									dstn1.inverse(temp, scale);
									for (int s = 0; s < n1; s++) {
										int idx3 = s * sliceStride + idx1 + c;
										a[idx3] = temp[s];
									}
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
				for (int s = 0; s < n1; s++) {
					int idx1 = s * sliceStride;
					for (int r = 0; r < n2; r++) {
						dstn3.inverse(a, idx1 + r * rowStride, scale);
					}
				}
				double[] temp = new double[n2];
				for (int s = 0; s < n1; s++) {
					int idx1 = s * sliceStride;
					for (int c = 0; c < n3; c++) {
						for (int r = 0; r < n2; r++) {
							int idx3 = idx1 + r * rowStride + c;
							temp[r] = a[idx3];
						}
						dstn2.inverse(temp, scale);
						for (int r = 0; r < n2; r++) {
							int idx3 = idx1 + r * rowStride + c;
							a[idx3] = temp[r];
						}
					}
				}
				temp = new double[n1];
				for (int r = 0; r < n2; r++) {
					int idx1 = r * rowStride;
					for (int c = 0; c < n3; c++) {
						for (int s = 0; s < n1; s++) {
							int idx3 = s * sliceStride + idx1 + c;
							temp[s] = a[idx3];
						}
						dstn1.inverse(temp, scale);
						for (int s = 0; s < n1; s++) {
							int idx3 = s * sliceStride + idx1 + c;
							a[idx3] = temp[s];
						}
					}
				}
			}
		}

	}

	/**
	 * Computes the 3D inverse DST (DST-III) leaving the result in
	 * <code>a</code>. The data is stored in 3D array.
	 * 
	 * @param a
	 *            data to transform
	 * @param scale
	 *            if true then scaling is performed
	 */
	public void inverse(final double[][][] a, final boolean scale) {
		int nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (isPowerOfTwo) {
			if (nthread != oldNthread) {
				nt = n1;
				if (nt < n2) {
					nt = n2;
				}
				nt *= 4;
				if (nthread > 1) {
					nt *= nthread;
				}
				if (n3 == 2) {
					nt >>= 1;
				}
				t = new double[nt];
				oldNthread = nthread;
			}
			if ((nthread > 1) && useThreads) {
				ddxt3da_subth(1, a, scale);
				ddxt3db_subth(1, a, scale);
			} else {
				ddxt3da_sub(1, a, scale);
				ddxt3db_sub(1, a, scale);
			}
		} else {
			if ((nthread > 1) && useThreads && (n1 >= nthread) && (n2 >= nthread) && (n3 >= nthread)) {
				Future[] futures = new Future[nthread];
				int p = n1 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							for (int s = startSlice; s < stopSlice; s++) {
								for (int r = 0; r < n2; r++) {
									dstn3.inverse(a[s][r], scale);
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
					final int startSlice = l * p;
					final int stopSlice;
					if (l == nthread - 1) {
						stopSlice = n1;
					} else {
						stopSlice = startSlice + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n2];
							for (int s = startSlice; s < stopSlice; s++) {
								for (int c = 0; c < n3; c++) {
									for (int r = 0; r < n2; r++) {
										temp[r] = a[s][r][c];
									}
									dstn2.inverse(temp, scale);
									for (int r = 0; r < n2; r++) {
										a[s][r][c] = temp[r];
									}
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

				p = n2 / nthread;
				for (int l = 0; l < nthread; l++) {
					final int startRow = l * p;
					final int stopRow;
					if (l == nthread - 1) {
						stopRow = n2;
					} else {
						stopRow = startRow + p;
					}
					futures[l] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							double[] temp = new double[n1];
							for (int r = startRow; r < stopRow; r++) {
								for (int c = 0; c < n3; c++) {
									for (int s = 0; s < n1; s++) {
										temp[s] = a[s][r][c];
									}
									dstn1.inverse(temp, scale);
									for (int s = 0; s < n1; s++) {
										a[s][r][c] = temp[s];
									}
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
				for (int s = 0; s < n1; s++) {
					for (int r = 0; r < n2; r++) {
						dstn3.inverse(a[s][r], scale);
					}
				}
				double[] temp = new double[n2];
				for (int s = 0; s < n1; s++) {
					for (int c = 0; c < n3; c++) {
						for (int r = 0; r < n2; r++) {
							temp[r] = a[s][r][c];
						}
						dstn2.inverse(temp, scale);
						for (int r = 0; r < n2; r++) {
							a[s][r][c] = temp[r];
						}
					}
				}
				temp = new double[n1];
				for (int r = 0; r < n2; r++) {
					for (int c = 0; c < n3; c++) {
						for (int s = 0; s < n1; s++) {
							temp[s] = a[s][r][c];
						}
						dstn1.inverse(temp, scale);
						for (int s = 0; s < n1; s++) {
							a[s][r][c] = temp[s];
						}
					}
				}
			}
		}
	}

	/* -------- child routines -------- */

	private void ddxt3da_sub(int isgn, double[] a, boolean scale) {
		int i, j, k, idx0, idx1, idx2;

		if (isgn == -1) {
			for (i = 0; i < n1; i++) {
				idx0 = i * sliceStride;
				for (j = 0; j < n2; j++) {
					dstn3.forward(a, idx0 + j * rowStride, scale);
				}
				if (n3 > 2) {
					for (k = 0; k < n3; k += 4) {
						for (j = 0; j < n2; j++) {
							idx1 = idx0 + j * rowStride + k;
							idx2 = n2 + j;
							t[j] = a[idx1];
							t[idx2] = a[idx1 + 1];
							t[idx2 + n2] = a[idx1 + 2];
							t[idx2 + 2 * n2] = a[idx1 + 3];
						}
						dstn2.forward(t, 0, scale);
						dstn2.forward(t, n2, scale);
						dstn2.forward(t, 2 * n2, scale);
						dstn2.forward(t, 3 * n2, scale);
						for (j = 0; j < n2; j++) {
							idx1 = idx0 + j * rowStride + k;
							idx2 = n2 + j;
							a[idx1] = t[j];
							a[idx1 + 1] = t[idx2];
							a[idx1 + 2] = t[idx2 + n2];
							a[idx1 + 3] = t[idx2 + 2 * n2];
						}
					}
				} else if (n3 == 2) {
					for (j = 0; j < n2; j++) {
						idx1 = idx0 + j * rowStride;
						t[j] = a[idx1];
						t[n2 + j] = a[idx1 + 1];
					}
					dstn2.forward(t, 0, scale);
					dstn2.forward(t, n2, scale);
					for (j = 0; j < n2; j++) {
						idx1 = idx0 + j * rowStride;
						a[idx1] = t[j];
						a[idx1 + 1] = t[n2 + j];
					}
				}
			}
		} else {
			for (i = 0; i < n1; i++) {
				idx0 = i * sliceStride;
				for (j = 0; j < n2; j++) {
					dstn3.inverse(a, idx0 + j * rowStride, scale);
				}
				if (n3 > 2) {
					for (k = 0; k < n3; k += 4) {
						for (j = 0; j < n2; j++) {
							idx1 = idx0 + j * rowStride + k;
							idx2 = n2 + j;
							t[j] = a[idx1];
							t[idx2] = a[idx1 + 1];
							t[idx2 + n2] = a[idx1 + 2];
							t[idx2 + 2 * n2] = a[idx1 + 3];
						}
						dstn2.inverse(t, 0, scale);
						dstn2.inverse(t, n2, scale);
						dstn2.inverse(t, 2 * n2, scale);
						dstn2.inverse(t, 3 * n2, scale);
						for (j = 0; j < n2; j++) {
							idx1 = idx0 + j * rowStride + k;
							idx2 = n2 + j;
							a[idx1] = t[j];
							a[idx1 + 1] = t[idx2];
							a[idx1 + 2] = t[idx2 + n2];
							a[idx1 + 3] = t[idx2 + 2 * n2];
						}
					}
				} else if (n3 == 2) {
					for (j = 0; j < n2; j++) {
						idx1 = idx0 + j * rowStride;
						t[j] = a[idx1];
						t[n2 + j] = a[idx1 + 1];
					}
					dstn2.inverse(t, 0, scale);
					dstn2.inverse(t, n2, scale);
					for (j = 0; j < n2; j++) {
						idx1 = idx0 + j * rowStride;
						a[idx1] = t[j];
						a[idx1 + 1] = t[n2 + j];
					}
				}
			}
		}
	}

	private void ddxt3da_sub(int isgn, double[][][] a, boolean scale) {
		int i, j, k, idx2;

		if (isgn == -1) {
			for (i = 0; i < n1; i++) {
				for (j = 0; j < n2; j++) {
					dstn3.forward(a[i][j], scale);
				}
				if (n3 > 2) {
					for (k = 0; k < n3; k += 4) {
						for (j = 0; j < n2; j++) {
							idx2 = n2 + j;
							t[j] = a[i][j][k];
							t[idx2] = a[i][j][k + 1];
							t[idx2 + n2] = a[i][j][k + 2];
							t[idx2 + 2 * n2] = a[i][j][k + 3];
						}
						dstn2.forward(t, 0, scale);
						dstn2.forward(t, n2, scale);
						dstn2.forward(t, 2 * n2, scale);
						dstn2.forward(t, 3 * n2, scale);
						for (j = 0; j < n2; j++) {
							idx2 = n2 + j;
							a[i][j][k] = t[j];
							a[i][j][k + 1] = t[idx2];
							a[i][j][k + 2] = t[idx2 + n2];
							a[i][j][k + 3] = t[idx2 + 2 * n2];
						}
					}
				} else if (n3 == 2) {
					for (j = 0; j < n2; j++) {
						t[j] = a[i][j][0];
						t[n2 + j] = a[i][j][1];
					}
					dstn2.forward(t, 0, scale);
					dstn2.forward(t, n2, scale);
					for (j = 0; j < n2; j++) {
						a[i][j][0] = t[j];
						a[i][j][1] = t[n2 + j];
					}
				}
			}
		} else {
			for (i = 0; i < n1; i++) {
				for (j = 0; j < n2; j++) {
					dstn3.inverse(a[i][j], scale);
				}
				if (n3 > 2) {
					for (k = 0; k < n3; k += 4) {
						for (j = 0; j < n2; j++) {
							idx2 = n2 + j;
							t[j] = a[i][j][k];
							t[idx2] = a[i][j][k + 1];
							t[idx2 + n2] = a[i][j][k + 2];
							t[idx2 + 2 * n2] = a[i][j][k + 3];
						}
						dstn2.inverse(t, 0, scale);
						dstn2.inverse(t, n2, scale);
						dstn2.inverse(t, 2 * n2, scale);
						dstn2.inverse(t, 3 * n2, scale);
						for (j = 0; j < n2; j++) {
							idx2 = n2 + j;
							a[i][j][k] = t[j];
							a[i][j][k + 1] = t[idx2];
							a[i][j][k + 2] = t[idx2 + n2];
							a[i][j][k + 3] = t[idx2 + 2 * n2];
						}
					}
				} else if (n3 == 2) {
					for (j = 0; j < n2; j++) {
						t[j] = a[i][j][0];
						t[n2 + j] = a[i][j][1];
					}
					dstn2.inverse(t, 0, scale);
					dstn2.inverse(t, n2, scale);
					for (j = 0; j < n2; j++) {
						a[i][j][0] = t[j];
						a[i][j][1] = t[n2 + j];
					}
				}
			}
		}
	}

	private void ddxt3db_sub(int isgn, double[] a, boolean scale) {
		int i, j, k, idx0, idx1, idx2;

		if (isgn == -1) {
			if (n3 > 2) {
				for (j = 0; j < n2; j++) {
					idx0 = j * rowStride;
					for (k = 0; k < n3; k += 4) {
						for (i = 0; i < n1; i++) {
							idx1 = i * sliceStride + idx0 + k;
							idx2 = n1 + i;
							t[i] = a[idx1];
							t[idx2] = a[idx1 + 1];
							t[idx2 + n1] = a[idx1 + 2];
							t[idx2 + 2 * n1] = a[idx1 + 3];
						}
						dstn1.forward(t, 0, scale);
						dstn1.forward(t, n1, scale);
						dstn1.forward(t, 2 * n1, scale);
						dstn1.forward(t, 3 * n1, scale);
						for (i = 0; i < n1; i++) {
							idx1 = i * sliceStride + idx0 + k;
							idx2 = n1 + i;
							a[idx1] = t[i];
							a[idx1 + 1] = t[idx2];
							a[idx1 + 2] = t[idx2 + n1];
							a[idx1 + 3] = t[idx2 + 2 * n1];
						}
					}
				}
			} else if (n3 == 2) {
				for (j = 0; j < n2; j++) {
					idx0 = j * rowStride;
					for (i = 0; i < n1; i++) {
						idx1 = i * sliceStride + idx0;
						t[i] = a[idx1];
						t[n1 + i] = a[idx1 + 1];
					}
					dstn1.forward(t, 0, scale);
					dstn1.forward(t, n1, scale);
					for (i = 0; i < n1; i++) {
						idx1 = i * sliceStride + idx0;
						a[idx1] = t[i];
						a[idx1 + 1] = t[n1 + i];
					}
				}
			}
		} else {
			if (n3 > 2) {
				for (j = 0; j < n2; j++) {
					idx0 = j * rowStride;
					for (k = 0; k < n3; k += 4) {
						for (i = 0; i < n1; i++) {
							idx1 = i * sliceStride + idx0 + k;
							idx2 = n1 + i;
							t[i] = a[idx1];
							t[idx2] = a[idx1 + 1];
							t[idx2 + n1] = a[idx1 + 2];
							t[idx2 + 2 * n1] = a[idx1 + 3];
						}
						dstn1.inverse(t, 0, scale);
						dstn1.inverse(t, n1, scale);
						dstn1.inverse(t, 2 * n1, scale);
						dstn1.inverse(t, 3 * n1, scale);

						for (i = 0; i < n1; i++) {
							idx1 = i * sliceStride + idx0 + k;
							idx2 = n1 + i;
							a[idx1] = t[i];
							a[idx1 + 1] = t[idx2];
							a[idx1 + 2] = t[idx2 + n1];
							a[idx1 + 3] = t[idx2 + 2 * n1];
						}
					}
				}
			} else if (n3 == 2) {
				for (j = 0; j < n2; j++) {
					idx0 = j * rowStride;
					for (i = 0; i < n1; i++) {
						idx1 = i * sliceStride + idx0;
						t[i] = a[idx1];
						t[n1 + i] = a[idx1 + 1];
					}
					dstn1.inverse(t, 0, scale);
					dstn1.inverse(t, n1, scale);
					for (i = 0; i < n1; i++) {
						idx1 = i * sliceStride + idx0;
						a[idx1] = t[i];
						a[idx1 + 1] = t[n1 + i];
					}
				}
			}
		}
	}

	private void ddxt3db_sub(int isgn, double[][][] a, boolean scale) {
		int i, j, k, idx2;

		if (isgn == -1) {
			if (n3 > 2) {
				for (j = 0; j < n2; j++) {
					for (k = 0; k < n3; k += 4) {
						for (i = 0; i < n1; i++) {
							idx2 = n1 + i;
							t[i] = a[i][j][k];
							t[idx2] = a[i][j][k + 1];
							t[idx2 + n1] = a[i][j][k + 2];
							t[idx2 + 2 * n1] = a[i][j][k + 3];
						}
						dstn1.forward(t, 0, scale);
						dstn1.forward(t, n1, scale);
						dstn1.forward(t, 2 * n1, scale);
						dstn1.forward(t, 3 * n1, scale);
						for (i = 0; i < n1; i++) {
							idx2 = n1 + i;
							a[i][j][k] = t[i];
							a[i][j][k + 1] = t[idx2];
							a[i][j][k + 2] = t[idx2 + n1];
							a[i][j][k + 3] = t[idx2 + 2 * n1];
						}
					}
				}
			} else if (n3 == 2) {
				for (j = 0; j < n2; j++) {
					for (i = 0; i < n1; i++) {
						t[i] = a[i][j][0];
						t[n1 + i] = a[i][j][1];
					}
					dstn1.forward(t, 0, scale);
					dstn1.forward(t, n1, scale);
					for (i = 0; i < n1; i++) {
						a[i][j][0] = t[i];
						a[i][j][1] = t[n1 + i];
					}
				}
			}
		} else {
			if (n3 > 2) {
				for (j = 0; j < n2; j++) {
					for (k = 0; k < n3; k += 4) {
						for (i = 0; i < n1; i++) {
							idx2 = n1 + i;
							t[i] = a[i][j][k];
							t[idx2] = a[i][j][k + 1];
							t[idx2 + n1] = a[i][j][k + 2];
							t[idx2 + 2 * n1] = a[i][j][k + 3];
						}
						dstn1.inverse(t, 0, scale);
						dstn1.inverse(t, n1, scale);
						dstn1.inverse(t, 2 * n1, scale);
						dstn1.inverse(t, 3 * n1, scale);

						for (i = 0; i < n1; i++) {
							idx2 = n1 + i;
							a[i][j][k] = t[i];
							a[i][j][k + 1] = t[idx2];
							a[i][j][k + 2] = t[idx2 + n1];
							a[i][j][k + 3] = t[idx2 + 2 * n1];
						}
					}
				}
			} else if (n3 == 2) {
				for (j = 0; j < n2; j++) {
					for (i = 0; i < n1; i++) {
						t[i] = a[i][j][0];
						t[n1 + i] = a[i][j][1];
					}
					dstn1.inverse(t, 0, scale);
					dstn1.inverse(t, n1, scale);
					for (i = 0; i < n1; i++) {
						a[i][j][0] = t[i];
						a[i][j][1] = t[n1 + i];
					}
				}
			}
		}
	}

	private void ddxt3da_subth(final int isgn, final double[] a, final boolean scale) {
		int nthread, nt, i;
		nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (nthread > n1) {
			nthread = n1;
		}
		nt = 4 * n2;
		if (n3 == 2) {
			nt >>= 1;
		}
		final int nthread_f = nthread;
		Future[] futures = new Future[nthread];

		for (i = 0; i < nthread; i++) {
			final int n0 = i;
			final int startt = nt * i;
			futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

				public void run() {
					int idx0, idx1, idx2;
					if (isgn == -1) {
						for (int i = n0; i < n1; i += nthread_f) {
							idx0 = i * sliceStride;
							for (int j = 0; j < n2; j++) {
								dstn3.forward(a, idx0 + j * rowStride, scale);
							}
							if (n3 > 2) {
								for (int k = 0; k < n3; k += 4) {
									for (int j = 0; j < n2; j++) {
										idx1 = idx0 + j * rowStride + k;
										idx2 = startt + n2 + j;
										t[startt + j] = a[idx1];
										t[idx2] = a[idx1 + 1];
										t[idx2 + n2] = a[idx1 + 2];
										t[idx2 + 2 * n2] = a[idx1 + 3];
									}
									dstn2.forward(t, startt, scale);
									dstn2.forward(t, startt + n2, scale);
									dstn2.forward(t, startt + 2 * n2, scale);
									dstn2.forward(t, startt + 3 * n2, scale);
									for (int j = 0; j < n2; j++) {
										idx1 = idx0 + j * rowStride + k;
										idx2 = startt + n2 + j;
										a[idx1] = t[startt + j];
										a[idx1 + 1] = t[idx2];
										a[idx1 + 2] = t[idx2 + n2];
										a[idx1 + 3] = t[idx2 + 2 * n2];
									}
								}
							} else if (n3 == 2) {
								for (int j = 0; j < n2; j++) {
									idx1 = idx0 + j * rowStride;
									t[startt + j] = a[idx1];
									t[startt + n2 + j] = a[idx1 + 1];
								}
								dstn2.forward(t, startt, scale);
								dstn2.forward(t, startt + n2, scale);
								for (int j = 0; j < n2; j++) {
									idx1 = idx0 + j * rowStride;
									a[idx1] = t[startt + j];
									a[idx1 + 1] = t[startt + n2 + j];
								}
							}
						}
					} else {
						for (int i = n0; i < n1; i += nthread_f) {
							idx0 = i * sliceStride;
							for (int j = 0; j < n2; j++) {
								dstn3.inverse(a, idx0 + j * rowStride, scale);
							}
							if (n3 > 2) {
								for (int k = 0; k < n3; k += 4) {
									for (int j = 0; j < n2; j++) {
										idx1 = idx0 + j * rowStride + k;
										idx2 = startt + n2 + j;
										t[startt + j] = a[idx1];
										t[idx2] = a[idx1 + 1];
										t[idx2 + n2] = a[idx1 + 2];
										t[idx2 + 2 * n2] = a[idx1 + 3];
									}
									dstn2.inverse(t, startt, scale);
									dstn2.inverse(t, startt + n2, scale);
									dstn2.inverse(t, startt + 2 * n2, scale);
									dstn2.inverse(t, startt + 3 * n2, scale);
									for (int j = 0; j < n2; j++) {
										idx1 = idx0 + j * rowStride + k;
										idx2 = startt + n2 + j;
										a[idx1] = t[startt + j];
										a[idx1 + 1] = t[idx2];
										a[idx1 + 2] = t[idx2 + n2];
										a[idx1 + 3] = t[idx2 + 2 * n2];
									}
								}
							} else if (n3 == 2) {
								for (int j = 0; j < n2; j++) {
									idx1 = idx0 + j * rowStride;
									t[startt + j] = a[idx1];
									t[startt + n2 + j] = a[idx1 + 1];
								}
								dstn2.inverse(t, startt, scale);
								dstn2.inverse(t, startt + n2, scale);
								for (int j = 0; j < n2; j++) {
									idx1 = idx0 + j * rowStride;
									a[idx1] = t[startt + j];
									a[idx1 + 1] = t[startt + n2 + j];
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

	private void ddxt3da_subth(final int isgn, final double[][][] a, final boolean scale) {
		int nthread, nt, i;
		nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (nthread > n1) {
			nthread = n1;
		}
		nt = 4 * n2;
		if (n3 == 2) {
			nt >>= 1;
		}
		final int nthread_f = nthread;
		Future[] futures = new Future[nthread];

		for (i = 0; i < nthread; i++) {
			final int n0 = i;
			final int startt = nt * i;
			futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

				public void run() {
					int idx2;
					if (isgn == -1) {
						for (int i = n0; i < n1; i += nthread_f) {
							for (int j = 0; j < n2; j++) {
								dstn3.forward(a[i][j], scale);
							}
							if (n3 > 2) {
								for (int k = 0; k < n3; k += 4) {
									for (int j = 0; j < n2; j++) {
										idx2 = startt + n2 + j;
										t[startt + j] = a[i][j][k];
										t[idx2] = a[i][j][k + 1];
										t[idx2 + n2] = a[i][j][k + 2];
										t[idx2 + 2 * n2] = a[i][j][k + 3];
									}
									dstn2.forward(t, startt, scale);
									dstn2.forward(t, startt + n2, scale);
									dstn2.forward(t, startt + 2 * n2, scale);
									dstn2.forward(t, startt + 3 * n2, scale);
									for (int j = 0; j < n2; j++) {
										idx2 = startt + n2 + j;
										a[i][j][k] = t[startt + j];
										a[i][j][k + 1] = t[idx2];
										a[i][j][k + 2] = t[idx2 + n2];
										a[i][j][k + 3] = t[idx2 + 2 * n2];
									}
								}
							} else if (n3 == 2) {
								for (int j = 0; j < n2; j++) {
									t[startt + j] = a[i][j][0];
									t[startt + n2 + j] = a[i][j][1];
								}
								dstn2.forward(t, startt, scale);
								dstn2.forward(t, startt + n2, scale);
								for (int j = 0; j < n2; j++) {
									a[i][j][0] = t[startt + j];
									a[i][j][1] = t[startt + n2 + j];
								}
							}
						}
					} else {
						for (int i = n0; i < n1; i += nthread_f) {
							for (int j = 0; j < n2; j++) {
								dstn3.inverse(a[i][j], scale);
							}
							if (n3 > 2) {
								for (int k = 0; k < n3; k += 4) {
									for (int j = 0; j < n2; j++) {
										idx2 = startt + n2 + j;
										t[startt + j] = a[i][j][k];
										t[idx2] = a[i][j][k + 1];
										t[idx2 + n2] = a[i][j][k + 2];
										t[idx2 + 2 * n2] = a[i][j][k + 3];
									}
									dstn2.inverse(t, startt, scale);
									dstn2.inverse(t, startt + n2, scale);
									dstn2.inverse(t, startt + 2 * n2, scale);
									dstn2.inverse(t, startt + 3 * n2, scale);
									for (int j = 0; j < n2; j++) {
										idx2 = startt + n2 + j;
										a[i][j][k] = t[startt + j];
										a[i][j][k + 1] = t[idx2];
										a[i][j][k + 2] = t[idx2 + n2];
										a[i][j][k + 3] = t[idx2 + 2 * n2];
									}
								}
							} else if (n3 == 2) {
								for (int j = 0; j < n2; j++) {
									t[startt + j] = a[i][j][0];
									t[startt + n2 + j] = a[i][j][1];
								}
								dstn2.inverse(t, startt, scale);
								dstn2.inverse(t, startt + n2, scale);
								for (int j = 0; j < n2; j++) {
									a[i][j][0] = t[startt + j];
									a[i][j][1] = t[startt + n2 + j];
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

	private void ddxt3db_subth(final int isgn, final double[] a, final boolean scale) {
		int nthread, nt, i;
		nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (nthread > n2) {
			nthread = n2;
		}
		nt = 4 * n1;
		if (n3 == 2) {
			nt >>= 1;
		}
		Future[] futures = new Future[nthread];
		final int nthread_f = nthread;

		for (i = 0; i < nthread; i++) {
			final int n0 = i;
			final int startt = nt * i;
			futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

				public void run() {
					int idx0, idx1, idx2;
					if (isgn == -1) {
						if (n3 > 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								idx0 = j * rowStride;
								for (int k = 0; k < n3; k += 4) {
									for (int i = 0; i < n1; i++) {
										idx1 = i * sliceStride + idx0 + k;
										idx2 = startt + n1 + i;
										t[startt + i] = a[idx1];
										t[idx2] = a[idx1 + 1];
										t[idx2 + n1] = a[idx1 + 2];
										t[idx2 + 2 * n1] = a[idx1 + 3];
									}
									dstn1.forward(t, startt, scale);
									dstn1.forward(t, startt + n1, scale);
									dstn1.forward(t, startt + 2 * n1, scale);
									dstn1.forward(t, startt + 3 * n1, scale);
									for (int i = 0; i < n1; i++) {
										idx1 = i * sliceStride + idx0 + k;
										idx2 = startt + n1 + i;
										a[idx1] = t[startt + i];
										a[idx1 + 1] = t[idx2];
										a[idx1 + 2] = t[idx2 + n1];
										a[idx1 + 3] = t[idx2 + 2 * n1];
									}
								}
							}
						} else if (n3 == 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								idx0 = j * rowStride;
								for (int i = 0; i < n1; i++) {
									idx1 = i * sliceStride + idx0;
									t[startt + i] = a[idx1];
									t[startt + n1 + i] = a[idx1 + 1];
								}
								dstn1.forward(t, startt, scale);
								dstn1.forward(t, startt + n1, scale);
								for (int i = 0; i < n1; i++) {
									idx1 = i * sliceStride + idx0;
									a[idx1] = t[startt + i];
									a[idx1 + 1] = t[startt + n1 + i];
								}
							}
						}
					} else {
						if (n3 > 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								idx0 = j * rowStride;
								for (int k = 0; k < n3; k += 4) {
									for (int i = 0; i < n1; i++) {
										idx1 = i * sliceStride + idx0 + k;
										idx2 = startt + n1 + i;
										t[startt + i] = a[idx1];
										t[idx2] = a[idx1 + 1];
										t[idx2 + n1] = a[idx1 + 2];
										t[idx2 + 2 * n1] = a[idx1 + 3];
									}
									dstn1.inverse(t, startt, scale);
									dstn1.inverse(t, startt + n1, scale);
									dstn1.inverse(t, startt + 2 * n1, scale);
									dstn1.inverse(t, startt + 3 * n1, scale);
									for (int i = 0; i < n1; i++) {
										idx1 = i * sliceStride + idx0 + k;
										idx2 = startt + n1 + i;
										a[idx1] = t[startt + i];
										a[idx1 + 1] = t[idx2];
										a[idx1 + 2] = t[idx2 + n1];
										a[idx1 + 3] = t[idx2 + 2 * n1];
									}
								}
							}
						} else if (n3 == 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								idx0 = j * rowStride;
								for (int i = 0; i < n1; i++) {
									idx1 = i * sliceStride + idx0;
									t[startt + i] = a[idx1];
									t[startt + n1 + i] = a[idx1 + 1];
								}
								dstn1.inverse(t, startt, scale);
								dstn1.inverse(t, startt + n1, scale);

								for (int i = 0; i < n1; i++) {
									idx1 = i * sliceStride + idx0;
									a[idx1] = t[startt + i];
									a[idx1 + 1] = t[startt + n1 + i];
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

	private void ddxt3db_subth(final int isgn, final double[][][] a, final boolean scale) {
		int nthread, nt, i;
		nthread = ConcurrencyUtils.getNumberOfProcessors();
		if (nthread > n2) {
			nthread = n2;
		}
		nt = 4 * n1;
		if (n3 == 2) {
			nt >>= 1;
		}
		Future[] futures = new Future[nthread];
		final int nthread_f = nthread;

		for (i = 0; i < nthread; i++) {
			final int n0 = i;
			final int startt = nt * i;
			futures[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {

				public void run() {
					int idx2;
					if (isgn == -1) {
						if (n3 > 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								for (int k = 0; k < n3; k += 4) {
									for (int i = 0; i < n1; i++) {
										idx2 = startt + n1 + i;
										t[startt + i] = a[i][j][k];
										t[idx2] = a[i][j][k + 1];
										t[idx2 + n1] = a[i][j][k + 2];
										t[idx2 + 2 * n1] = a[i][j][k + 3];
									}
									dstn1.forward(t, startt, scale);
									dstn1.forward(t, startt + n1, scale);
									dstn1.forward(t, startt + 2 * n1, scale);
									dstn1.forward(t, startt + 3 * n1, scale);
									for (int i = 0; i < n1; i++) {
										idx2 = startt + n1 + i;
										a[i][j][k] = t[startt + i];
										a[i][j][k + 1] = t[idx2];
										a[i][j][k + 2] = t[idx2 + n1];
										a[i][j][k + 3] = t[idx2 + 2 * n1];
									}
								}
							}
						} else if (n3 == 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								for (int i = 0; i < n1; i++) {
									t[startt + i] = a[i][j][0];
									t[startt + n1 + i] = a[i][j][1];
								}
								dstn1.forward(t, startt, scale);
								dstn1.forward(t, startt + n1, scale);
								for (int i = 0; i < n1; i++) {
									a[i][j][0] = t[startt + i];
									a[i][j][1] = t[startt + n1 + i];
								}
							}
						}
					} else {
						if (n3 > 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								for (int k = 0; k < n3; k += 4) {
									for (int i = 0; i < n1; i++) {
										idx2 = startt + n1 + i;
										t[startt + i] = a[i][j][k];
										t[idx2] = a[i][j][k + 1];
										t[idx2 + n1] = a[i][j][k + 2];
										t[idx2 + 2 * n1] = a[i][j][k + 3];
									}
									dstn1.inverse(t, startt, scale);
									dstn1.inverse(t, startt + n1, scale);
									dstn1.inverse(t, startt + 2 * n1, scale);
									dstn1.inverse(t, startt + 3 * n1, scale);
									for (int i = 0; i < n1; i++) {
										idx2 = startt + n1 + i;
										a[i][j][k] = t[startt + i];
										a[i][j][k + 1] = t[idx2];
										a[i][j][k + 2] = t[idx2 + n1];
										a[i][j][k + 3] = t[idx2 + 2 * n1];
									}
								}
							}
						} else if (n3 == 2) {
							for (int j = n0; j < n2; j += nthread_f) {
								for (int i = 0; i < n1; i++) {
									t[startt + i] = a[i][j][0];
									t[startt + n1 + i] = a[i][j][1];
								}
								dstn1.inverse(t, startt, scale);
								dstn1.inverse(t, startt + n1, scale);

								for (int i = 0; i < n1; i++) {
									a[i][j][0] = t[startt + i];
									a[i][j][1] = t[startt + n1 + i];
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
}
