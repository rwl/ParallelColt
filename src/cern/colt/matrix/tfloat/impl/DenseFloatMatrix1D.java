/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import edu.emory.mathcs.jtransforms.dct.FloatDCT_1D;
import edu.emory.mathcs.jtransforms.dht.FloatDHT_1D;
import edu.emory.mathcs.jtransforms.dst.FloatDST_1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 1-d matrix (aka <i>vector</i>) holding <tt>float</tt> elements. First
 * see the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array. Note that this
 * implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*size()</tt>. Thus, a 1000000 matrix uses 8 MB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 * <p>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseFloatMatrix1D extends FloatMatrix1D {
	private static final long serialVersionUID = -706456704651139684L;

	private FloatFFT_1D fft;

	private FloatDCT_1D dct;

	private FloatDST_1D dst;

	private FloatDHT_1D dht;

	/**
	 * The elements of this matrix.
	 */
	protected float[] elements;

	/**
	 * Constructs a matrix with a copy of the given values. The values are
	 * copied. So subsequent changes in <tt>values</tt> are not reflected in the
	 * matrix, and vice-versa.
	 * 
	 * @param values
	 *            The values to be filled into the new matrix.
	 */
	public DenseFloatMatrix1D(float[] values) {
		this(values.length);
		assign(values);
	}

	/**
	 * Constructs a matrix with a given number of cells. All entries are
	 * initially <tt>0</tt>.
	 * 
	 * @param size
	 *            the number of cells the matrix shall have.
	 * @throws IllegalArgumentException
	 *             if <tt>size<0</tt>.
	 */
	public DenseFloatMatrix1D(int size) {
		setUp(size);
		this.elements = new float[size];
	}

	/**
	 * Constructs a matrix view with the given parameters.
	 * 
	 * @param size
	 *            the number of cells the matrix shall have.
	 * @param elements
	 *            the cells.
	 * @param zero
	 *            the index of the first element.
	 * @param stride
	 *            the number of indexes between any two elements, i.e.
	 *            <tt>index(i+1)-index(i)</tt>.
	 * @throws IllegalArgumentException
	 *             if <tt>size<0</tt>.
	 */
	public DenseFloatMatrix1D(int size, float[] elements, int zero, int stride) {
		setUp(size, zero, stride);
		this.elements = elements;
		this.isNoView = false;
	}

	public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFunction f) {
		if (size == 0)
			return Float.NaN;
		float a = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			Float[] results = new Float[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {
					public Float call() throws Exception {
						int idx = zero + startidx * stride;
						float a = f.apply(elements[idx]);
						for (int i = startidx + 1; i < stopidx; i++) {
							idx += stride;
							a = aggr.apply(a, f.apply(elements[idx]));
						}
						return a;
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (Float) futures[j].get();
				}
				a = results[0];
				for (int j = 1; j < np; j++) {
					a = aggr.apply(a, results[j]);
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			a = f.apply(elements[zero]);
			int idx = zero;
			for (int i = 1; i < size; i++) {
				idx += stride;
				a = aggr.apply(a, f.apply(elements[idx]));
			}
		}
		return a;
	}

	public float aggregate(final FloatMatrix1D other, final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFloatFunction f) {
		if (!(other instanceof DenseFloatMatrix1D)) {
			return super.aggregate(other, aggr, f);
		}
		checkSize(other);
		if (size == 0)
			return Float.NaN;
		final int zeroOther = other.index(0);
		final int strideOther = other.stride();
		final float[] elemsOther = (float[]) other.elements();
		float a = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			Float[] results = new Float[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {
					public Float call() throws Exception {
						int idx = zero + startidx * stride;
						int idxOther = zeroOther + startidx * strideOther;
						float a = f.apply(elements[idx], elemsOther[idxOther]);
						for (int i = startidx + 1; i < stopidx; i++) {
							idx += stride;
							idxOther += strideOther;
							a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
						}
						return a;
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (Float) futures[j].get();
				}
				a = results[0];
				for (int j = 1; j < np; j++) {
					a = aggr.apply(a, results[j]);
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			a = f.apply(elements[zero], elemsOther[zeroOther]);
			int idx = zero;
			int idxOther = zeroOther;
			for (int i = 1; i < size; i++) {
				idx += stride;
				idxOther += strideOther;
				a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
			}
		}
		return a;
	}

	public FloatMatrix1D assign(final cern.colt.function.tfloat.FloatFunction function) {
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			if (function instanceof cern.jet.math.tfloat.FloatMult) {
				// x[i] = mult*x[i]
				float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
				if (multiplicator == 1)
					return this;
			}
			Future[] futures = new Future[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

					public void run() {
						int idx;
						// specialization for speed
						if (function instanceof cern.jet.math.tfloat.FloatMult) {
							// x[i] = mult*x[i]
							float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
							idx = zero + startidx * stride;
							for (int k = startidx; k < stopidx; k++) {
								elements[idx] *= multiplicator;
								idx += stride;
							}
						} else {
							// the general case x[i] = f(x[i])
							idx = zero + startidx * stride;
							for (int k = startidx; k < stopidx; k++) {
								elements[idx] = function.apply(elements[idx]);
								idx += stride;
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
			int idx;
			// specialization for speed
			if (function instanceof cern.jet.math.tfloat.FloatMult) {
				// x[i] = mult*x[i]
				float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
				if (multiplicator == 1)
					return this;
				idx = zero;
				for (int k = 0; k < size; k++) {
					elements[idx] *= multiplicator;
					idx += stride;
				}
			} else {
				// the general case x[i] = f(x[i])
				idx = zero;
				for (int k = 0; k < size; k++) {
					elements[idx] = function.apply(elements[idx]);
					idx += stride;
				}
			}
		}
		return this;
	}

	public FloatMatrix1D assign(final cern.colt.function.tfloat.FloatProcedure cond, final cern.colt.function.tfloat.FloatFunction function) {
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

					public void run() {
						int idx = zero + startidx * stride;
						for (int i = startidx; i < stopidx; i++) {
							if (cond.apply(elements[idx]) == true) {
								elements[idx] = function.apply(elements[idx]);
							}
							idx += stride;
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
			int idx = zero;
			for (int i = 0; i < size; i++) {
				if (cond.apply(elements[idx]) == true) {
					elements[idx] = function.apply(elements[idx]);
				}
				idx += stride;
			}
		}
		return this;
	}

	public FloatMatrix1D assign(final cern.colt.function.tfloat.FloatProcedure cond, final float value) {
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

					public void run() {
						int idx = zero + startidx * stride;
						for (int i = startidx; i < stopidx; i++) {
							if (cond.apply(elements[idx]) == true) {
								elements[idx] = value;
							}
							idx += stride;
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
			int idx = zero;
			for (int i = 0; i < size; i++) {
				if (cond.apply(elements[idx]) == true) {
					elements[idx] = value;
				}
				idx += stride;
			}
		}
		return this;
	}

	public FloatMatrix1D assign(final float value) {
		final float[] elems = this.elements;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx = zero + startidx * stride;
						for (int k = startidx; k < stopidx; k++) {
							elems[idx] = value;
							idx += stride;
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
			int idx = zero;
			for (int i = 0; i < size; i++) {
				elems[idx] = value;
				idx += stride;
			}
		}
		return this;
	}

	public FloatMatrix1D assign(final float[] values) {
		if (values.length != size)
			throw new IllegalArgumentException("Must have same number of cells: length=" + values.length + "size()=" + size());
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if (isNoView) {
			if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
				Future[] futures = new Future[np];
				int k = size / np;
				for (int j = 0; j < np; j++) {
					final int startidx = j * k;
					final int length;
					if (j == np - 1) {
						length = size - startidx;
					} else {
						length = k;
					}
					futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							System.arraycopy(values, startidx, elements, startidx, length);
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
				System.arraycopy(values, 0, this.elements, 0, values.length);
			}
		} else {
			if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
				Future[] futures = new Future[np];
				int k = size / np;
				for (int j = 0; j < np; j++) {
					final int startidx = j * k;
					final int stopidx;
					if (j == np - 1) {
						stopidx = size;
					} else {
						stopidx = startidx + k;
					}
					futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

						public void run() {
							int idx = zero + startidx * stride;
							for (int i = startidx; i < stopidx; i++) {
								elements[idx] = values[i];
								idx += stride;
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
				int idx = zero;
				for (int i = 0; i < size; i++) {
					elements[idx] = values[i];
					idx += stride;
				}
			}
		}
		return this;
	}

	public FloatMatrix1D assign(FloatMatrix1D source) {
		// overriden for performance only
		if (!(source instanceof DenseFloatMatrix1D)) {
			super.assign(source);
			return this;
		}
		DenseFloatMatrix1D other = (DenseFloatMatrix1D) source;
		if (other == this)
			return this;
		checkSize(other);
		if (isNoView && other.isNoView) {
			// quickest
			System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
			return this;
		}
		if (haveSharedCells(other)) {
			FloatMatrix1D c = other.copy();
			if (!(c instanceof DenseFloatMatrix1D)) {
				// should not happen
				super.assign(source);
				return this;
			}
			other = (DenseFloatMatrix1D) c;
		}

		final float[] elemsOther = other.elements;
		if (elements == null || elemsOther == null)
			throw new InternalError();
		final int zeroOther = other.index(0);
		final int strideOther = other.stride;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx = zero + startidx * stride;
						int idxOther = zeroOther + startidx * strideOther;
						for (int k = startidx; k < stopidx; k++) {
							elements[idx] = elemsOther[idxOther];
							idx += stride;
							idxOther += strideOther;
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
			int idx = zero;
			int idxOther = zeroOther;
			for (int k = 0; k < size; k++) {
				elements[idx] = elemsOther[idxOther];
				idx += stride;
				idxOther += strideOther;
			}
		}
		return this;
	}

	public FloatMatrix1D assign(final FloatMatrix1D y, final cern.colt.function.tfloat.FloatFloatFunction function) {
		// overriden for performance only
		if (!(y instanceof DenseFloatMatrix1D)) {
			super.assign(y, function);
			return this;
		}
		checkSize(y);
		final int zeroOther = y.index(0);
		final int strideOther = y.stride();
		final float[] elemsOther = (float[]) y.elements();
		if (elements == null || elemsOther == null)
			throw new InternalError();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx;
						int idxOther;
						// specialized for speed
						if (function == cern.jet.math.tfloat.FloatFunctions.mult) {
							// x[i] = x[i] * y[i]
							idx = zero + startidx * stride;
							idxOther = zeroOther + startidx * strideOther;
							for (int k = startidx; k < stopidx; k++) {
								elements[idx] *= elemsOther[idxOther];
								idx += stride;
								idxOther += strideOther;
							}
						} else if (function == cern.jet.math.tfloat.FloatFunctions.div) {
							// x[i] = x[i] / y[i]
							idx = zero + startidx * stride;
							idxOther = zeroOther + startidx * strideOther;
							for (int k = startidx; k < stopidx; k++) {
								elements[idx] /= elemsOther[idxOther];
								idx += stride;
								idxOther += strideOther;

							}
						} else if (function instanceof cern.jet.math.tfloat.FloatPlusMult) {
							float multiplicator = ((cern.jet.math.tfloat.FloatPlusMult) function).multiplicator;
							if (multiplicator == 0) {
								// x[i] = x[i] + 0*y[i]
								return;
							} else if (multiplicator == 1) {
								// x[i] = x[i] + y[i]
								idx = zero + startidx * stride;
								idxOther = zeroOther + startidx * strideOther;
								for (int k = startidx; k < stopidx; k++) {
									elements[idx] += elemsOther[idxOther];
									idx += stride;
									idxOther += strideOther;
								}
							} else if (multiplicator == -1) {
								// x[i] = x[i] - y[i]
								idx = zero + startidx * stride;
								idxOther = zeroOther + startidx * strideOther;
								for (int k = startidx; k < stopidx; k++) {
									elements[idx] -= elemsOther[idxOther];
									idx += stride;
									idxOther += strideOther;
								}
							} else {
								// the general case x[i] = x[i] + mult*y[i]
								idx = zero + startidx * stride;
								idxOther = zeroOther + startidx * strideOther;
								for (int k = startidx; k < stopidx; k++) {
									elements[idx] += multiplicator * elemsOther[idxOther];
									idx += stride;
									idxOther += strideOther;
								}

							}
						} else {
							// the general case x[i] = f(x[i],y[i])
							idx = zero + startidx * stride;
							idxOther = zeroOther + startidx * strideOther;
							for (int k = startidx; k < stopidx; k++) {
								elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
								idx += stride;
								idxOther += strideOther;
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
			// specialized for speed
			int idx;
			int idxOther;
			if (function == cern.jet.math.tfloat.FloatFunctions.mult) {
				// x[i] = x[i] * y[i]
				idx = zero;
				idxOther = zeroOther;
				for (int k = 0; k < size; k++) {
					elements[idx] *= elemsOther[idxOther];
					idx += stride;
					idxOther += strideOther;
				}
			} else if (function == cern.jet.math.tfloat.FloatFunctions.div) {
				// x[i] = x[i] / y[i]
				idx = zero;
				idxOther = zeroOther;
				for (int k = 0; k < size; k++) {
					elements[idx] /= elemsOther[idxOther];
					idx += stride;
					idxOther += strideOther;
				}
			} else if (function instanceof cern.jet.math.tfloat.FloatPlusMult) {
				float multiplicator = ((cern.jet.math.tfloat.FloatPlusMult) function).multiplicator;
				if (multiplicator == 0) {
					// x[i] = x[i] + 0*y[i]
					return this;
				} else if (multiplicator == 1) {
					// x[i] = x[i] + y[i]
					idx = zero;
					idxOther = zeroOther;
					for (int k = 0; k < size; k++) {
						elements[idx] += elemsOther[idxOther];
						idx += stride;
						idxOther += strideOther;
					}
				} else if (multiplicator == -1) {
					// x[i] = x[i] - y[i]
					idx = zero;
					idxOther = zeroOther;
					for (int k = 0; k < size; k++) {
						elements[idx] -= elemsOther[idxOther];
						idx += stride;
						idxOther += strideOther;
					}
				} else {
					// the general case x[i] = x[i] + mult*y[i]
					idx = zero;
					idxOther = zeroOther;
					for (int k = 0; k < size; k++) {
						elements[idx] += multiplicator * elemsOther[idxOther];
						idx += stride;
						idxOther += strideOther;
					}
				}
			} else {
				// the general case x[i] = f(x[i],y[i])
				idx = zero;
				idxOther = zeroOther;
				for (int k = 0; k < size; k++) {
					elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
					idx += stride;
					idxOther += strideOther;
				}
			}
		}
		return this;
	}

	public int cardinality() {
		int cardinality = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			Integer[] results = new Integer[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopsize;
				if (j == np - 1) {
					stopsize = size;
				} else {
					stopsize = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Integer>() {
					public Integer call() throws Exception {
						int cardinality = 0;
						int idx = zero + startidx * stride;
						for (int i = startidx; i < stopsize; i++) {
							if (elements[idx] != 0)
								cardinality++;
							idx += stride;
						}
						return cardinality;
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (Integer) futures[j].get();
				}
				cardinality = results[0];
				for (int j = 1; j < np; j++) {
					cardinality += results[j];
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			int idx = zero;
			for (int i = 0; i < size; i++) {
				if (elements[idx] != 0)
					cardinality++;
				idx += stride;
			}
		}
		return cardinality;
	}

	public void dct(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (dct == null) {
			dct = new FloatDCT_1D(size);
		}
		if (isNoView) {
			dct.forward(elements, scale);
		} else {
			FloatMatrix1D copy = this.copy();
			dct.forward((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void dht() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (dht == null) {
			dht = new FloatDHT_1D(size);
		}
		if (isNoView) {
			dht.forward(elements);
		} else {
			FloatMatrix1D copy = this.copy();
			dht.forward((float[]) copy.elements());
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void dst(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (dst == null) {
			dst = new FloatDST_1D(size);
		}
		if (isNoView) {
			dst.forward(elements, scale);
		} else {
			FloatMatrix1D copy = this.copy();
			dst.forward((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public float[] elements() {
		return elements;
	}

	public void fft() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (fft == null) {
			fft = new FloatFFT_1D(size);
		}
		if (isNoView) {
			fft.realForward(elements);
		} else {
			FloatMatrix1D copy = this.copy();
			fft.realForward((float[]) copy.elements());
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public FComplexMatrix1D getFft() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		final float[] elems;
		if (isNoView == true) {
			elems = elements;
		} else {
			elems = (float[]) this.copy().elements();
		}
		FComplexMatrix1D c = new DenseFComplexMatrix1D(size);
		final float[] cElems = (float[]) ((DenseFComplexMatrix1D) c).elements();
		System.arraycopy(elems, 0, cElems, 0, size);
		if (fft == null) {
			fft = new FloatFFT_1D(size);
		}
		fft.realForwardFull(cElems);
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
		return c;
	}

	public FComplexMatrix1D getIfft(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		final float[] elems;
		if (isNoView == true) {
			elems = elements;
		} else {
			elems = (float[]) this.copy().elements();
		}
		FComplexMatrix1D c = new DenseFComplexMatrix1D(size);
		final float[] cElems = (float[]) ((DenseFComplexMatrix1D) c).elements();
		System.arraycopy(elems, 0, cElems, 0, size);
		if (fft == null) {
			fft = new FloatFFT_1D(size);
		}
		fft.realInverseFull(cElems, scale);
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
		return c;
	}

	public void getNonZeros(final IntArrayList indexList, final FloatArrayList valueList) {
		indexList.clear();
		valueList.clear();
		int idx = zero;
		int rem = size % 2;
		if (rem == 1) {
			float value = elements[idx];
			if (value != 0) {
				indexList.add(0);
				valueList.add(value);
			}
			idx += stride;

		}
		for (int i = rem; i < size; i += 2) {
			float value = elements[idx];
			if (value != 0) {
				indexList.add(i);
				valueList.add(value);
			}
			idx += stride;
			value = elements[idx];
			if (value != 0) {
				indexList.add(i + 1);
				valueList.add(value);
			}
			idx += stride;
		}
	}

	public void getPositiveValues(final IntArrayList indexList, final FloatArrayList valueList) {
		indexList.clear();
		valueList.clear();
		int idx = zero;
		int rem = size % 2;
		if (rem == 1) {
			float value = elements[idx];
			if (value > 0) {
				indexList.add(0);
				valueList.add(value);
			}
			idx += stride;

		}
		for (int i = rem; i < size; i += 2) {
			float value = elements[idx];
			if (value > 0) {
				indexList.add(i);
				valueList.add(value);
			}
			idx += stride;
			value = elements[idx];
			if (value > 0) {
				indexList.add(i + 1);
				valueList.add(value);
			}
			idx += stride;
		}
	}

	public void getNegativeValues(final IntArrayList indexList, final FloatArrayList valueList) {
		indexList.clear();
		valueList.clear();
		int idx = zero;
		int rem = size % 2;
		if (rem == 1) {
			float value = elements[idx];
			if (value < 0) {
				indexList.add(0);
				valueList.add(value);
			}
			idx += stride;

		}
		for (int i = rem; i < size; i += 2) {
			float value = elements[idx];
			if (value < 0) {
				indexList.add(i);
				valueList.add(value);
			}
			idx += stride;
			value = elements[idx];
			if (value < 0) {
				indexList.add(i + 1);
				valueList.add(value);
			}
			idx += stride;
		}
	}

	public float[] getMaxLocation() {
		int location = 0;
		float maxValue = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			float[][] results = new float[np][2];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<float[]>() {
					public float[] call() throws Exception {
						int idx = zero + startidx * stride;
						float maxValue = elements[idx];
						int location = (idx - zero) / stride;
						for (int i = startidx + 1; i < stopidx; i++) {
							idx += stride;
							if (maxValue < elements[idx]) {
								maxValue = elements[idx];
								location = (idx - zero) / stride;
							}
						}
						return new float[] { maxValue, location };
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (float[]) futures[j].get();
				}
				maxValue = results[0][0];
				location = (int) results[0][1];
				for (int j = 1; j < np; j++) {
					if (maxValue < results[j][0]) {
						maxValue = results[j][0];
						location = (int) results[j][1];
					}
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			maxValue = elements[zero];
			location = 0;
			int idx = zero;
			for (int i = 1; i < size; i++) {
				idx += stride;
				if (maxValue < elements[idx]) {
					maxValue = elements[idx];
					location = (idx - zero) / stride;
				}
			}
		}
		return new float[] { maxValue, location };
	}

	public float[] getMinLocation() {
		int location = 0;
		float minValue = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			float[][] results = new float[np][2];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<float[]>() {
					public float[] call() throws Exception {
						int idx = zero + startidx * stride;
						float minValue = elements[idx];
						int location = (idx - zero) / stride;
						for (int i = startidx + 1; i < stopidx; i++) {
							idx += stride;
							if (minValue > elements[idx]) {
								minValue = elements[idx];
								location = (idx - zero) / stride;
							}
						}
						return new float[] { minValue, location };
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (float[]) futures[j].get();
				}
				minValue = results[0][0];
				location = (int) results[0][1];
				for (int j = 1; j < np; j++) {
					if (minValue > results[j][0]) {
						minValue = results[j][0];
						location = (int) results[j][1];
					}
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			minValue = elements[zero];
			location = 0;
			int idx = zero;
			for (int i = 1; i < size; i++) {
				idx += stride;
				if (minValue > elements[idx]) {
					minValue = elements[idx];
					location = (idx - zero) / stride;
				}
			}
		}
		return new float[] { minValue, location };
	}

	public float getQuick(int index) {
		return elements[zero + index * stride];
	}

	public void idct(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (dct == null) {
			dct = new FloatDCT_1D(size);
		}
		if (isNoView) {
			dct.inverse(elements, scale);
		} else {
			FloatMatrix1D copy = this.copy();
			dct.inverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void idht(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (dht == null) {
			dht = new FloatDHT_1D(size);
		}
		if (isNoView) {
			dht.inverse(elements, scale);
		} else {
			FloatMatrix1D copy = this.copy();
			dht.inverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void idst(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (dst == null) {
			dst = new FloatDST_1D(size);
		}
		if (isNoView) {
			dst.inverse(elements, scale);
		} else {
			FloatMatrix1D copy = this.copy();
			dst.inverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void ifft(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
		if (fft == null) {
			fft = new FloatFFT_1D(size);
		}
		if (isNoView) {
			fft.realInverse(elements, scale);
		} else {
			FloatMatrix1D copy = this.copy();
			fft.realInverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public FloatMatrix1D like(int size) {
		return new DenseFloatMatrix1D(size);
	}

	public FloatMatrix2D like2D(int rows, int columns) {
		return new DenseFloatMatrix2D(rows, columns);
	}

	public FloatMatrix2D reshape(final int rows, final int cols) {
		if (rows * cols != size) {
			throw new IllegalArgumentException("rows*cols != size");
		}
		FloatMatrix2D M = new DenseFloatMatrix2D(rows, cols);
		final float[] elemsOther = (float[]) M.elements();
		final int zeroOther = M.index(0, 0);
		final int rowStrideOther = M.rowStride();
		final int colStrideOther = M.columnStride();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = cols / np;
			for (int j = 0; j < np; j++) {
				final int startcol = j * k;
				final int stopcol;
				if (j == np - 1) {
					stopcol = cols;
				} else {
					stopcol = startcol + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx;
						int idxOther;
						for (int c = startcol; c < stopcol; c++) {
							idxOther = zeroOther + c * colStrideOther;
							idx = zero + (c * rows) * stride;
							for (int r = 0; r < rows; r++) {
								elemsOther[idxOther] = elements[idx];
								idxOther += rowStrideOther;
								idx += stride;
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
			int idxOther;
			int idx = zero;
			for (int c = 0; c < cols; c++) {
				idxOther = zeroOther + c * colStrideOther;
				for (int r = 0; r < rows; r++) {
					elemsOther[idxOther] = elements[idx];
					idxOther += rowStrideOther;
					idx += stride;
				}
			}
		}
		return M;
	}

	public FloatMatrix3D reshape(final int slices, final int rows, final int cols) {
		if (slices * rows * cols != size) {
			throw new IllegalArgumentException("slices*rows*cols != size");
		}
		FloatMatrix3D M = new DenseFloatMatrix3D(slices, rows, cols);
		final float[] elemsOther = (float[]) M.elements();
		final int zeroOther = M.index(0, 0, 0);
		final int sliceStrideOther = M.sliceStride();
		final int rowStrideOther = M.rowStride();
		final int colStrideOther = M.columnStride();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = slices / np;
			for (int j = 0; j < np; j++) {
				final int startslice = j * k;
				final int stopslice;
				if (j == np - 1) {
					stopslice = slices;
				} else {
					stopslice = startslice + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx;
						int idxOther;
						for (int s = startslice; s < stopslice; s++) {
							for (int c = 0; c < cols; c++) {
								idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
								idx = zero + (s * rows * cols + c * rows) * stride;
								for (int r = 0; r < rows; r++) {
									elemsOther[idxOther] = elements[idx];
									idxOther += rowStrideOther;
									idx += stride;
								}
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
			int idxOther;
			int idx = zero;
			for (int s = 0; s < slices; s++) {
				for (int c = 0; c < cols; c++) {
					idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
					for (int r = 0; r < rows; r++) {
						elemsOther[idxOther] = elements[idx];
						idxOther += rowStrideOther;
						idx += stride;
					}
				}
			}
		}
		return M;
	}

	public void setQuick(int index, float value) {
		elements[zero + index * stride] = value;
	}

	public void swap(final FloatMatrix1D other) {
		// overriden for performance only
		if (!(other instanceof DenseFloatMatrix1D)) {
			super.swap(other);
		}
		DenseFloatMatrix1D y = (DenseFloatMatrix1D) other;
		if (y == this)
			return;
		checkSize(y);
		final float[] elemsOther = y.elements;
		if (elements == null || elemsOther == null)
			throw new InternalError();
		final int zeroOther = other.index(0);
		final int strideOther = other.stride();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx = zero + startidx * stride;
						int idxOther = zeroOther + startidx * strideOther;
						for (int k = startidx; k < stopidx; k++) {
							float tmp = elements[idx];
							elements[idx] = elemsOther[idxOther];
							elemsOther[idxOther] = tmp;
							idx += stride;
							idxOther += strideOther;
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
			int idx = zero;
			int idxOther = zeroOther;
			for (int k = 0; k < size; k++) {
				float tmp = elements[idx];
				elements[idx] = elemsOther[idxOther];
				elemsOther[idxOther] = tmp;
				idx += stride;
				idxOther += strideOther;
			}
		}
	}

	public void toArray(float[] values) {
		if (values.length < size)
			throw new IllegalArgumentException("values too small");
		if (this.isNoView)
			System.arraycopy(this.elements, 0, values, 0, this.elements.length);
		else
			super.toArray(values);
	}

	public float zDotProduct(FloatMatrix1D y, int from, int length) {
		if (!(y instanceof DenseFloatMatrix1D)) {
			return super.zDotProduct(y, from, length);
		}
		DenseFloatMatrix1D yy = (DenseFloatMatrix1D) y;

		int tail = from + length;
		if (from < 0 || length < 0)
			return 0;
		if (size < tail)
			tail = size;
		if (y.size() < tail)
			tail = y.size();
		final float[] elemsOther = yy.elements;
		int zeroThis = index(from);
		int zeroOther = yy.index(from);
		int strideOther = yy.stride;
		if (elements == null || elemsOther == null)
			throw new InternalError();
		float sum = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (length >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			final int zeroThisF = zeroThis;
			final int zeroOtherF = zeroOther;
			final int strideOtherF = strideOther;
			Future[] futures = new Future[np];
			Float[] results = new Float[np];
			int k = length / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = length;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {
					public Float call() throws Exception {
						int idx = zeroThisF + startidx * stride;
						int idxOther = zeroOtherF + startidx * strideOtherF;
						idx -= stride;
						idxOther -= strideOtherF;
						float sum = 0;
						int min = stopidx - startidx;
						for (int k = min / 4; --k >= 0;) {
							sum += elements[idx += stride] * elemsOther[idxOther += strideOtherF] + elements[idx += stride] * elemsOther[idxOther += strideOtherF] + elements[idx += stride] * elemsOther[idxOther += strideOtherF] + elements[idx += stride] * elemsOther[idxOther += strideOtherF];
						}
						for (int k = min % 4; --k >= 0;) {
							sum += elements[idx += stride] * elemsOther[idxOther += strideOtherF];
						}
						return sum;
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (Float) futures[j].get();
				}
				sum = results[0];
				for (int j = 1; j < np; j++) {
					sum += results[j];
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			zeroThis -= stride;
			zeroOther -= strideOther;
			int min = tail - from;
			for (int k = min / 4; --k >= 0;) {
				sum += elements[zeroThis += stride] * elemsOther[zeroOther += strideOther] + elements[zeroThis += stride] * elemsOther[zeroOther += strideOther] + elements[zeroThis += stride] * elemsOther[zeroOther += strideOther] + elements[zeroThis += stride]
						* elemsOther[zeroOther += strideOther];
			}
			for (int k = min % 4; --k >= 0;) {
				sum += elements[zeroThis += stride] * elemsOther[zeroOther += strideOther];
			}
		}
		return sum;
	}

	public float zSum() {
		float sum = 0;
		final float[] elems = this.elements;
		if (elems == null)
			throw new InternalError();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			Future[] futures = new Future[np];
			Float[] results = new Float[np];
			int k = size / np;
			for (int j = 0; j < np; j++) {
				final int startidx = j * k;
				final int stopidx;
				if (j == np - 1) {
					stopidx = size;
				} else {
					stopidx = startidx + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {
					public Float call() throws Exception {
						float sum = 0;
						int idx = zero + startidx * stride;
						for (int i = startidx; i < stopidx; i++) {
							sum += elems[idx];
							idx += stride;
						}
						return Float.valueOf(sum);
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (Float) futures[j].get();
				}
				sum = results[0];
				for (int j = 1; j < np; j++) {
					sum += results[j];
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			int idx = zero;
			for (int k = 0; k < size; k++) {
				sum += elems[idx];
				idx += stride;
			}
		}
		return sum;
	}

	protected int cardinality(int maxCardinality) {
		int cardinality = 0;
		int index = zero;
		float[] elems = this.elements;
		int i = size;
		while (--i >= 0 && cardinality < maxCardinality) {
			if (elems[index] != 0)
				cardinality++;
			index += stride;
		}
		return cardinality;
	}

	protected boolean haveSharedCellsRaw(FloatMatrix1D other) {
		if (other instanceof SelectedDenseFloatMatrix1D) {
			SelectedDenseFloatMatrix1D otherMatrix = (SelectedDenseFloatMatrix1D) other;
			return this.elements == otherMatrix.elements;
		} else if (other instanceof DenseFloatMatrix1D) {
			DenseFloatMatrix1D otherMatrix = (DenseFloatMatrix1D) other;
			return this.elements == otherMatrix.elements;
		}
		return false;
	}

	public int index(int rank) {
		return zero + rank * stride;
	}

	protected FloatMatrix1D viewSelectionLike(int[] offsets) {
		return new SelectedDenseFloatMatrix1D(this.elements, offsets);
	}
}
