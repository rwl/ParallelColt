/*
 * Copyright (C) 2010-2012 Richard Lincoln
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

package edu.emory.mathcs.utils;

import java.util.concurrent.Future;

import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdcomplex.DComplexFactory1D;
import cern.colt.matrix.tdcomplex.DComplexFactory2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tint.IntFactory1D;
import cern.colt.matrix.tint.IntFactory2D;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import cern.jet.math.tdcomplex.DComplexFunctions;
import cern.jet.math.tdouble.DoubleFunctions;
import cern.jet.math.tint.IntFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Utility class for translating Octave/Numpy code to Colt.
 *
 * @author Richard Lincoln
 */
@SuppressWarnings("static-access")
public class Utils extends Object {

	public static final IntFunctions ifunc = IntFunctions.intFunctions;
	public static final DoubleFunctions dfunc = DoubleFunctions.functions;
	public static final DComplexFunctions cfunc = DComplexFunctions.functions;

	public static final double[] j = {0.0, 1.0};
	
	public static final double[] CZERO = {0.0, 0.0};
	public static final double[] CONE = {1.0, 0.0};
	public static final double[] CNEG_ONE = {-1.0, 0.0};

	public static final Utils util = new Utils();

	/**
	 * Makes this class non instantiable, but still let's others inherit from
	 * it.
	 */
	protected Utils() {
	}

	/**
	 * Machine epsilon.
	 */
	public static final double EPS;

	static {
		double d = 0.5;
		while (1 + d > 1) d /= 2;
		EPS = d;
	}

	/**
	 *
	 * @param stop
	 * @return
	 */
	public static int[] irange(int stop) {
		return irange(0, stop);
	}

	/**
	 *
	 * @param start
	 * @param stop
	 * @return
	 */
	public static int[] irange(int start, int stop) {
		return irange(start, stop, 1);
	}

	/**
	 *
	 * @param start
	 * @param stop
	 * @param step
	 * @return
	 */
	public static int[] irange(int start, int stop, int step) {
		int[] r = new int[stop - start];
		int v = start;
		for (int i = 0; i < r.length; i++) {
			r[i] = v;
			v += step;
		}
		return r;
	}

	/**
	 *
	 * @param stop
	 * @return
	 */
	public static double[] drange(int stop) {
		return drange(0, stop);
	}

	/**
	 *
	 * @param start
	 * @param stop
	 * @return
	 */
	public static double[] drange(int start, int stop) {
		return drange(start, stop, 1);
	}

	/**
	 *
	 * @param start
	 * @param stop
	 * @param step
	 * @return
	 */
	public static double[] drange(int start, int stop, int step) {
		double[] r = new double[stop - start];
		int v = start;
		for (int i = 0; i < r.length; i++) {
			r[i] = v;
			v += step;
		}
		return r;
	}

	/**
	 *
	 * @param stop
	 * @return an arithmetic progression.
	 */
	public static double[] drange(double stop) {
		return drange(0, stop, 1);
	}

	/**
	 *
	 * @param start
	 * @param stop an arithmetic progression.
	 * @return
	 */
	public static double[] drange(double start, double stop) {
		return drange(start, stop, 1);
	}

	/**
	 *
	 * @param start
	 * @param stop
	 * @param step increment (or decrement)
	 * @return an arithmetic progression.
	 */
	public static double[] drange(double start, double stop, double step) {
		double[] r = new double[(int) ((stop - start) / step)];
		double v = start;
		for (int i = 0; i < r.length; i++) {
			r[i] = v;
			v += step;
		}
		return r;
	}

	/**
	 *
	 * @param n
	 * @return
	 */
	public static int[] zeros(int size) {
		final int[] values = new int[size];
		int nthreads = ConcurrencyUtils.getNumberOfThreads();
		if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			nthreads = Math.min(nthreads, size);
			Future<?>[] futures = new Future[nthreads];
			int k = size / nthreads;
			for (int j = 0; j < nthreads; j++) {
				final int firstIdx = j * k;
				final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
				futures[j] = ConcurrencyUtils.submit(new Runnable() {
					public void run() {
						for (int i = firstIdx; i < lastIdx; i++) {
							values[i] = 0;
						}
					}
				});
			}
			ConcurrencyUtils.waitForCompletion(futures);
		} else {
			for (int i = 0; i < size; i++) {
				values[i] = 0;
			}
		}
		return values;
	}

	/**
	 *
	 * @param size array length
	 * @return an integer array with all elements = 1.
	 */
	public static int[] ones(int size) {
		final int[] values = new int[size];
		int nthreads = ConcurrencyUtils.getNumberOfThreads();
		if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			nthreads = Math.min(nthreads, size);
			Future<?>[] futures = new Future[nthreads];
			int k = size / nthreads;
			for (int j = 0; j < nthreads; j++) {
				final int firstIdx = j * k;
				final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
				futures[j] = ConcurrencyUtils.submit(new Runnable() {
					public void run() {
						for (int i = firstIdx; i < lastIdx; i++) {
							values[i] = 1;
						}
					}
				});
			}
			ConcurrencyUtils.waitForCompletion(futures);
		} else {
			for (int i = 0; i < size; i++) {
				values[i] = 1;
			}
		}
		return values;
	}

	/**
	 *
	 * @param d
	 * @return
	 */
	public static int[] inta(final DoubleMatrix1D d) {
		int size = (int) d.size();
		final int[] values = new int[size];
		int nthreads = ConcurrencyUtils.getNumberOfThreads();
		if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			nthreads = Math.min(nthreads, size);
			Future<?>[] futures = new Future[nthreads];
			int k = size / nthreads;
			for (int j = 0; j < nthreads; j++) {
				final int firstIdx = j * k;
				final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
				futures[j] = ConcurrencyUtils.submit(new Runnable() {
					public void run() {
						for (int i = firstIdx; i < lastIdx; i++) {
							values[i] = (int) Math.round(d.getQuick(i));
						}
					}
				});
			}
			ConcurrencyUtils.waitForCompletion(futures);
		} else {
			for (int i = 0; i < size; i++) {
				values[i] = (int) d.getQuick(i);
			}
		}
		return values;
	}

	/**
	 *
	 * @param d
	 * @return
	 */
	public static IntMatrix1D intm(final DoubleMatrix1D d) {
		int size = (int) d.size();
		final IntMatrix1D values = IntFactory1D.dense.make(size);
		int nthreads = ConcurrencyUtils.getNumberOfThreads();
		if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			nthreads = Math.min(nthreads, size);
			Future<?>[] futures = new Future[nthreads];
			int k = size / nthreads;
			for (int j = 0; j < nthreads; j++) {
				final int firstIdx = j * k;
				final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
				futures[j] = ConcurrencyUtils.submit(new Runnable() {
					public void run() {
						for (int i = firstIdx; i < lastIdx; i++) {
							values.setQuick(i, (int) Math.round( d.getQuick(i)) );
						}
					}
				});
			}
			ConcurrencyUtils.waitForCompletion(futures);
		} else {
			for (int i = 0; i < size; i++) {
				values.setQuick(i, (int) d.getQuick(i));
			}
		}
		return values;
	}

	/**
	 *
	 * @param d
	 * @return
	 */
	public static DoubleMatrix1D dbla(final int[] ix) {
		int size = ix.length;
		final DoubleMatrix1D values = DoubleFactory1D.dense.make(size);
		int nthreads = ConcurrencyUtils.getNumberOfThreads();
		if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			nthreads = Math.min(nthreads, size);
			Future<?>[] futures = new Future[nthreads];
			int k = size / nthreads;
			for (int j = 0; j < nthreads; j++) {
				final int firstIdx = j * k;
				final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
				futures[j] = ConcurrencyUtils.submit(new Runnable() {
					public void run() {
						for (int i = firstIdx; i < lastIdx; i++) {
							values.setQuick(i, ix[i]);
						}
					}
				});
			}
			ConcurrencyUtils.waitForCompletion(futures);
		} else {
			for (int i = 0; i < size; i++) {
				values.setQuick(i, ix[i]);
			}
		}
		return values;
	}

	/**
	 *
	 * @param d
	 * @return
	 */
	public static DoubleMatrix1D dblm(final IntMatrix1D ix) {
		int size = (int) ix.size();
		final DoubleMatrix1D values = DoubleFactory1D.dense.make(size);
		int nthreads = ConcurrencyUtils.getNumberOfThreads();
		if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			nthreads = Math.min(nthreads, size);
			Future<?>[] futures = new Future[nthreads];
			int k = size / nthreads;
			for (int j = 0; j < nthreads; j++) {
				final int firstIdx = j * k;
				final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
				futures[j] = ConcurrencyUtils.submit(new Runnable() {
					public void run() {
						for (int i = firstIdx; i < lastIdx; i++) {
							values.setQuick(i, ix.getQuick(i));
						}
					}
				});
			}
			ConcurrencyUtils.waitForCompletion(futures);
		} else {
			for (int i = 0; i < size; i++) {
				values.setQuick(i, ix.getQuick(i));
			}
		}
		return values;
	}

	/**
	 *
	 * @param d
	 * @return
	 */
	public static DComplexMatrix1D cplxm(final IntMatrix1D ix) {
		int size = (int) ix.size();
		final DComplexMatrix1D values = DComplexFactory1D.dense.make(size);
		int nthreads = ConcurrencyUtils.getNumberOfThreads();
		if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
			nthreads = Math.min(nthreads, size);
			Future<?>[] futures = new Future[nthreads];
			int k = size / nthreads;
			for (int j = 0; j < nthreads; j++) {
				final int firstIdx = j * k;
				final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
				futures[j] = ConcurrencyUtils.submit(new Runnable() {
					public void run() {
						for (int i = firstIdx; i < lastIdx; i++) {
							values.setQuick(i, ix.getQuick(i), 0);
						}
					}
				});
			}
			ConcurrencyUtils.waitForCompletion(futures);
		} else {
			for (int i = 0; i < size; i++) {
				values.setQuick(i, ix.getQuick(i), 0);
			}
		}
		return values;
	}
	
	/**
	 * 
	 * @param a
	 * @return
	 */
	public static double max(DoubleMatrix1D a) {
		return a.aggregate(dfunc.max, dfunc.identity);
	}
	
	/**
	 * 
	 * @param a
	 * @return
	 */
	public static double min(DoubleMatrix1D a) {
		return a.aggregate(dfunc.min, dfunc.identity);
	}

	/**
	 *
	 * @param t
	 * @return
	 */
	public static int max(int[] t) {
		int maximum = t[0];
		for (int i=1; i < t.length; i++)
			if (t[i] > maximum)
				maximum = t[i];
		return maximum;
	}

	/**
	 *
	 * @param a
	 * @param b
	 * @return
	 */
	public static int[] icat(int[] a, int[] b) {
		int[] c = new int[a.length + b.length];
		System.arraycopy(a, 0, c, 0, a.length);
		System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}

	/**
	 *
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] dcat(double[] a, double[] b) {
		double[] c = new double[a.length + b.length];
		System.arraycopy(a, 0, c, 0, a.length);
		System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}

	/**
	 *
	 * @param a
	 * @param b
	 * @return
	 */
	public static String[] scat(String[] a, String[] b) {
		String[] c = new String[a.length + b.length];
		System.arraycopy(a, 0, c, 0, a.length);
		System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}

	/**
	 *
	 * @param a
	 * @return
	 */
	public static int[] nonzero(IntMatrix1D a) {
		IntArrayList indexList = new IntArrayList();
		int size = (int) a.size();
		int rem = size % 2;
		if (rem == 1) {
			int value = a.getQuick(0);
			if (value != 0)
				indexList.add(0);
		}

		for (int i = rem; i < size; i += 2) {
			int value = a.getQuick(i);
			if (value != 0)
				indexList.add(i);
			value = a.getQuick(i + 1);
			if (value != 0)
				indexList.add(i + 1);
		}
		indexList.trimToSize();
		return indexList.elements();
	}

	/**
	 *
	 * @param a
	 * @return
	 */
	public static int[] nonzero(DoubleMatrix1D a) {
		IntArrayList indexList = new IntArrayList();
		int size = (int) a.size();
		int rem = size % 2;
		if (rem == 1) {
			double value = a.getQuick(0);
			if (value != 0)
				indexList.add(0);
		}

		for (int i = rem; i < size; i += 2) {
			double value = a.getQuick(i);
			if (value != 0)
				indexList.add(i);
			value = a.getQuick(i + 1);
			if (value != 0)
				indexList.add(i + 1);
		}
		indexList.trimToSize();
		return indexList.elements();
	}

	/**
	 *
	 * @param r polar radius.
	 * @param theta polar angle in radians.
	 * @return complex polar representation.
	 */
	public static DComplexMatrix1D polar(DoubleMatrix1D r, DoubleMatrix1D theta) {
		return polar(r, theta, true);
	}

	/**
	 *
	 * @param r polar radius.
	 * @param theta polar angle.
	 * @param radians is 'theta' expressed in radians.
	 * @return complex polar representation.
	 */
	public static DComplexMatrix1D polar(DoubleMatrix1D r, DoubleMatrix1D theta, boolean radians) {
		DoubleMatrix1D real = theta.copy();
		DoubleMatrix1D imag = theta.copy();
		if (!radians) {
			real.assign(dfunc.chain(dfunc.mult(Math.PI), dfunc.div(180)));
			imag.assign(dfunc.chain(dfunc.mult(Math.PI), dfunc.div(180)));
		}
		real.assign(dfunc.cos);
		imag.assign(dfunc.sin);
		real.assign(r, dfunc.mult);
		imag.assign(r, dfunc.mult);

		DComplexMatrix1D cmplx = DComplexFactory1D.dense.make((int) r.size());
		cmplx.assignReal(real);
		cmplx.assignImaginary(imag);

		return cmplx;
	}

	public static DComplexMatrix2D polar(DoubleMatrix2D r, DoubleMatrix2D theta) {
		return polar(r, theta, true);
	}

	public static DComplexMatrix2D polar(DoubleMatrix2D r, DoubleMatrix2D theta, boolean radians) {
		DoubleMatrix2D real = theta.copy();
		DoubleMatrix2D imag = theta.copy();
		if (!radians) {
			real.assign(dfunc.chain(dfunc.mult(Math.PI), dfunc.div(180)));
			imag.assign(dfunc.chain(dfunc.mult(Math.PI), dfunc.div(180)));
		}
		real.assign(dfunc.cos);
		imag.assign(dfunc.sin);
		real.assign(r, dfunc.mult);
		imag.assign(r, dfunc.mult);

		DComplexMatrix2D cmplx = DComplexFactory2D.dense.make(r.rows(), r.columns());
		cmplx.assignReal(real);
		cmplx.assignImaginary(imag);

		return cmplx;
	}

	/**
	 *
	 * @param x
	 * @return [x(1)-x(0)  x(2)-x(1) ... x(n)-x(n-1)]
	 */
	public static IntMatrix1D diff(IntMatrix1D x) {
		int size = (int) x.size() -1;
		IntMatrix1D d = IntFactory1D.dense.make(size);
		for (int i = 0; i < size; i++)
			d.set(i, ifunc.minus.apply(x.get(i+1), x.get(i)));
		return d;
	}

	/**
	 *
	 * @param x
	 * @return [x(1)-x(0)  x(2)-x(1) ... x(n)-x(n-1)]
	 */
	public static DoubleMatrix1D diff(DoubleMatrix1D x) {
		int size = (int) x.size() -1;
		DoubleMatrix1D d = DoubleFactory1D.dense.make(size);
		for (int i = 0; i < size; i++)
			d.set(i, dfunc.minus.apply(x.get(i+1), x.get(i)));
		return d;
	}

	/**
	 *
	 * @param x an array of integers.
	 * @return true if any element of vector x is a nonzero number.
	 */
	public static boolean any(int[] x) {
		for (int i : x)
			if (i != 0)
				return true;
		return false;
	}

	/**
	 *
	 * @param x a vector of integers.
	 * @return true if any element of vector x is a nonzero number.
	 */
	public static boolean any(IntMatrix1D x) {
		IntArrayList indexList = new IntArrayList();
		x.getNonZeros(indexList, new IntArrayList());
		return indexList.size() > 0;
	}

	/**
	 *
	 * @param x a vector of doubles.
	 * @return true if any element of vector x is a nonzero number.
	 */
	public static boolean any(DoubleMatrix1D x) {
		IntArrayList indexList = new IntArrayList();
		x.getNonZeros(indexList, new DoubleArrayList());
		return indexList.size() > 0;
	}

	/**
	 *
	 * @param x
	 * @return
	 */
	public static IntMatrix1D any(DoubleMatrix2D x) {
		int cols = x.columns();
		IntMatrix1D y = IntFactory1D.dense.make(cols);
		for (int i = 0; i < cols; i++) {
			int a = any(x.viewColumn(i)) ? 1 : 0;
			y.set(i, a);
		}
		return y;
	}

	/**
	 *
	 * @param x a vector of integers.
	 * @return true if all elements of 'x' are nonzero.
	 */
	public static boolean all(IntMatrix1D x) {
		IntArrayList indexList = new IntArrayList();
		x.getNonZeros(indexList, null);
		return x.size() == indexList.size();
	}

	/**
	 *
	 * @param x a vector of doubles.
	 * @return true if all elements of 'x' are nonzero.
	 */
	public static boolean all(DoubleMatrix1D x) {
		IntArrayList indexList = new IntArrayList();
		x.getNonZeros(indexList, null);
		return x.size() == indexList.size();
	}

	/**
	 *
	 * @param real real component, may be null
	 * @param imaginary imaginary component, may be null
	 * @return a complex vector
	 */
	public static DComplexMatrix1D complex(DoubleMatrix1D real, DoubleMatrix1D imaginary) {
		int size = 0;
		if (real != null)
			size = (int) real.size();
		if (imaginary != null)
			size = (int) imaginary.size();

		DComplexMatrix1D cmplx = DComplexFactory1D.dense.make(size);

		if (real != null)
			cmplx.assignReal(real);
		if (imaginary != null)
			cmplx.assignImaginary(imaginary);

		return cmplx;
	}

	/**
	 *
	 * @param real real component, may be null
	 * @param imaginary imaginary component, may be null
	 * @return a complex matrix
	 */
	public static DComplexMatrix2D complex(DoubleMatrix2D real, DoubleMatrix2D imaginary) {
		DComplexMatrix2D cmplx = DComplexFactory2D.dense.make(real.rows(), real.columns());
		if (real != null)
			cmplx.assignReal(real);
		if (imaginary != null)
			cmplx.assignImaginary(imaginary);
		return cmplx;
	}

	/**
	 *
	 * @param rows
	 * @param cols
	 * @param I
	 * @param J
	 * @return
	 */
	public static IntMatrix1D sub2ind(int rows, int cols, IntMatrix1D I, IntMatrix1D J) {
		return sub2ind(rows, cols, I, J, true);
	}

	/**
	 *
	 * @param rows
	 * @param cols
	 * @param I
	 * @param J
	 * @param row_major
	 * @return
	 */
	public static IntMatrix1D sub2ind(int rows, int cols, IntMatrix1D I, IntMatrix1D J, boolean row_major) {
		IntMatrix1D ind;
		if (row_major) {
			ind = I.copy().assign(ifunc.mod(rows)).assign(ifunc.mult(cols)).assign(J.copy().assign(ifunc.mod(cols)), ifunc.plus);
		} else {
			ind = J.copy().assign(ifunc.mod(cols)).assign(ifunc.max(rows)).assign(I.copy().assign(ifunc.mod(rows)), ifunc.plus);
		}
		return ind;
	}

	/**
	 * Appends a value to the given matrix.
	 *
	 * @param x input matrix of length n
	 * @param a value to append
	 * @return new matrix of length n + 1
	 */
	public static DoubleMatrix1D append(DoubleMatrix1D x, double a) {
		int n = (int) x.size();
		DoubleMatrix1D y = DoubleFactory1D.dense.make(n + 1);
		y.viewPart(0, n).assign(x);
		y.setQuick(n, a);
		return y;
	}

	/**
	 * Creates a 2-dimensional copy of the given matrix
	 *
	 * @param a 1D input matrix
	 * @param col create a column vector?
	 * @return a 2D copy of a
	 */
	public static DoubleMatrix2D unflatten(DoubleMatrix1D a, boolean col) {
		int nrow, ncol;

		if (col) {
			nrow = (int) a.size();
			ncol = 1;
		} else {
			nrow = 1;
			ncol = (int) a.size();
		}

		DoubleMatrix2D b = DoubleFactory2D.dense.make( nrow, ncol );
		b.assign( a.toArray() );

		return b;
	}

	/**
	 * Creates a 2-dimensional copy of the given matrix
	 *
	 * @param a 1D input matrix
	 * @param col create a column vector?
	 * @return a 2D copy of a
	 */
	public static DComplexMatrix2D unflatten(DComplexMatrix1D a, boolean col) {
		int nrow, ncol;

		if (col) {
			nrow = (int) a.size();
			ncol = 1;
		} else {
			nrow = 1;
			ncol = (int) a.size();
		}

		DComplexMatrix2D b = DComplexFactory2D.dense.make( nrow, ncol );
		b.assign( a.toArray() );

		return b;
	}

	public static IntMatrix2D unflatten(IntMatrix1D a) {
		return unflatten( a, true );
	}

	public static IntMatrix2D unflatten(IntMatrix1D a, boolean col) {
		int nrow, ncol;

		if (col) {
			nrow = (int) a.size();
			ncol = 1;
		} else {
			nrow = 1;
			ncol = (int) a.size();
		}

		IntMatrix2D b = IntFactory2D.dense.make( nrow, ncol );
		b.assign( a.toArray() );

		return b;
	}

	public static DoubleMatrix2D delete(DoubleMatrix2D m, int idx) {
		return delete(m, idx, 0);
	}

	public static DoubleMatrix2D delete(DoubleMatrix2D m, int idx, int axis) {
		int[] irow, icol;

		if (axis == 0) {
			irow = icat(irange(idx), irange(idx + 1, m.rows()));
			icol = null;
		} else {
			irow = null;
			icol = icat(irange(idx), irange(idx + 1, m.columns()));
		}

		return m.viewSelection(irow, icol).copy();
	}

}
