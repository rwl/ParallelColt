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
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import edu.emory.mathcs.jtransforms.dct.FloatDCT_3D;
import edu.emory.mathcs.jtransforms.dht.FloatDHT_3D;
import edu.emory.mathcs.jtransforms.dst.FloatDST_3D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 3-d matrix holding <tt>float</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contiguous one-dimensional array, addressed in
 * (in decreasing order of significance): slice major, row major, column major.
 * Note that this implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*slices()*rows()*columns()</tt>. Thus, a 100*100*100
 * matrix uses 8 MB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 * <p>
 * Applications demanding utmost speed can exploit knowledge about the internal
 * addressing. Setting/getting values in a loop slice-by-slice, row-by-row,
 * column-by-column is quicker than, for example, column-by-column, row-by-row,
 * slice-by-slice. Thus
 * 
 * <pre>
 * for (int slice = 0; slice &lt; slices; slice++) {
 * 	for (int row = 0; row &lt; rows; row++) {
 * 		for (int column = 0; column &lt; columns; column++) {
 * 			matrix.setQuick(slice, row, column, someValue);
 * 		}
 * 	}
 * }
 * </pre>
 * 
 * is quicker than
 * 
 * <pre>
 * for (int column = 0; column &lt; columns; column++) {
 * 	for (int row = 0; row &lt; rows; row++) {
 * 		for (int slice = 0; slice &lt; slices; slice++) {
 * 			matrix.setQuick(slice, row, column, someValue);
 * 		}
 * 	}
 * }
 * </pre>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DenseFloatMatrix3D extends FloatMatrix3D {
	private static final long serialVersionUID = 5711401505315728697L;

	private FloatFFT_3D fft3;

	private FloatDCT_3D dct3;

	private FloatDST_3D dst3;

	private FloatDHT_3D dht3;

	/**
	 * The elements of this matrix. elements are stored in slice major, then row
	 * major, then column major, in order of significance, i.e.
	 * index==slice*sliceStride+ row*rowStride + column*columnStride i.e.
	 * {slice0 row0..m}, {slice1 row0..m}, ..., {sliceN row0..m} with each row
	 * storead as {row0 column0..m}, {row1 column0..m}, ..., {rown column0..m}
	 */
	protected float[] elements;

	/**
	 * Constructs a matrix with a copy of the given values. <tt>values</tt> is
	 * required to have the form <tt>values[slice][row][column]</tt> and have
	 * exactly the same number of rows in in every slice and exactly the same
	 * number of columns in in every row.
	 * <p>
	 * The values are copied. So subsequent changes in <tt>values</tt> are not
	 * reflected in the matrix, and vice-versa.
	 * 
	 * @param values
	 *            The values to be filled into the new matrix.
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>for any 1 &lt;= slice &lt; values.length: values[slice].length != values[slice-1].length</tt>
	 *             .
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>for any 1 &lt;= row &lt; values[0].length: values[slice][row].length != values[slice][row-1].length</tt>
	 *             .
	 */
	public DenseFloatMatrix3D(float[][][] values) {
		this(values.length, (values.length == 0 ? 0 : values[0].length), (values.length == 0 ? 0 : values[0].length == 0 ? 0 : values[0][0].length));
		assign(values);
	}

	/**
	 * Constructs a matrix with a given number of slices, rows and columns. All
	 * entries are initially <tt>0</tt>.
	 * 
	 * @param slices
	 *            the number of slices the matrix shall have.
	 * @param rows
	 *            the number of rows the matrix shall have.
	 * @param columns
	 *            the number of columns the matrix shall have.
	 * @throws IllegalArgumentException
	 *             if <tt>(float)slices*columns*rows > Integer.MAX_VALUE</tt>.
	 * @throws IllegalArgumentException
	 *             if <tt>slices<0 || rows<0 || columns<0</tt>.
	 */
	public DenseFloatMatrix3D(int slices, int rows, int columns) {
		setUp(slices, rows, columns);
		this.elements = new float[slices * rows * columns];
	}

	/**
	 * Constructs a view with the given parameters.
	 * 
	 * @param slices
	 *            the number of slices the matrix shall have.
	 * @param rows
	 *            the number of rows the matrix shall have.
	 * @param columns
	 *            the number of columns the matrix shall have.
	 * @param elements
	 *            the cells.
	 * @param sliceZero
	 *            the position of the first element.
	 * @param rowZero
	 *            the position of the first element.
	 * @param columnZero
	 *            the position of the first element.
	 * @param sliceStride
	 *            the number of elements between two slices, i.e.
	 *            <tt>index(k+1,i,j)-index(k,i,j)</tt>.
	 * @param rowStride
	 *            the number of elements between two rows, i.e.
	 *            <tt>index(k,i+1,j)-index(k,i,j)</tt>.
	 * @param columnStride
	 *            the number of elements between two columns, i.e.
	 *            <tt>index(k,i,j+1)-index(k,i,j)</tt>.
	 * @throws IllegalArgumentException
	 *             if <tt>(float)slices*columns*rows > Integer.MAX_VALUE</tt>.
	 * @throws IllegalArgumentException
	 *             if <tt>slices<0 || rows<0 || columns<0</tt>.
	 */
	public DenseFloatMatrix3D(int slices, int rows, int columns, float[] elements, int sliceZero, int rowZero, int columnZero, int sliceStride, int rowStride, int columnStride) {
		setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
		this.elements = elements;
		this.isNoView = false;
	}

	public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFunction f) {
		if (size() == 0)
			return Float.NaN;
		float a = 0;
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
			Future[] futures = new Future[np];
			Float[] results = new Float[np];
			int k = slices / np;
			for (int j = 0; j < np; j++) {
				final int startslice = j * k;
				final int stopslice;
				if (j == np - 1) {
					stopslice = slices;
				} else {
					stopslice = startslice + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {

					public Float call() throws Exception {
						float a = f.apply(elements[zero + startslice * sliceStride]);
						int d = 1;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								for (int c = d; c < columns; c++) {
									a = aggr.apply(a, f.apply(elements[zero + s * sliceStride + r * rowStride + c * columnStride]));
								}
								d = 0;
							}
						}
						return a;
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (Float) futures[j].get();
				}
				a = results[0].floatValue();
				for (int j = 1; j < np; j++) {
					a = aggr.apply(a, results[j].floatValue());
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			a = f.apply(elements[zero]);
			int d = 1; // first cell already done
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					for (int c = d; c < columns; c++) {
						a = aggr.apply(a, f.apply(elements[zero + s * sliceStride + r * rowStride + c * columnStride]));
					}
					d = 0;
				}
			}
		}
		return a;
	}

	public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFunction f, final cern.colt.function.tfloat.FloatProcedure cond) {
		if (size() == 0)
			return Float.NaN;
		float a = 0;
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
			Future[] futures = new Future[np];
			Float[] results = new Float[np];
			int k = slices / np;
			for (int j = 0; j < np; j++) {
				final int startslice = j * k;
				final int stopslice;
				if (j == np - 1) {
					stopslice = slices;
				} else {
					stopslice = startslice + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {

					public Float call() throws Exception {
						float elem = elements[zero + startslice * sliceStride];
						float a = 0;
						if (cond.apply(elem) == true) {
							a = aggr.apply(a, f.apply(elem));
						}
						int d = 1;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								for (int c = d; c < columns; c++) {
									elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
									if (cond.apply(elem) == true) {
										a = aggr.apply(a, f.apply(elem));
									}
									d = 0;
								}
							}
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
			float elem = elements[zero];
			if (cond.apply(elem) == true) {
				a = aggr.apply(a, f.apply(elem));
			}
			int d = 1;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					for (int c = d; c < columns; c++) {
						elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
						if (cond.apply(elem) == true) {
							a = aggr.apply(a, f.apply(elem));
						}
						d = 0;
					}
				}
			}
		}
		return a;
	}

	public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFunction f, final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList) {
		if (size() == 0)
			return Float.NaN;
		if (sliceList.size() == 0 || rowList.size() == 0 || columnList.size() == 0)
			return Float.NaN;
		final int size = sliceList.size();
		final int[] sliceElements = sliceList.elements();
		final int[] rowElements = rowList.elements();
		final int[] columnElements = columnList.elements();
		final int zero = index(0, 0, 0);
		float a = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						float a = f.apply(elements[zero + sliceElements[startidx] * sliceStride + rowElements[startidx] * rowStride + columnElements[startidx] * columnStride]);
						float elem;
						for (int i = startidx + 1; i < stopidx; i++) {
							elem = elements[zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i] * columnStride];
							a = aggr.apply(a, f.apply(elem));
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
			a = f.apply(elements[zero + sliceElements[0] * sliceStride + rowElements[0] * rowStride + columnElements[0] * columnStride]);
			float elem;
			for (int i = 1; i < size; i++) {
				elem = elements[zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i] * columnStride];
				a = aggr.apply(a, f.apply(elem));
			}
		}
		return a;
	}

	public float aggregate(final FloatMatrix3D other, final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFloatFunction f) {
		if (!(other instanceof DenseFloatMatrix3D)) {
			return super.aggregate(other, aggr, f);
		}
		checkShape(other);
		if (size() == 0)
			return Float.NaN;
		float a = 0;
		final int zero = index(0, 0, 0);
		final int zeroOther = other.index(0, 0, 0);
		final int sliceStrideOther = other.sliceStride();
		final int rowStrideOther = other.rowStride();
		final int colStrideOther = other.columnStride();
		final float[] elemsOther = (float[]) other.elements();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
			Future[] futures = new Future[np];
			Float[] results = new Float[np];
			int k = slices / np;
			for (int j = 0; j < np; j++) {
				final int startslice = j * k;
				final int stopslice;
				if (j == np - 1) {
					stopslice = slices;
				} else {
					stopslice = startslice + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {
					public Float call() throws Exception {
						int idx = zero + startslice * sliceStride;
						int idxOther = zeroOther + startslice * sliceStrideOther;
						float a = f.apply(elements[idx], elemsOther[idxOther]);
						int d = 1;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								for (int c = d; c < columns; c++) {
									idx = zero + s * sliceStride + r * rowStride + c * columnStride;
									idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c * colStrideOther;
									a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
								}
								d = 0;
							}
						}
						return a;
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (Float) futures[j].get();
				}
				a = results[0].floatValue();
				for (int j = 1; j < np; j++) {
					a = aggr.apply(a, results[j].floatValue());
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			a = f.apply(getQuick(0, 0, 0), other.getQuick(0, 0, 0));
			int d = 1; // first cell already done
			int idx;
			int idxOther;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					for (int c = d; c < columns; c++) {
						idx = zero + s * sliceStride + r * rowStride + c * columnStride;
						idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther + c * colStrideOther;
						a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
					}
					d = 0;
				}
			}
		}
		return a;
	}

	public FloatMatrix3D assign(final cern.colt.function.tfloat.FloatFunction function) {
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								for (int c = 0; c < columns; c++) {
									elements[idx] = function.apply(elements[idx]);
									idx += columnStride;
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
			int idx;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					for (int c = 0; c < columns; c++) {
						elements[idx] = function.apply(elements[idx]);
						idx += columnStride;
					}
				}
			}
		}
		return this;
	}

	public FloatMatrix3D assign(final float value) {
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								for (int c = 0; c < columns; c++) {
									elements[idx] = value;
									idx += columnStride;
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
			int idx;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					for (int c = 0; c < columns; c++) {
						elements[idx] = value;
						idx += columnStride;
					}
				}
			}
		}
		return this;
	}

	public FloatMatrix3D assign(final float[] values) {
		if (values.length != size())
			throw new IllegalArgumentException("Must have same length: length=" + values.length + "slices()*rows()*columns()=" + slices() * rows() * columns());
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if (this.isNoView) {
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
				Future[] futures = new Future[np];
				int k = size() / np;
				for (int j = 0; j < np; j++) {
					final int startidx = j * k;
					final int length;
					if (j == np - 1) {
						length = size() - startidx;
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
			final int zero = index(0, 0, 0);
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
				Future[] futures = new Future[np];
				int k = slices / np;
				for (int j = 0; j < np; j++) {
					final int startslice = j * k;
					final int stopslice;
					final int glob_idx = j * k * rows * columns;
					if (j == np - 1) {
						stopslice = slices;
					} else {
						stopslice = startslice + k;
					}
					futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							int idxOther = glob_idx;
							int idx;
							for (int s = startslice; s < stopslice; s++) {
								for (int r = 0; r < rows; r++) {
									idx = zero + s * sliceStride + r * rowStride;
									for (int c = 0; c < columns; c++) {
										elements[idx] = values[idxOther++];
										idx += columnStride;
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
				int idxOther = 0;
				int idx;
				for (int s = 0; s < slices; s++) {
					for (int r = 0; r < rows; r++) {
						idx = zero + s * sliceStride + r * rowStride;
						for (int c = 0; c < columns; c++) {
							elements[idx] = values[idxOther++];
							idx += columnStride;
						}
					}
				}
			}
		}
		return this;
	}

	public FloatMatrix3D assign(final float[][][] values) {
		if (values.length != slices)
			throw new IllegalArgumentException("Must have same number of slices: slices=" + values.length + "slices()=" + slices());
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if (this.isNoView) {
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
							int i = startslice * sliceStride;
							for (int s = startslice; s < stopslice; s++) {
								float[][] currentSlice = values[s];
								if (currentSlice.length != rows)
									throw new IllegalArgumentException("Must have same number of rows in every slice: rows=" + currentSlice.length + "rows()=" + rows());
								for (int r = 0; r < rows; r++) {
									float[] currentRow = currentSlice[r];
									if (currentRow.length != columns)
										throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
									System.arraycopy(currentRow, 0, elements, i, columns);
									i += columns;
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
				int i = 0;
				for (int s = 0; s < slices; s++) {
					float[][] currentSlice = values[s];
					if (currentSlice.length != rows)
						throw new IllegalArgumentException("Must have same number of rows in every slice: rows=" + currentSlice.length + "rows()=" + rows());
					for (int r = 0; r < rows; r++) {
						float[] currentRow = currentSlice[r];
						if (currentRow.length != columns)
							throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
						System.arraycopy(currentRow, 0, this.elements, i, columns);
						i += columns;
					}
				}
			}
		} else {
			final int zero = index(0, 0, 0);
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
							for (int s = startslice; s < stopslice; s++) {
								float[][] currentSlice = values[s];
								if (currentSlice.length != rows)
									throw new IllegalArgumentException("Must have same number of rows in every slice: rows=" + currentSlice.length + "rows()=" + rows());
								for (int r = 0; r < rows; r++) {
									idx = zero + s * sliceStride + r * rowStride;
									float[] currentRow = currentSlice[r];
									if (currentRow.length != columns)
										throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
									for (int c = 0; c < columns; c++) {
										elements[idx] = currentRow[c];
										idx += columnStride;
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
				int idx;
				for (int s = 0; s < slices; s++) {
					float[][] currentSlice = values[s];
					if (currentSlice.length != rows)
						throw new IllegalArgumentException("Must have same number of rows in every slice: rows=" + currentSlice.length + "rows()=" + rows());
					for (int r = 0; r < rows; r++) {
						idx = zero + s * sliceStride + r * rowStride;
						float[] currentRow = currentSlice[r];
						if (currentRow.length != columns)
							throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
						for (int c = 0; c < columns; c++) {
							elements[idx] = currentRow[c];
							idx += columnStride;
						}
					}
				}
			}
		}
		return this;
	}

	public FloatMatrix3D assign(final cern.colt.function.tfloat.FloatProcedure cond, final cern.colt.function.tfloat.FloatFunction f) {
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						float elem;
						int idx;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								for (int c = 0; c < columns; c++) {
									elem = elements[idx];
									if (cond.apply(elem) == true) {
										elements[idx] = f.apply(elem);
									}
									idx += columnStride;
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
			float elem;
			int idx;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					for (int c = 0; c < columns; c++) {
						elem = elements[idx];
						if (cond.apply(elem) == true) {
							elements[idx] = f.apply(elem);
						}
						idx += columnStride;
					}
				}
			}
		}
		return this;
	}

	public FloatMatrix3D assign(final cern.colt.function.tfloat.FloatProcedure cond, final float value) {
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (slices * rows * columns >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						float elem;
						int idx;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								for (int c = 0; c < columns; c++) {
									elem = elements[idx];
									if (cond.apply(elem) == true) {
										elements[idx] = value;
									}
									idx += columnStride;
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
			float elem;
			int idx;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					for (int c = 0; c < columns; c++) {
						elem = elements[idx];
						if (cond.apply(elem) == true) {
							elements[idx] = value;
						}
						idx += columnStride;
					}
				}
			}
		}
		return this;
	}

	public FloatMatrix3D assign(FloatMatrix3D source) {
		// overriden for performance only
		if (!(source instanceof DenseFloatMatrix3D)) {
			super.assign(source);
			return this;
		}
		DenseFloatMatrix3D other = (DenseFloatMatrix3D) source;
		if (other == this)
			return this;
		checkShape(other);
		if (haveSharedCells(other)) {
			FloatMatrix3D c = other.copy();
			if (!(c instanceof DenseFloatMatrix3D)) { // should not happen
				super.assign(source);
				return this;
			}
			other = (DenseFloatMatrix3D) c;
		}

		final DenseFloatMatrix3D other_final = (DenseFloatMatrix3D) other;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if (this.isNoView && other.isNoView) { // quickest
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
				Future[] futures = new Future[np];
				int k = size() / np;
				for (int j = 0; j < np; j++) {
					final int startidx = j * k;
					final int length;
					if (j == np - 1) {
						length = size() - startidx;
					} else {
						length = k;
					}
					futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							System.arraycopy(other_final.elements, startidx, elements, startidx, length);
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
				System.arraycopy(other_final.elements, 0, this.elements, 0, this.elements.length);
			}
			return this;
		} else {
			final int zero = index(0, 0, 0);
			final int zeroOther = other_final.index(0, 0, 0);
			final int sliceStrideOther = other_final.sliceStride;
			final int rowStrideOther = other_final.rowStride;
			final int columnStrideOther = other_final.columnStride;
			final float[] elemsOther = other_final.elements;
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
								for (int r = 0; r < rows; r++) {
									idx = zero + s * sliceStride + r * rowStride;
									idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
									for (int c = 0; c < columns; c++) {
										elements[idx] = elemsOther[idxOther];
										idx += columnStride;
										idxOther += columnStrideOther;
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
				int idx;
				int idxOther;
				for (int s = 0; s < slices; s++) {
					for (int r = 0; r < rows; r++) {
						idx = zero + s * sliceStride + r * rowStride;
						idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
						for (int c = 0; c < columns; c++) {
							elements[idx] = elemsOther[idxOther];
							idx += columnStride;
							idxOther += columnStrideOther;
						}
					}
				}
			}
			return this;
		}
	}

	public FloatMatrix3D assign(final FloatMatrix3D y, final cern.colt.function.tfloat.FloatFloatFunction function) {
		if (!(y instanceof DenseFloatMatrix3D)) {
			super.assign(y, function);
			return this;
		}
		checkShape(y);
		final int zero = index(0, 0, 0);
		final int zeroOther = y.index(0, 0, 0);
		final int sliceStrideOther = y.sliceStride();
		final int rowStrideOther = y.rowStride();
		final int columnStrideOther = y.columnStride();
		final float[] elemsOther = (float[]) y.elements();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
								for (int c = 0; c < columns; c++) {
									elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
									idx += columnStride;
									idxOther += columnStrideOther;
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
			int idx;
			int idxOther;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					idxOther = zeroOther + s * sliceStrideOther + r * rowStrideOther;
					for (int c = 0; c < columns; c++) {
						elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
						idx += columnStride;
						idxOther += columnStrideOther;
					}
				}
			}
		}

		return this;
	}

	public FloatMatrix3D assign(final FloatMatrix3D y, final cern.colt.function.tfloat.FloatFloatFunction function, final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList) {
		if (!(y instanceof DenseFloatMatrix3D)) {
			super.assign(y, function);
			return this;
		}
		checkShape(y);
		final int zero = index(0, 0, 0);
		final int zeroOther = y.index(0, 0, 0);
		final int sliceStrideOther = y.sliceStride();
		final int rowStrideOther = y.rowStride();
		final int columnStrideOther = y.columnStride();
		final float[] elemsOther = (float[]) y.elements();
		int size = sliceList.size();
		final int[] sliceElements = sliceList.elements();
		final int[] rowElements = rowList.elements();
		final int[] columnElements = columnList.elements();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int i = startidx; i < stopidx; i++) {
							int idx = zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i] * columnStride;
							int idxOther = zeroOther + sliceElements[i] * sliceStrideOther + rowElements[i] * rowStrideOther + columnElements[i] * columnStrideOther;
							elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
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
			for (int i = 0; i < size; i++) {
				int idx = zero + sliceElements[i] * sliceStride + rowElements[i] * rowStride + columnElements[i] * columnStride;
				int idxOther = zeroOther + sliceElements[i] * sliceStrideOther + rowElements[i] * rowStrideOther + columnElements[i] * columnStrideOther;
				elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
			}
		}
		return this;
	}

	public int cardinality() {
		int cardinality = 0;
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
			Future[] futures = new Future[np];
			Integer[] results = new Integer[np];
			int k = slices / np;
			for (int j = 0; j < np; j++) {
				final int startslice = j * k;
				final int stopslice;
				if (j == np - 1) {
					stopslice = slices;
				} else {
					stopslice = startslice + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Integer>() {
					public Integer call() throws Exception {
						int cardinality = 0;
						int idx;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								for (int c = 0; c < columns; c++) {
									if (elements[idx] != 0) {
										cardinality++;
									}
									idx += columnStride;
								}
							}
						}
						return Integer.valueOf(cardinality);
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
			int idx;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					for (int c = 0; c < columns; c++) {
						if (elements[idx] != 0) {
							cardinality++;
						}
						idx += columnStride;
					}
				}
			}
		}
		return cardinality;
	}

	public void dct2Slices(final boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							viewSlice(s).dct2(scale);
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
			for (int s = 0; s < slices; s++) {
				viewSlice(s).dct2(scale);
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void dht3() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (dht3 == null) {
			dht3 = new FloatDHT_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			dht3.forward(elements);
		} else {
			FloatMatrix3D copy = this.copy();
			dht3.forward((float[]) copy.elements());
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void dht2Slices() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							viewSlice(s).dht2();
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
			for (int s = 0; s < slices; s++) {
				viewSlice(s).dht2();
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void dct3(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (dct3 == null) {
			dct3 = new FloatDCT_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			dct3.forward(elements, scale);
		} else {
			FloatMatrix3D copy = this.copy();
			dct3.forward((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void dst2Slices(final boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							viewSlice(s).dst2(scale);
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
			for (int s = 0; s < slices; s++) {
				viewSlice(s).dst2(scale);
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void dst3(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (dst3 == null) {
			dst3 = new FloatDST_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			dst3.forward(elements, scale);
		} else {
			FloatMatrix3D copy = this.copy();
			dst3.forward((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public float[] elements() {
		return elements;
	}

	public void fft3() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (fft3 == null) {
			fft3 = new FloatFFT_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			fft3.realForward(elements);
		} else {
			FloatMatrix3D copy = this.copy();
			fft3.realForward((float[]) copy.elements());
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public FComplexMatrix3D getFft2Slices() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		final FComplexMatrix3D C = new DenseFComplexMatrix3D(slices, rows, columns);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							C.viewSlice(s).assign(viewSlice(s).getFft2());
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
			for (int s = 0; s < slices; s++) {
				C.viewSlice(s).assign(viewSlice(s).getFft2());
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
		return C;
	}

	public FComplexMatrix3D getFft3() {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		FComplexMatrix3D C = new DenseFComplexMatrix3D(slices, rows, columns);
		final int sliceStride = rows * columns;
		final int rowStride = columns;
		final float[] elems;
		if (isNoView == true) {
			elems = elements;
		} else {
			elems = (float[]) this.copy().elements();
		}
		final float[] cElems = (float[]) ((DenseFComplexMatrix3D) C).elements();
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = s * sliceStride + r * rowStride;
								System.arraycopy(elems, idx, cElems, idx, columns);
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
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = s * sliceStride + r * rowStride;
					System.arraycopy(elems, idx, cElems, idx, columns);
				}
			}
		}
		if (fft3 == null) {
			fft3 = new FloatFFT_3D(slices, rows, columns);
		}
		fft3.realForwardFull(cElems);
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
		return C;
	}

	public FComplexMatrix3D getIfft2Slices(final boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		final FComplexMatrix3D C = new DenseFComplexMatrix3D(slices, rows, columns);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							C.viewSlice(s).assign(viewSlice(s).getIfft2(scale));
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
			for (int s = 0; s < slices; s++) {
				C.viewSlice(s).assign(viewSlice(s).getIfft2(scale));
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
		return C;
	}

	public FComplexMatrix3D getIfft3(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		FComplexMatrix3D C = new DenseFComplexMatrix3D(slices, rows, columns);
		final int sliceStride = rows * columns;
		final int rowStride = columns;
		final float[] cElems = (float[]) ((DenseFComplexMatrix3D) C).elements();
		final float[] elems;
		if (isNoView == true) {
			elems = elements;
		} else {
			elems = (float[]) this.copy().elements();
		}
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = s * sliceStride + r * rowStride;
								System.arraycopy(elems, idx, cElems, idx, columns);
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
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = s * sliceStride + r * rowStride;
					System.arraycopy(elems, idx, cElems, idx, columns);
				}
			}
		}
		if (fft3 == null) {
			fft3 = new FloatFFT_3D(slices, rows, columns);
		}
		fft3.realInverseFull(cElems, scale);
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
		return C;
	}

	public void getNegativeValues(final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList, final FloatArrayList valueList) {
		sliceList.clear();
		rowList.clear();
		columnList.clear();
		valueList.clear();
		int zero = index(0, 0, 0);

		int idx;
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) {
				idx = zero + s * sliceStride + r * rowStride;
				for (int c = 0; c < columns; c++) {
					float value = elements[idx];
					if (value < 0) {
						sliceList.add(s);
						rowList.add(r);
						columnList.add(c);
						valueList.add(value);
					}
					idx += columnStride;
				}
			}
		}

	}

	public void getNonZeros(final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList, final FloatArrayList valueList) {
		sliceList.clear();
		rowList.clear();
		columnList.clear();
		valueList.clear();
		int zero = index(0, 0, 0);

		int idx;
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) {
				idx = zero + s * sliceStride + r * rowStride;
				for (int c = 0; c < columns; c++) {
					float value = elements[idx];
					if (value != 0) {
						sliceList.add(s);
						rowList.add(r);
						columnList.add(c);
						valueList.add(value);
					}
					idx += columnStride;
				}
			}

		}
	}

	public void getPositiveValues(final IntArrayList sliceList, final IntArrayList rowList, final IntArrayList columnList, final FloatArrayList valueList) {
		sliceList.clear();
		rowList.clear();
		columnList.clear();
		valueList.clear();
		int zero = index(0, 0, 0);

		int idx;
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) {
				idx = zero + s * sliceStride + r * rowStride;
				for (int c = 0; c < columns; c++) {
					float value = elements[idx];
					if (value > 0) {
						sliceList.add(s);
						rowList.add(r);
						columnList.add(c);
						valueList.add(value);
					}
					idx += columnStride;
				}
			}
		}

	}

	public float getQuick(int slice, int row, int column) {
		return elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride];
	}

	public void idct2Slices(final boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							viewSlice(s).idct2(scale);
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
			for (int s = 0; s < slices; s++) {
				viewSlice(s).idct2(scale);
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void idht3(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (dht3 == null) {
			dht3 = new FloatDHT_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			dht3.inverse(elements, scale);
		} else {
			FloatMatrix3D copy = this.copy();
			dht3.inverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void idht2Slices(final boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							viewSlice(s).idht2(scale);
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
			for (int s = 0; s < slices; s++) {
				viewSlice(s).idht2(scale);
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void idct3(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (dct3 == null) {
			dct3 = new FloatDCT_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			dct3.inverse(elements, scale);
		} else {
			FloatMatrix3D copy = this.copy();
			dct3.inverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void idst2Slices(final boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							viewSlice(s).idst2(scale);
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
			for (int s = 0; s < slices; s++) {
				viewSlice(s).idst2(scale);
			}
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void idst3(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (dst3 == null) {
			dst3 = new FloatDST_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			dst3.inverse(elements, scale);
		} else {
			FloatMatrix3D copy = this.copy();
			dst3.inverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public void ifft3(boolean scale) {
		int oldNp = ConcurrencyUtils.getNumberOfProcessors();
		ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
		if (fft3 == null) {
			fft3 = new FloatFFT_3D(slices, rows, columns);
		}
		if (isNoView == true) {
			fft3.realInverse(elements, scale);
		} else {
			FloatMatrix3D copy = this.copy();
			fft3.realInverse((float[]) copy.elements(), scale);
			this.assign((float[]) copy.elements());
		}
		ConcurrencyUtils.setNumberOfProcessors(oldNp);
	}

	public int index(int slice, int row, int column) {
		return sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
	}

	public FloatMatrix3D like(int slices, int rows, int columns) {
		return new DenseFloatMatrix3D(slices, rows, columns);
	}

	public float[] getMaxLocation() {
		final int zero = index(0, 0, 0);
		int slice_loc = 0;
		int row_loc = 0;
		int col_loc = 0;
		float maxValue = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
			Future[] futures = new Future[np];
			float[][] results = new float[np][2];
			int k = slices / np;
			for (int j = 0; j < np; j++) {
				final int startslice = j * k;
				final int stopslice;
				if (j == np - 1) {
					stopslice = slices;
				} else {
					stopslice = startslice + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<float[]>() {
					public float[] call() throws Exception {
						int slice_loc = startslice;
						int row_loc = 0;
						int col_loc = 0;
						float maxValue = elements[zero + startslice * sliceStride];
						int d = 1;
						float elem;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								for (int c = d; c < columns; c++) {
									elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
									if (maxValue < elem) {
										maxValue = elem;
										slice_loc = s;
										row_loc = r;
										col_loc = c;
									}
								}
								d = 0;
							}
						}
						return new float[] { maxValue, slice_loc, row_loc, col_loc };
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (float[]) futures[j].get();
				}
				maxValue = results[0][0];
				slice_loc = (int) results[0][1];
				row_loc = (int) results[0][2];
				col_loc = (int) results[0][3];
				for (int j = 1; j < np; j++) {
					if (maxValue < results[j][0]) {
						maxValue = results[j][0];
						slice_loc = (int) results[j][1];
						row_loc = (int) results[j][2];
						col_loc = (int) results[j][3];
					}
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			maxValue = elements[zero];
			float elem;
			int d = 1;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					for (int c = d; c < columns; c++) {
						elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
						if (maxValue < elem) {
							maxValue = elem;
							slice_loc = s;
							row_loc = r;
							col_loc = c;
						}
					}
					d = 0;
				}
			}
		}
		return new float[] { maxValue, slice_loc, row_loc, col_loc };
	}

	public float[] getMinLocation() {
		final int zero = index(0, 0, 0);
		int slice_loc = 0;
		int row_loc = 0;
		int col_loc = 0;
		float minValue = 0;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
			Future[] futures = new Future[np];
			float[][] results = new float[np][2];
			int k = slices / np;
			for (int j = 0; j < np; j++) {
				final int startslice = j * k;
				final int stopslice;
				if (j == np - 1) {
					stopslice = slices;
				} else {
					stopslice = startslice + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<float[]>() {
					public float[] call() throws Exception {
						int slice_loc = startslice;
						int row_loc = 0;
						int col_loc = 0;
						float minValue = elements[zero + slice_loc * sliceStride];
						int d = 1;
						float elem;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								for (int c = d; c < columns; c++) {
									elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
									if (minValue > elem) {
										minValue = elem;
										slice_loc = s;
										row_loc = r;
										col_loc = c;
									}
								}
								d = 0;
							}
						}
						return new float[] { minValue, slice_loc, row_loc, col_loc };
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					results[j] = (float[]) futures[j].get();
				}
				minValue = results[0][0];
				slice_loc = (int) results[0][1];
				row_loc = (int) results[0][2];
				col_loc = (int) results[0][3];
				for (int j = 1; j < np; j++) {
					if (minValue > results[j][0]) {
						minValue = results[j][0];
						slice_loc = (int) results[j][1];
						row_loc = (int) results[j][2];
						col_loc = (int) results[j][3];
					}
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			minValue = elements[zero];
			float elem;
			int d = 1;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					for (int c = d; c < columns; c++) {
						elem = elements[zero + s * sliceStride + r * rowStride + c * columnStride];
						if (minValue > elem) {
							minValue = elem;
							slice_loc = s;
							row_loc = r;
							col_loc = c;
						}
					}
					d = 0;
				}
			}
		}
		return new float[] { minValue, slice_loc, row_loc, col_loc };
	}

	public void setQuick(int slice, int row, int column, float value) {
		elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride] = value;
	}

	public float[][][] toArray() {
		final float[][][] values = new float[slices][rows][columns];
		int np = ConcurrencyUtils.getNumberOfProcessors();
		final int zero = index(0, 0, 0);
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
						for (int s = startslice; s < stopslice; s++) {
							float[][] currentSlice = values[s];
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								float[] currentRow = currentSlice[r];
								for (int c = 0; c < columns; c++) {
									currentRow[c] = elements[idx];
									idx += columnStride;
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
			int idx;
			for (int s = 0; s < slices; s++) {
				float[][] currentSlice = values[s];
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					float[] currentRow = currentSlice[r];
					for (int c = 0; c < columns; c++) {
						currentRow[c] = elements[idx];
						idx += columnStride;
					}
				}
			}
		}
		return values;
	}

	public FloatMatrix1D vectorize() {
		FloatMatrix1D v = new DenseFloatMatrix1D(size());
		int length = rows * columns;
		for (int s = 0; s < slices; s++) {
			v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
		}
		return v;
	}

	public void zAssign27Neighbors(FloatMatrix3D B, cern.colt.function.tfloat.Float27Function function) {
		// overridden for performance only
		if (!(B instanceof DenseFloatMatrix3D)) {
			super.zAssign27Neighbors(B, function);
			return;
		}
		if (function == null)
			throw new NullPointerException("function must not be null.");
		checkShape(B);
		int r = rows - 1;
		int c = columns - 1;
		if (rows < 3 || columns < 3 || slices < 3)
			return; // nothing to do

		DenseFloatMatrix3D BB = (DenseFloatMatrix3D) B;
		int A_ss = sliceStride;
		int A_rs = rowStride;
		int B_rs = BB.rowStride;
		int A_cs = columnStride;
		int B_cs = BB.columnStride;
		float[] elems = this.elements;
		float[] B_elems = BB.elements;
		if (elems == null || B_elems == null)
			throw new InternalError();

		for (int k = 1; k < slices - 1; k++) {
			int A_index = index(k, 1, 1);
			int B_index = BB.index(k, 1, 1);

			for (int i = 1; i < r; i++) {
				int A002 = A_index - A_ss - A_rs - A_cs;
				int A012 = A002 + A_rs;
				int A022 = A012 + A_rs;

				int A102 = A002 + A_ss;
				int A112 = A102 + A_rs;
				int A122 = A112 + A_rs;

				int A202 = A102 + A_ss;
				int A212 = A202 + A_rs;
				int A222 = A212 + A_rs;

				float a000, a001, a002;
				float a010, a011, a012;
				float a020, a021, a022;

				float a100, a101, a102;
				float a110, a111, a112;
				float a120, a121, a122;

				float a200, a201, a202;
				float a210, a211, a212;
				float a220, a221, a222;

				a000 = elems[A002];
				A002 += A_cs;
				a001 = elems[A002];
				a010 = elems[A012];
				A012 += A_cs;
				a011 = elems[A012];
				a020 = elems[A022];
				A022 += A_cs;
				a021 = elems[A022];

				a100 = elems[A102];
				A102 += A_cs;
				a101 = elems[A102];
				a110 = elems[A112];
				A112 += A_cs;
				a111 = elems[A112];
				a120 = elems[A122];
				A122 += A_cs;
				a121 = elems[A122];

				a200 = elems[A202];
				A202 += A_cs;
				a201 = elems[A202];
				a210 = elems[A212];
				A212 += A_cs;
				a211 = elems[A212];
				a220 = elems[A222];
				A222 += A_cs;
				a221 = elems[A222];

				int B11 = B_index;
				for (int j = 1; j < c; j++) {
					// in each step 18 cells can be remembered in registers -
					// they don't need to be reread from slow memory
					// in each step 9 instead of 27 cells need to be read from
					// memory.
					a002 = elems[A002 += A_cs];
					a012 = elems[A012 += A_cs];
					a022 = elems[A022 += A_cs];

					a102 = elems[A102 += A_cs];
					a112 = elems[A112 += A_cs];
					a122 = elems[A122 += A_cs];

					a202 = elems[A202 += A_cs];
					a212 = elems[A212 += A_cs];
					a222 = elems[A222 += A_cs];

					B_elems[B11] = function.apply(a000, a001, a002, a010, a011, a012, a020, a021, a022,

					a100, a101, a102, a110, a111, a112, a120, a121, a122,

					a200, a201, a202, a210, a211, a212, a220, a221, a222);
					B11 += B_cs;

					// move remembered cells
					a000 = a001;
					a001 = a002;
					a010 = a011;
					a011 = a012;
					a020 = a021;
					a021 = a022;

					a100 = a101;
					a101 = a102;
					a110 = a111;
					a111 = a112;
					a120 = a121;
					a121 = a122;

					a200 = a201;
					a201 = a202;
					a210 = a211;
					a211 = a212;
					a220 = a221;
					a221 = a222;
				}
				A_index += A_rs;
				B_index += B_rs;
			}
		}
	}

	public float zSum() {
		float sum = 0;
		final int zero = index(0, 0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
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
				futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {

					public Float call() throws Exception {
						float sum = 0;
						int idx;
						for (int s = startslice; s < stopslice; s++) {
							for (int r = 0; r < rows; r++) {
								idx = zero + s * sliceStride + r * rowStride;
								for (int c = 0; c < columns; c++) {
									sum += elements[idx];
									idx += columnStride;
								}
							}
						}
						return Float.valueOf(sum);
					}
				});
			}
			try {
				for (int j = 0; j < np; j++) {
					sum = sum + ((Float) futures[j].get());
				}
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			int idx;
			for (int s = 0; s < slices; s++) {
				for (int r = 0; r < rows; r++) {
					idx = zero + s * sliceStride + r * rowStride;
					for (int c = 0; c < columns; c++) {
						sum += elements[idx];
						idx += columnStride;
					}
				}
			}
		}
		return sum;
	}

	protected boolean haveSharedCellsRaw(FloatMatrix3D other) {
		if (other instanceof SelectedDenseFloatMatrix3D) {
			SelectedDenseFloatMatrix3D otherMatrix = (SelectedDenseFloatMatrix3D) other;
			return this.elements == otherMatrix.elements;
		} else if (other instanceof DenseFloatMatrix3D) {
			DenseFloatMatrix3D otherMatrix = (DenseFloatMatrix3D) other;
			return this.elements == otherMatrix.elements;
		}
		return false;
	}

	protected FloatMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
		return new DenseFloatMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride);
	}

	protected FloatMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
		return new SelectedDenseFloatMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
	}
}
