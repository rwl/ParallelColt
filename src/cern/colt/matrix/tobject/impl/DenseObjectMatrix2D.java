/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import cern.colt.matrix.tobject.ObjectMatrix1D;
import cern.colt.matrix.tobject.ObjectMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>Object</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array, addressed in row
 * major. Note that this implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*rows()*columns()</tt>. Thus, a 1000*1000 matrix uses 8
 * MB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 * <p>
 * Cells are internally addressed in row-major. Applications demanding utmost
 * speed can exploit this fact. Setting/getting values in a loop row-by-row is
 * quicker than column-by-column. Thus
 * 
 * <pre>
 * for (int row = 0; row &lt; rows; row++) {
 * 	for (int column = 0; column &lt; columns; column++) {
 * 		matrix.setQuick(row, column, someValue);
 * 	}
 * }
 * </pre>
 * 
 * is quicker than
 * 
 * <pre>
 * for (int column = 0; column &lt; columns; column++) {
 * 	for (int row = 0; row &lt; rows; row++) {
 * 		matrix.setQuick(row, column, someValue);
 * 	}
 * }
 * </pre>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
public class DenseObjectMatrix2D extends ObjectMatrix2D {
	/**
	 * The elements of this matrix. elements are stored in row major, i.e.
	 * index==row*columns + column columnOf(index)==index%columns
	 * rowOf(index)==index/columns i.e. {row0 column0..m}, {row1 column0..m},
	 * ..., {rown column0..m}
	 */
	protected Object[] elements;

	/**
	 * Constructs a matrix with a copy of the given values. <tt>values</tt> is
	 * required to have the form <tt>values[row][column]</tt> and have exactly
	 * the same number of columns in every row.
	 * <p>
	 * The values are copied. So subsequent changes in <tt>values</tt> are not
	 * reflected in the matrix, and vice-versa.
	 * 
	 * @param values
	 *            The values to be filled into the new matrix.
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>for any 1 &lt;= row &lt; values.length: values[row].length != values[row-1].length</tt>
	 *             .
	 */
	public DenseObjectMatrix2D(Object[][] values) {
		this(values.length, values.length == 0 ? 0 : values[0].length);
		assign(values);
	}

	/**
	 * Constructs a matrix with a given number of rows and columns. All entries
	 * are initially <tt>0</tt>.
	 * 
	 * @param rows
	 *            the number of rows the matrix shall have.
	 * @param columns
	 *            the number of columns the matrix shall have.
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>rows<0 || columns<0 || (Object)columns*rows > Integer.MAX_VALUE</tt>
	 *             .
	 */
	public DenseObjectMatrix2D(int rows, int columns) {
		setUp(rows, columns);
		this.elements = new Object[rows * columns];
	}

	/**
	 * Constructs a view with the given parameters.
	 * 
	 * @param rows
	 *            the number of rows the matrix shall have.
	 * @param columns
	 *            the number of columns the matrix shall have.
	 * @param elements
	 *            the cells.
	 * @param rowZero
	 *            the position of the first element.
	 * @param columnZero
	 *            the position of the first element.
	 * @param rowStride
	 *            the number of elements between two rows, i.e.
	 *            <tt>index(i+1,j)-index(i,j)</tt>.
	 * @param columnStride
	 *            the number of elements between two columns, i.e.
	 *            <tt>index(i,j+1)-index(i,j)</tt>.
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>rows<0 || columns<0 || (Object)columns*rows > Integer.MAX_VALUE</tt>
	 *             or flip's are illegal.
	 */
	protected DenseObjectMatrix2D(int rows, int columns, Object[] elements, int rowZero, int columnZero, int rowStride, int columnStride) {
		setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
		this.elements = elements;
		this.isNoView = false;
	}

	/**
	 * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
	 * is required to have the form <tt>values[row][column]</tt> and have
	 * exactly the same number of rows and columns as the receiver.
	 * <p>
	 * The values are copied. So subsequent changes in <tt>values</tt> are not
	 * reflected in the matrix, and vice-versa.
	 * 
	 * @param values
	 *            the values to be filled into the cells.
	 * @return <tt>this</tt> (for convenience only).
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>values.length != rows() || for any 0 &lt;= row &lt; rows(): values[row].length != columns()</tt>
	 *             .
	 */
	public ObjectMatrix2D assign(final Object[][] values) {
		if (values.length != rows)
			throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "rows()=" + rows());
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if (this.isNoView) {
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
				Future[] futures = new Future[np];
				int k = rows / np;
				for (int j = 0; j < np; j++) {
					final int startrow = j * k;
					final int stoprow;
					if (j == np - 1) {
						stoprow = rows;
					} else {
						stoprow = startrow + k;
					}
					futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
						public void run() {
							int i = startrow * rowStride;
							for (int r = startrow; r < stoprow; r++) {
								Object[] currentRow = values[r];
								if (currentRow.length != columns)
									throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
								System.arraycopy(currentRow, 0, elements, i, columns);
								i += columns;
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
				for (int r = 0; r < rows; r++) {
					Object[] currentRow = values[r];
					if (currentRow.length != columns)
						throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
					System.arraycopy(currentRow, 0, this.elements, i, columns);
					i += columns;
				}
			}
		} else {
			final int zero = index(0, 0);
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
				Future[] futures = new Future[np];
				int k = rows / np;
				for (int j = 0; j < np; j++) {
					final int startrow = j * k;
					final int stoprow;
					if (j == np - 1) {
						stoprow = rows;
					} else {
						stoprow = startrow + k;
					}
					futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

						public void run() {
							int idx = zero + startrow * rowStride;
							for (int r = startrow; r < stoprow; r++) {
								Object[] currentRow = values[r];
								if (currentRow.length != columns)
									throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
								for (int i = idx, c = 0; c < columns; c++) {
									elements[i] = currentRow[c];
									i += columnStride;
								}
								idx += rowStride;
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
				for (int r = 0; r < rows; r++) {
					Object[] currentRow = values[r];
					if (currentRow.length != columns)
						throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
					for (int i = idx, c = 0; c < columns; c++) {
						elements[i] = currentRow[c];
						i += columnStride;
					}
					idx += rowStride;
				}
			}
			return this;
		}
		return this;
	}

	/**
	 * Assigns the result of a function to each cell;
	 * <tt>x[row,col] = function(x[row,col])</tt>.
	 * <p>
	 * <b>Example:</b>
	 * 
	 * <pre>
	 * 	 matrix = 2 x 2 matrix
	 * 	 0.5 1.5      
	 * 	 2.5 3.5
	 * 
	 * 	 // change each cell to its sine
	 * 	 matrix.assign(cern.jet.math.Functions.sin);
	 * 	 --&gt;
	 * 	 2 x 2 matrix
	 * 	 0.479426  0.997495 
	 * 	 0.598472 -0.350783
	 * 
	 * </pre>
	 * 
	 * For further examples, see the <a
	 * href="package-summary.html#FunctionObjects">package doc</a>.
	 * 
	 * @param function
	 *            a function object taking as argument the current cell's value.
	 * @return <tt>this</tt> (for convenience only).
	 * @see cern.jet.math.tdouble.DoubleFunctions
	 */
	public ObjectMatrix2D assign(final cern.colt.function.tobject.ObjectFunction function) {
		final Object[] elems = this.elements;
		if (elems == null)
			throw new InternalError();
		final int zero = index(0, 0);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
			Future[] futures = new Future[np];
			int k = rows / np;
			for (int j = 0; j < np; j++) {
				final int startrow = j * k;
				final int stoprow;
				if (j == np - 1) {
					stoprow = rows;
				} else {
					stoprow = startrow + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

					public void run() {
						int idx = zero + startrow * rowStride;
						// the general case x[i] = f(x[i])
						for (int r = startrow; r < stoprow; r++) {
							for (int i = idx, c = 0; c < columns; c++) {
								elems[i] = function.apply(elems[i]);
								i += columnStride;
							}
							idx += rowStride;
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
			for (int r = 0; r < rows; r++) {
				for (int i = idx, c = 0; c < columns; c++) {
					elems[i] = function.apply(elems[i]);
					i += columnStride;
				}
				idx += rowStride;
			}
		}
		return this;
	}

	/**
	 * Replaces all cell values of the receiver with the values of another
	 * matrix. Both matrices must have the same number of rows and columns. If
	 * both matrices share the same cells (as is the case if they are views
	 * derived from the same matrix) and intersect in an ambiguous way, then
	 * replaces <i>as if</i> using an intermediate auxiliary deep copy of
	 * <tt>other</tt>.
	 * 
	 * @param source
	 *            the source matrix to copy from (may be identical to the
	 *            receiver).
	 * @return <tt>this</tt> (for convenience only).
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>columns() != source.columns() || rows() != source.rows()</tt>
	 */
	public ObjectMatrix2D assign(ObjectMatrix2D source) {
		// overriden for performance only
		if (!(source instanceof DenseObjectMatrix2D)) {
			return super.assign(source);
		}
		final DenseObjectMatrix2D other_final = (DenseObjectMatrix2D) source;
		if (other_final == this)
			return this; // nothing to do
		checkShape(other_final);
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if (this.isNoView && other_final.isNoView) { // quickest
			if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
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
		}
		DenseObjectMatrix2D other = (DenseObjectMatrix2D) source;
		if (haveSharedCells(other)) {
			ObjectMatrix2D c = other.copy();
			if (!(c instanceof DenseObjectMatrix2D)) { // should not happen
				super.assign(other);
				return this;
			}
			other = (DenseObjectMatrix2D) c;
		}

		final Object[] elemsOther = other.elements;
		if (elements == null || elemsOther == null)
			throw new InternalError();
		final int zeroOther = other.index(0, 0);
		final int zero = index(0, 0);
		final int columnStrideOther = other.columnStride;
		final int rowStrideOther = other.rowStride;
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
			Future[] futures = new Future[np];
			int k = rows / np;
			for (int j = 0; j < np; j++) {
				final int startrow = j * k;
				final int stoprow;
				if (j == np - 1) {
					stoprow = rows;
				} else {
					stoprow = startrow + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {
					public void run() {
						int idx = zero + startrow * rowStride;
						int idxOther = zeroOther + startrow * rowStrideOther;
						for (int r = startrow; r < stoprow; r++) {
							for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
								elements[i] = elemsOther[j];
								i += columnStride;
								j += columnStrideOther;
							}
							idx += rowStride;
							idxOther += rowStrideOther;
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
			for (int r = 0; r < rows; r++) {
				for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
					elements[i] = elemsOther[j];
					i += columnStride;
					j += columnStrideOther;
				}
				idx += rowStride;
				idxOther += rowStrideOther;
			}
		}
		return this;
	}

	/**
	 * Assigns the result of a function to each cell;
	 * <tt>x[row,col] = function(x[row,col],y[row,col])</tt>.
	 * <p>
	 * <b>Example:</b>
	 * 
	 * <pre>
	 * 	 // assign x[row,col] = x[row,col]&lt;sup&gt;y[row,col]&lt;/sup&gt;
	 * 	 m1 = 2 x 2 matrix 
	 * 	 0 1 
	 * 	 2 3
	 * 
	 * 	 m2 = 2 x 2 matrix 
	 * 	 0 2 
	 * 	 4 6
	 * 
	 * 	 m1.assign(m2, cern.jet.math.Functions.pow);
	 * 	 --&gt;
	 * 	 m1 == 2 x 2 matrix
	 * 	 1   1 
	 * 	 16 729
	 * 
	 * </pre>
	 * 
	 * For further examples, see the <a
	 * href="package-summary.html#FunctionObjects">package doc</a>.
	 * 
	 * @param y
	 *            the secondary matrix to operate on.
	 * @param function
	 *            a function object taking as first argument the current cell's
	 *            value of <tt>this</tt>, and as second argument the current
	 *            cell's value of <tt>y</tt>,
	 * @return <tt>this</tt> (for convenience only).
	 * @throws IllegalArgumentException
	 *             if
	 *             <tt>columns() != other.columns() || rows() != other.rows()</tt>
	 * @see cern.jet.math.tdouble.DoubleFunctions
	 */
	public ObjectMatrix2D assign(final ObjectMatrix2D y, final cern.colt.function.tobject.ObjectObjectFunction function) {
		// overriden for performance only
		if (!(y instanceof DenseObjectMatrix2D)) {
			return super.assign(y, function);
		}
		DenseObjectMatrix2D other = (DenseObjectMatrix2D) y;
		checkShape(y);
		final Object[] elemsOther = other.elements();
		if (elements == null || elemsOther == null)
			throw new InternalError();
		final int zeroOther = other.index(0, 0);
		final int zero = index(0, 0);
		final int columnStrideOther = other.columnStride;
		final int rowStrideOther = other.rowStride;
		int np = ConcurrencyUtils.getNumberOfProcessors();
		if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
			Future[] futures = new Future[np];
			int k = rows / np;
			for (int j = 0; j < np; j++) {
				final int startrow = j * k;
				final int stoprow;
				if (j == np - 1) {
					stoprow = rows;
				} else {
					stoprow = startrow + k;
				}
				futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

					public void run() {
						int idx;
						int idxOther;
						idx = zero + startrow * rowStride;
						idxOther = zeroOther + startrow * rowStrideOther;
						for (int r = startrow; r < stoprow; r++) {
							for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
								elements[i] = function.apply(elements[i], elemsOther[j]);
								i += columnStride;
								j += columnStrideOther;
							}
							idx += rowStride;
							idxOther += rowStrideOther;
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
			idx = zero;
			idxOther = zeroOther;
			for (int r = 0; r < rows; r++) {
				for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
					elements[i] = function.apply(elements[i], elemsOther[j]);
					i += columnStride;
					j += columnStrideOther;
				}
				idx += rowStride;
				idxOther += rowStrideOther;
			}
		}
		return this;
	}

	public Object[] elements() {
		return elements;
	}

	/**
	 * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
	 * 
	 * <p>
	 * Provided with invalid parameters this method may return invalid objects
	 * without throwing any exception. <b>You should only use this method when
	 * you are absolutely sure that the coordinate is within bounds.</b>
	 * Precondition (unchecked):
	 * <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
	 * 
	 * @param row
	 *            the index of the row-coordinate.
	 * @param column
	 *            the index of the column-coordinate.
	 * @return the value at the specified coordinate.
	 */
	public Object getQuick(int row, int column) {
		// if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
		// throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
		// return elements[index(row,column)];
		// manually inlined:
		return elements[rowZero + row * rowStride + columnZero + column * columnStride];
	}

	/**
	 * Returns <tt>true</tt> if both matrices share common cells. More formally,
	 * returns <tt>true</tt> if <tt>other != null</tt> and at least one of the
	 * following conditions is met
	 * <ul>
	 * <li>the receiver is a view of the other matrix
	 * <li>the other matrix is a view of the receiver
	 * <li><tt>this == other</tt>
	 * </ul>
	 */
	protected boolean haveSharedCellsRaw(ObjectMatrix2D other) {
		if (other instanceof SelectedDenseObjectMatrix2D) {
			SelectedDenseObjectMatrix2D otherMatrix = (SelectedDenseObjectMatrix2D) other;
			return this.elements == otherMatrix.elements;
		} else if (other instanceof DenseObjectMatrix2D) {
			DenseObjectMatrix2D otherMatrix = (DenseObjectMatrix2D) other;
			return this.elements == otherMatrix.elements;
		}
		return false;
	}

	/**
	 * Returns the position of the given coordinate within the (virtual or
	 * non-virtual) internal 1-dimensional array.
	 * 
	 * @param row
	 *            the index of the row-coordinate.
	 * @param column
	 *            the index of the column-coordinate.
	 */
	public int index(int row, int column) {
		// return super.index(row,column);
		// manually inlined for speed:
		return rowZero + row * rowStride + columnZero + column * columnStride;
	}

	/**
	 * Construct and returns a new empty matrix <i>of the same dynamic type</i>
	 * as the receiver, having the specified number of rows and columns. For
	 * example, if the receiver is an instance of type
	 * <tt>DenseObjectMatrix2D</tt> the new matrix must also be of type
	 * <tt>DenseObjectMatrix2D</tt>, if the receiver is an instance of type
	 * <tt>SparseObjectMatrix2D</tt> the new matrix must also be of type
	 * <tt>SparseObjectMatrix2D</tt>, etc. In general, the new matrix should
	 * have internal parametrization as similar as possible.
	 * 
	 * @param rows
	 *            the number of rows the matrix shall have.
	 * @param columns
	 *            the number of columns the matrix shall have.
	 * @return a new empty matrix of the same dynamic type.
	 */
	public ObjectMatrix2D like(int rows, int columns) {
		return new DenseObjectMatrix2D(rows, columns);
	}

	/**
	 * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
	 * type</i>, entirelly independent of the receiver. For example, if the
	 * receiver is an instance of type <tt>DenseObjectMatrix2D</tt> the new
	 * matrix must be of type <tt>DenseObjectMatrix1D</tt>, if the receiver is
	 * an instance of type <tt>SparseObjectMatrix2D</tt> the new matrix must be
	 * of type <tt>SparseObjectMatrix1D</tt>, etc.
	 * 
	 * @param size
	 *            the number of cells the matrix shall have.
	 * @return a new matrix of the corresponding dynamic type.
	 */
	public ObjectMatrix1D like1D(int size) {
		return new DenseObjectMatrix1D(size);
	}

	/**
	 * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
	 * type</i>, sharing the same cells. For example, if the receiver is an
	 * instance of type <tt>DenseObjectMatrix2D</tt> the new matrix must be of
	 * type <tt>DenseObjectMatrix1D</tt>, if the receiver is an instance of type
	 * <tt>SparseObjectMatrix2D</tt> the new matrix must be of type
	 * <tt>SparseObjectMatrix1D</tt>, etc.
	 * 
	 * @param size
	 *            the number of cells the matrix shall have.
	 * @param zero
	 *            the index of the first element.
	 * @param stride
	 *            the number of indexes between any two elements, i.e.
	 *            <tt>index(i+1)-index(i)</tt>.
	 * @return a new matrix of the corresponding dynamic type.
	 */
	protected ObjectMatrix1D like1D(int size, int zero, int stride) {
		return new DenseObjectMatrix1D(size, this.elements, zero, stride);
	}

	/**
	 * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified
	 * value.
	 * 
	 * <p>
	 * Provided with invalid parameters this method may access illegal indexes
	 * without throwing any exception. <b>You should only use this method when
	 * you are absolutely sure that the coordinate is within bounds.</b>
	 * Precondition (unchecked):
	 * <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
	 * 
	 * @param row
	 *            the index of the row-coordinate.
	 * @param column
	 *            the index of the column-coordinate.
	 * @param value
	 *            the value to be filled into the specified cell.
	 */
	public void setQuick(int row, int column, Object value) {
		// if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
		// throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
		// elements[index(row,column)] = value;
		// manually inlined:
		elements[rowZero + row * rowStride + columnZero + column * columnStride] = value;
	}

	/**
	 * Construct and returns a new selection view.
	 * 
	 * @param rowOffsets
	 *            the offsets of the visible elements.
	 * @param columnOffsets
	 *            the offsets of the visible elements.
	 * @return a new view.
	 */
	protected ObjectMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
		return new SelectedDenseObjectMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
	}
}
