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
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import edu.emory.mathcs.jtransforms.dct.FloatDCT_2D;
import edu.emory.mathcs.jtransforms.dht.FloatDHT_2D;
import edu.emory.mathcs.jtransforms.dst.FloatDST_2D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>float</tt> elements. First see the <a
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
 *     for (int column = 0; column &lt; columns; column++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * is quicker than
 * 
 * <pre>
 * for (int column = 0; column &lt; columns; column++) {
 *     for (int row = 0; row &lt; rows; row++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseFloatMatrix2D extends FloatMatrix2D {
    static final long serialVersionUID = 1020177651L;

    private FloatFFT_2D fft2;

    private FloatDCT_2D dct2;

    private FloatDST_2D dst2;

    private FloatDHT_2D dht2;
    
    /**
     * The elements of this matrix. elements are stored in row major, i.e.
     * index==row*columns + column columnOf(index)==index%columns
     * rowOf(index)==index/columns i.e. {row0 column0..m}, {row1 column0..m},
     * ..., {rown column0..m}
     */
    protected float[] elements;

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
    public DenseFloatMatrix2D(float[][] values) {
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public DenseFloatMatrix2D(int rows, int columns) {
        setUp(rows, columns);
        this.elements = new float[rows * columns];
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    public DenseFloatMatrix2D(int rows, int columns, float[] elements, int rowZero, int columnZero, int rowStride, int columnStride) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = false;
    }

    public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFunction f) {
        if (size() == 0)
            return Float.NaN;
        final int zero = index(0, 0);
        float a = 0;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            Float[] results = new Float[np];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rows;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + startrow * rowStride]);
                        int d = 1;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = d; c < columns; c++) {
                                a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride]));
                            }
                            d = 0;
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
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride]));
                }
                d = 0;
            }
        }
        return a;
    }

    public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFunction f, final cern.colt.function.tfloat.FloatProcedure cond) {
        if (size() == 0)
            return Float.NaN;
        final int zero = index(0, 0);
        float a = 0;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            Float[] results = new Float[np];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rows;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float elem = elements[zero + startrow * rowStride];
                        float a = 0;
                        if (cond.apply(elem) == true) {
                            a = f.apply(elem);
                        }
                        int d = 1;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = d; c < columns; c++) {
                                elem = elements[zero + r * rowStride + c * columnStride];
                                if (cond.apply(elem) == true) {
                                    a = aggr.apply(a, f.apply(elem));
                                }
                            }
                            d = 0;
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
                a = f.apply(elements[zero]);
            }
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    elem = elements[zero + r * rowStride + c * columnStride];
                    if (cond.apply(elem) == true) {
                        a = aggr.apply(a, f.apply(elem));
                    }
                }
                d = 0;
            }
        }
        return a;
    }

    public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFunction f, final IntArrayList rowList, final IntArrayList columnList) {
        if (size() == 0)
            return Float.NaN;
        final int zero = index(0, 0);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        float a = 0;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
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
                        float a = f.apply(elements[zero + rowElements[startidx] * rowStride + columnElements[startidx] * columnStride]);
                        float elem;
                        for (int i = startidx + 1; i < stopidx; i++) {
                            elem = elements[zero + rowElements[i] * rowStride + columnElements[i] * columnStride];
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
            float elem;
            a = f.apply(elements[zero + rowElements[0] * rowStride + columnElements[0] * columnStride]);
            for (int i = 1; i < size; i++) {
                elem = elements[zero + rowElements[i] * rowStride + columnElements[i] * columnStride];
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    public float aggregate(final FloatMatrix2D other, final cern.colt.function.tfloat.FloatFloatFunction aggr, final cern.colt.function.tfloat.FloatFloatFunction f) {
        if (!(other instanceof DenseFloatMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            return Float.NaN;
        final int zero = index(0, 0);
        final int zeroOther = other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final float[] elemsOther = (float[]) other.elements();
        float a = 0;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            Float[] results = new Float[np];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rows;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + startrow * rowStride], elemsOther[zeroOther + startrow * rowStrideOther]);
                        int d = 1;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = d; c < columns; c++) {
                                a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride], elemsOther[zeroOther + r * rowStrideOther + c * colStrideOther]));
                            }
                            d = 0;
                        }
                        return Float.valueOf(a);
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
            int d = 1; // first cell already done
            a = f.apply(elements[zero], elemsOther[zeroOther]);
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride], elemsOther[zeroOther + r * rowStrideOther + c * colStrideOther]));
                }
                d = 0;
            }
        }
        return a;
    }

    public FloatMatrix2D assign(final cern.colt.function.tfloat.FloatFunction function) {
        final float[] elems = this.elements;
        if (elems == null)
            throw new InternalError();
        final int zero = index(0, 0);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] =
                // mult*x[i]
                float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                if (multiplicator == 1)
                    return this;
                if (multiplicator == 0)
                    return assign(0);
            }
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
                        // specialization for speed
                        if (function instanceof cern.jet.math.tfloat.FloatMult) {
                            // x[i] = mult*x[i]
                            float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                            if (multiplicator == 1)
                                return;
                            for (int r = startrow; r < stoprow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elems[i] *= multiplicator;
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        } else {
                            // the general case x[i] = f(x[i])
                            for (int r = startrow; r < stoprow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elems[i] = function.apply(elems[i]);
                                    i += columnStride;
                                }
                                idx += rowStride;
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
            int idx = zero;
            // specialization for speed
            if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] =
                // mult*x[i]
                float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                if (multiplicator == 1)
                    return this;
                if (multiplicator == 0)
                    return assign(0);
                for (int r = 0; r < rows; r++) { // the general case
                    for (int i = idx, c = 0; c < columns; c++) {
                        elems[i] *= multiplicator;
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            } else { // the general case x[i] = f(x[i])
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        elems[i] = function.apply(elems[i]);
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final cern.colt.function.tfloat.FloatProcedure cond, final cern.colt.function.tfloat.FloatFunction function) {
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
                        float elem;
                        int idx = zero + startrow * rowStride;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem = elements[i];
                                if (cond.apply(elem) == true) {
                                    elements[i] = function.apply(elem);
                                }
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
            float elem;
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem = elements[i];
                    if (cond.apply(elem) == true) {
                        elements[i] = function.apply(elem);
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final cern.colt.function.tfloat.FloatProcedure cond, final float value) {
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
                        float elem;
                        int idx = zero + startrow * rowStride;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem = elements[i];
                                if (cond.apply(elem) == true) {
                                    elements[i] = value;
                                }
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
            float elem;
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem = elements[i];
                    if (cond.apply(elem) == true) {
                        elements[i] = value;
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final float value) {
        final float[] elems = this.elements;
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
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elems[i] = value;
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
                    elems[i] = value;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final float[] values) {
        if (values.length != size())
            throw new IllegalArgumentException("Must have same length: length=" + values.length + " rows()*columns()=" + rows() * columns());
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if (this.isNoView) {
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
            final int zero = index(0, 0);
            if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future[] futures = new Future[np];
                int k = rows / np;
                for (int j = 0; j < np; j++) {
                    final int startrow = j * k;
                    final int stoprow;
                    final int glob_idxOther = j * k * columns;
                    if (j == np - 1) {
                        stoprow = rows;
                    } else {
                        stoprow = startrow + k;
                    }
                    futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                        public void run() {
                            int idxOther = glob_idxOther;
                            int idx = zero + startrow * rowStride;
                            for (int r = startrow; r < stoprow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = values[idxOther++];
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

                int idxOther = 0;
                int idx = zero;
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, c = 0; c < columns; c++) {
                        elements[i] = values[idxOther++];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final float[][] values) {
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
                                float[] currentRow = values[r];
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
                    float[] currentRow = values[r];
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
                                float[] currentRow = values[r];
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
                    float[] currentRow = values[r];
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

    public FloatMatrix2D assign(final FloatMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof DenseFloatMatrix2D)) {
            super.assign(source);
            return this;
        }
        final DenseFloatMatrix2D other_final = (DenseFloatMatrix2D) source;
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
        DenseFloatMatrix2D other = (DenseFloatMatrix2D) source;
        if (haveSharedCells(other)) {
            FloatMatrix2D c = other.copy();
            if (!(c instanceof DenseFloatMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseFloatMatrix2D) c;
        }

        final float[] elemsOther = other.elements;
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

    public FloatMatrix2D assign(final FloatMatrix2D y, final cern.colt.function.tfloat.FloatFloatFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseFloatMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        DenseFloatMatrix2D other = (DenseFloatMatrix2D) y;
        checkShape(y);
        final float[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = other.index(0, 0);
        final int zero = index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tfloat.FloatPlusMult) {
                float multiplicator = ((cern.jet.math.tfloat.FloatPlusMult) function).multiplicator;
                if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                    return this;
                }
            }
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
                        // specialized for speed
                        if (function == cern.jet.math.tfloat.FloatFunctions.mult) {
                            // x[i] = x[i]*y[i]
                            idx = zero + startrow * rowStride;
                            idxOther = zeroOther + startrow * rowStrideOther;
                            for (int r = startrow; r < stoprow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] *= elemsOther[j];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else if (function == cern.jet.math.tfloat.FloatFunctions.div) {
                            // x[i] = x[i] / y[i]
                            idx = zero + startrow * rowStride;
                            idxOther = zeroOther + startrow * rowStrideOther;
                            for (int r = startrow; r < stoprow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] /= elemsOther[j];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else if (function instanceof cern.jet.math.tfloat.FloatPlusMult) {
                            float multiplicator = ((cern.jet.math.tfloat.FloatPlusMult) function).multiplicator;
                            if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                idx = zero + startrow * rowStride;
                                idxOther = zeroOther + startrow * rowStrideOther;
                                for (int r = startrow; r < stoprow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] += elemsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = x[i] - y[i]
                                idx = zero + startrow * rowStride;
                                idxOther = zeroOther + startrow * rowStrideOther;
                                for (int r = startrow; r < stoprow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] -= elemsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            } else { // the general case
                                // x[i] = x[i] + mult*y[i]
                                idx = zero + startrow * rowStride;
                                idxOther = zeroOther + startrow * rowStrideOther;
                                for (int r = startrow; r < stoprow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] += multiplicator * elemsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            }
                        } else { // the general case x[i] = f(x[i],y[i])
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
            // specialized for speed
            if (function == cern.jet.math.tfloat.FloatFunctions.mult) {
                // x[i] = x[i] * y[i]
                idx = zero;
                idxOther = zeroOther;
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        elements[i] *= elemsOther[j];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            } else if (function == cern.jet.math.tfloat.FloatFunctions.div) {
                // x[i] = x[i] / y[i]
                idx = zero;
                idxOther = zeroOther;
                for (int r = 0; r < rows; r++) {
                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                        elements[i] /= elemsOther[j];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            } else if (function instanceof cern.jet.math.tfloat.FloatPlusMult) {
                float multiplicator = ((cern.jet.math.tfloat.FloatPlusMult) function).multiplicator;
                if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                    return this;
                } else if (multiplicator == 1) { // x[i] = x[i] + y[i]
                    idx = zero;
                    idxOther = zeroOther;
                    for (int r = 0; r < rows; r++) {
                        for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                            elements[i] += elemsOther[j];
                            i += columnStride;
                            j += columnStrideOther;
                        }
                        idx += rowStride;
                        idxOther += rowStrideOther;
                    }

                } else if (multiplicator == -1) { // x[i] = x[i] - y[i]
                    idx = zero;
                    idxOther = zeroOther;
                    for (int r = 0; r < rows; r++) {
                        for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                            elements[i] -= elemsOther[j];
                            i += columnStride;
                            j += columnStrideOther;
                        }
                        idx += rowStride;
                        idxOther += rowStrideOther;
                    }
                } else { // the general case
                    // x[i] = x[i] + mult*y[i]
                    idx = zero;
                    idxOther = zeroOther;
                    for (int r = 0; r < rows; r++) {
                        for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                            elements[i] += multiplicator * elemsOther[j];
                            i += columnStride;
                            j += columnStrideOther;
                        }
                        idx += rowStride;
                        idxOther += rowStrideOther;
                    }
                }
            } else { // the general case x[i] = f(x[i],y[i])
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
        }
        return this;
    }

    public FloatMatrix2D assign(final FloatMatrix2D y, final cern.colt.function.tfloat.FloatFloatFunction function, IntArrayList rowList, IntArrayList columnList) {
        checkShape(y);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final float[] elemsOther = (float[]) y.elements();
        final int zeroOther = y.index(0, 0);
        final int zero = index(0, 0);
        final int columnStrideOther = y.columnStride();
        final int rowStrideOther = y.rowStride();
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
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
                        for (int i = startidx; i < stopidx; i++) {
                            idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                            idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i] * columnStrideOther;
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
            int idx;
            int idxOther;
            for (int i = 0; i < size; i++) {
                idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i] * columnStrideOther;
                elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
            }
        }
        return this;
    }

    public int cardinality() {
        int cardinality = 0;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        final int zero = index(0, 0);
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            Integer[] results = new Integer[np];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rows;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        int idx = zero + startrow * rowStride;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                if (elements[i] != 0)
                                    cardinality++;
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        return Integer.valueOf(cardinality);
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (Integer) futures[j].get();
                }
                cardinality = results[0].intValue();
                for (int j = 1; j < np; j++) {
                    cardinality += results[j].intValue();
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
                    if (elements[i] != 0)
                        cardinality++;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return cardinality;
    }

    public void dct2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        if (dct2 == null) {
            dct2 = new FloatDCT_2D(rows, columns);
        }
        if (isNoView == true) {
            dct2.forward(elements, scale);
        } else {
            FloatMatrix2D copy = this.copy();
            dct2.forward((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void dctColumns(final boolean scale) {
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int c = startcol; c < stopcol; c++) {
                            viewColumn(c).dct(scale);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                viewColumn(c).dct(scale);
            }
        }
    }

    public void dctRows(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        for (int r = startrow; r < stoprow; r++) {
                            viewRow(r).dct(scale);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                viewRow(r).dct(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }
    
    public void dht2() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        if (dht2 == null) {
            dht2 = new FloatDHT_2D(rows, columns);
        }
        if (isNoView == true) {
            dht2.forward(elements);
        } else {
            FloatMatrix2D copy = this.copy();
            dht2.forward((float[]) copy.elements());
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void dhtColumns() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int c = startcol; c < stopcol; c++) {
                            viewColumn(c).dht();
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                viewColumn(c).dht();
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void dhtRows() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        for (int r = startrow; r < stoprow; r++) {
                            viewRow(r).dht();
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                viewRow(r).dht();
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void dst2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	if (dst2 == null) {
            dst2 = new FloatDST_2D(rows, columns);
        }
        if (isNoView == true) {
            dst2.forward(elements, scale);
        } else {
            FloatMatrix2D copy = this.copy();
            dst2.forward((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void dstColumns(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        FloatMatrix1D column;
                        for (int c = startcol; c < stopcol; c++) {
                            column = viewColumn(c).copy();
                            column.dst(scale);
                            viewColumn(c).assign(column);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                viewColumn(c).dst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void dstRows(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        FloatMatrix1D row;
                        for (int r = startrow; r < stoprow; r++) {
                            row = viewRow(r).copy();
                            row.dst(scale);
                            viewRow(r).assign(row);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                viewRow(r).dst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public float[] elements() {
        return elements;
    }

    public void fft2() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        if (isNoView == true) {
            fft2.realForward(elements);
        } else {
            FloatMatrix2D copy = this.copy();
            fft2.realForward((float[]) copy.elements());
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public FloatMatrix2D forEachNonZero(final cern.colt.function.tfloat.IntIntFloatFunction function) {
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
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                float value = elements[i];
                                if (value != 0) {
                                    elements[i] = function.apply(r, c, value);
                                }
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
                    float value = elements[i];
                    if (value != 0) {
                        elements[i] = function.apply(r, c, value);
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D getFft2() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        final float[] elems;
        if (isNoView == true) {
            elems = elements;
        } else {
            elems = (float[]) this.copy().elements();
        }
        FComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        final float[] cElems = (float[]) ((DenseFComplexMatrix2D) C).elements();
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
                        for (int r = startrow; r < stoprow; r++) {
                            System.arraycopy(elems, r * columns, cElems, r * columns, columns);
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
            for (int r = 0; r < rows; r++) {
                System.arraycopy(elems, r * columns, cElems, r * columns, columns);
            }
        }
        fft2.realForwardFull(cElems);
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
        return C;
    }

    public FComplexMatrix2D getFftColumns() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	final FComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int c = startcol; c < stopcol; c++) {
                        	C.viewColumn(c).assign(viewColumn(c).getFft());
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                C.viewColumn(c).assign(viewColumn(c).getFft());
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
        return C;
    }

    public FComplexMatrix2D getFftRows() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        final FComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        for (int r = startrow; r < stoprow; r++) {
                        	 C.viewRow(r).assign(viewRow(r).getFft());
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                C.viewRow(r).assign(viewRow(r).getFft());
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
        return C;        
    }

    public FComplexMatrix2D getIfft2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	FComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        final float[] cElems = (float[]) ((DenseFComplexMatrix2D) C).elements();
        final float[] elems;
        if (isNoView == true) {
            elems = elements;
        } else {
            elems = (float[]) this.copy().elements();
        }
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
                        for (int r = startrow; r < stoprow; r++) {
                            System.arraycopy(elems, r * columns, cElems, r * columns, columns);
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
            for (int r = 0; r < rows; r++) {
                System.arraycopy(elems, r * columns, cElems, r * columns, columns);
            }
        }
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.realInverseFull(cElems, scale);
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
        return C;
    }

    public FComplexMatrix2D getIfftColumns(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	final FComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int c = startcol; c < stopcol; c++) {
                            C.viewColumn(c).assign(viewColumn(c).getIfft(scale));
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                C.viewColumn(c).assign(viewColumn(c).getIfft(scale));
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
        return C;
    }

    public FComplexMatrix2D getIfftRows(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        final FComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        for (int r = startrow; r < stoprow; r++) {
                            C.viewRow(r).assign(viewRow(r).getIfft(scale));
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                C.viewRow(r).assign(viewRow(r).getIfft(scale));
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
        return C;
    }

    public void getNegativeValues(final IntArrayList rowList, final IntArrayList columnList, final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                float value = elements[i];
                if (value < 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += columnStride;
            }
            idx += rowStride;
        }
    }

    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList, final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                float value = elements[i];
                if (value != 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += columnStride;
            }
            idx += rowStride;
        }
    }

    public void getPositiveValues(final IntArrayList rowList, final IntArrayList columnList, final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                float value = elements[i];
                if (value > 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += columnStride;
            }
            idx += rowStride;
        }
    }

    public float getQuick(int row, int column) {
        return elements[rowZero + row * rowStride + columnZero + column * columnStride];
    }

    public void idct2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        if (dct2 == null) {
            dct2 = new FloatDCT_2D(rows, columns);
        }
        if (isNoView == true) {
            dct2.inverse(elements, scale);
        } else {
            FloatMatrix2D copy = this.copy();
            dct2.inverse((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void idctColumns(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int c = startcol; c < stopcol; c++) {
                            viewColumn(c).idct(scale);;
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                viewColumn(c).idct(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void idctRows(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        for (int r = startrow; r < stoprow; r++) {
                            viewRow(r).idct(scale);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                viewRow(r).idct(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }
    
    public void idht2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	if (dht2 == null) {
            dht2 = new FloatDHT_2D(rows, columns);
        }
        if (isNoView == true) {
            dht2.inverse(elements, scale);
        } else {
            FloatMatrix2D copy = this.copy();
            dht2.inverse((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void idhtColumns(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int c = startcol; c < stopcol; c++) {
                            viewColumn(c).idht(scale);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                viewColumn(c).idht(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void idhtRows(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        for (int r = startrow; r < stoprow; r++) {
                            viewRow(r).idht(scale);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                viewRow(r).idht(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void idst2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	if (dst2 == null) {
            dst2 = new FloatDST_2D(rows, columns);
        }
        if (isNoView == true) {
            dst2.inverse(elements, scale);
        } else {
            FloatMatrix2D copy = this.copy();
            dst2.inverse((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void idstColumns(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        for (int c = startcol; c < stopcol; c++) {
                            viewColumn(c).idst(scale);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                viewColumn(c).idst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void idstRows(final boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
    	int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
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
                        for (int r = startrow; r < stoprow; r++) {
                            viewRow(r).idst(scale);
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
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                viewRow(r).idst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void ifft2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.prevPow2(ConcurrencyUtils.getNumberOfProcessors()));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        if (isNoView == true) {
            fft2.realInverse(elements, scale);
        } else {
            FloatMatrix2D copy = this.copy();
            fft2.realInverse((float[]) copy.elements(), scale);
            this.assign((float[]) copy.elements());
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public int index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    public FloatMatrix2D like(int rows, int columns) {
        return new DenseFloatMatrix2D(rows, columns);
    }

    public FloatMatrix1D like1D(int size) {
        return new DenseFloatMatrix1D(size);
    }

    public float[] getMaxLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = index(0, 0);
        float maxValue = 0;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            float[][] results = new float[np][2];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rows;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float maxValue = elements[zero + startrow * rowStride];
                        int rowLocation = startrow;
                        int colLocation = 0;
                        float elem;
                        int d = 1;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = d; c < columns; c++) {
                                elem = elements[zero + r * rowStride + c * columnStride];
                                if (maxValue < elem) {
                                    maxValue = elem;
                                    rowLocation = r;
                                    colLocation = c;
                                }
                            }
                            d = 0;
                        }
                        return new float[] { maxValue, rowLocation, colLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (float[]) futures[j].get();
                }
                maxValue = results[0][0];
                rowLocation = (int) results[0][1];
                columnLocation = (int) results[0][2];
                for (int j = 1; j < np; j++) {
                    if (maxValue < results[j][0]) {
                        maxValue = results[j][0];
                        rowLocation = (int) results[j][1];
                        columnLocation = (int) results[j][2];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            maxValue = elements[zero];
            int d = 1;
            float elem;
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    elem = elements[zero + r * rowStride + c * columnStride];
                    if (maxValue < elem) {
                        maxValue = elem;
                        rowLocation = r;
                        columnLocation = c;
                    }
                }
                d = 0;
            }
        }
        return new float[] { maxValue, rowLocation, columnLocation };
    }

    public float[] getMinLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = index(0, 0);
        float minValue = 0;
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            float[][] results = new float[np][2];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rows;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        int rowLocation = startrow;
                        int columnLocation = 0;
                        float minValue = elements[zero + startrow * rowStride];
                        float elem;
                        int d = 1;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = d; c < columns; c++) {
                                elem = elements[zero + r * rowStride + c * columnStride];
                                if (minValue > elem) {
                                    minValue = elem;
                                    rowLocation = r;
                                    columnLocation = c;
                                }
                            }
                            d = 0;
                        }
                        return new float[] { minValue, rowLocation, columnLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (float[]) futures[j].get();
                }
                minValue = results[0][0];
                rowLocation = (int) results[0][1];
                columnLocation = (int) results[0][2];
                for (int j = 1; j < np; j++) {
                    if (minValue > results[j][0]) {
                        minValue = results[j][0];
                        rowLocation = (int) results[j][1];
                        columnLocation = (int) results[j][2];
                    }
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            minValue = elements[zero];
            int d = 1;
            float elem;
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    elem = elements[zero + r * rowStride + c * columnStride];
                    if (minValue > elem) {
                        minValue = elem;
                        rowLocation = r;
                        columnLocation = c;
                    }
                }
                d = 0;
            }
        }
        return new float[] { minValue, rowLocation, columnLocation };
    }

    public void setQuick(int row, int column, float value) {
        elements[rowZero + row * rowStride + columnZero + column * columnStride] = value;
    }

    public float[][] toArray() {
        final float[][] values = new float[rows][columns];
        int np = ConcurrencyUtils.getNumberOfProcessors();
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
                            float[] currentRow = values[r];
                            for (int i = idx, c = 0; c < columns; c++) {
                                currentRow[c] = elements[i];
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
                float[] currentRow = values[r];
                for (int i = idx, c = 0; c < columns; c++) {
                    currentRow[c] = elements[i];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return values;
    }

    public FloatMatrix1D vectorize() {
        final DenseFloatMatrix1D v = new DenseFloatMatrix1D(size());
        final int zero = index(0, 0);
        final int zeroOther = v.index(0);
        final int strideOther = v.stride();
        final float[] elemsOther = (float[]) v.elements();
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            int k = columns / np;
            for (int j = 0; j < np; j++) {
                final int startcol = j * k;
                final int stopcol;
                final int startidx = j * k * rows;
                if (j == np - 1) {
                    stopcol = columns;
                } else {
                    stopcol = startcol + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        int idx = 0;
                        int idxOther = zeroOther + startidx * strideOther;
                        for (int c = startcol; c < stopcol; c++) {
                            idx = zero + c * columnStride;
                            for (int r = 0; r < rows; r++) {
                                elemsOther[idxOther] = elements[idx];
                                idx += rowStride;
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
            int idx = zero;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                idx = zero + c * columnStride;
                for (int r = 0; r < rows; r++) {
                    elemsOther[idxOther] = elements[idx];
                    idx += rowStride;
                    idxOther += strideOther;
                }
            }
        }
        return v;
    }

    public void zAssign8Neighbors(FloatMatrix2D B, cern.colt.function.tfloat.Float9Function function) {
        // 1. using only 4-5 out of the 9 cells in "function" is *not* the
        // limiting factor for performance.

        // 2. if the "function" would be hardwired into the innermost loop, a
        // speedup of 1.5-2.0 would be seen
        // but then the multi-purpose interface is gone...

        if (!(B instanceof DenseFloatMatrix2D)) {
            super.zAssign8Neighbors(B, function);
            return;
        }
        if (function == null)
            throw new NullPointerException("function must not be null.");
        checkShape(B);
        int r = rows - 1;
        int c = columns - 1;
        if (rows < 3 || columns < 3)
            return; // nothing to do

        DenseFloatMatrix2D BB = (DenseFloatMatrix2D) B;
        int A_rs = rowStride;
        int B_rs = BB.rowStride;
        int A_cs = columnStride;
        int B_cs = BB.columnStride;
        float[] elems = this.elements;
        float[] B_elems = BB.elements;
        if (elems == null || B_elems == null)
            throw new InternalError();

        int A_index = index(1, 1);
        int B_index = BB.index(1, 1);
        for (int i = 1; i < r; i++) {
            float a00, a01, a02;
            float a10, a11, a12;
            float a20, a21, a22;

            int B11 = B_index;

            int A02 = A_index - A_rs - A_cs;
            int A12 = A02 + A_rs;
            int A22 = A12 + A_rs;

            // in each step six cells can be remembered in registers - they
            // don't need to be reread from slow memory
            a00 = elems[A02];
            A02 += A_cs;
            a01 = elems[A02]; // A02+=A_cs;
            a10 = elems[A12];
            A12 += A_cs;
            a11 = elems[A12]; // A12+=A_cs;
            a20 = elems[A22];
            A22 += A_cs;
            a21 = elems[A22]; // A22+=A_cs;

            for (int j = 1; j < c; j++) {
                // in each step 3 instead of 9 cells need to be read from
                // memory.
                a02 = elems[A02 += A_cs];
                a12 = elems[A12 += A_cs];
                a22 = elems[A22 += A_cs];

                B_elems[B11] = function.apply(a00, a01, a02, a10, a11, a12, a20, a21, a22);
                B11 += B_cs;

                // move remembered cells
                a00 = a01;
                a01 = a02;
                a10 = a11;
                a11 = a12;
                a20 = a21;
                a21 = a22;
            }
            A_index += A_rs;
            B_index += B_rs;
        }

    }

    public FloatMatrix1D zMult(final FloatMatrix1D y, FloatMatrix1D z, final float alpha, final float beta, final boolean transposeA) {
        final FloatMatrix1D z_loc;
        if (z == null) {
            z_loc = new DenseFloatMatrix1D(this.rows);
        } else {
            z_loc = z;
        }
        if (transposeA)
            return viewDice().zMult(y, z_loc, alpha, beta, false);
        if (!(y instanceof DenseFloatMatrix1D && z_loc instanceof DenseFloatMatrix1D))
            return super.zMult(y, z_loc, alpha, beta, transposeA);

        if (columns != y.size() || rows > z_loc.size())
            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort() + ", " + z_loc.toStringShort());

        final float[] elemsY = (float[]) y.elements();
        final float[] elemsZ = (float[]) z_loc.elements();
        if (elements == null || elemsY == null || elemsZ == null)
            throw new InternalError();
        final int strideY = y.stride();
        final int strideZ = z_loc.stride();
        final int zero = index(0, 0);
        final int zeroY = y.index(0);
        final int zeroZ = z_loc.index(0);
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
                        int idxZero = zero + startrow * rowStride;
                        int idxZeroZ = zeroZ + startrow * strideZ;
                        for (int r = startrow; r < stoprow; r++) {
                            float sum = 0;
                            int idx = idxZero;
                            int idxY = zeroY;
                            for (int c = 0; c < columns; c++) {
                                sum += elements[idx] * elemsY[idxY];
                                idx += columnStride;
                                idxY += strideY;
                            }
                            elemsZ[idxZeroZ] = alpha * sum + beta * elemsZ[idxZeroZ];
                            idxZero += rowStride;
                            idxZeroZ += strideZ;
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
            int idxZero = zero;
            int idxZeroZ = zeroZ;
            for (int r = 0; r < rows; r++) {
                float sum = 0;
                int idx = idxZero;
                int idxY = zeroY;
                for (int c = 0; c < columns; c++) {
                    sum += elements[idx] * elemsY[idxY];
                    idx += columnStride;
                    idxY += strideY;
                }
                elemsZ[idxZeroZ] = alpha * sum + beta * elemsZ[idxZeroZ];
                idxZero += rowStride;
                idxZeroZ += strideZ;
            }
        }
        z = z_loc;
        return z;
    }

    public FloatMatrix2D zMult(final FloatMatrix2D B, FloatMatrix2D C, final float alpha, final float beta, final boolean transposeA, final boolean transposeB) {
        final int m = rows;
        final int n = columns;
        final int p = B.columns();
        if (C == null)
            C = new DenseFloatMatrix2D(m, p);
        /*
         * determine how to split and parallelize best into blocks if more
         * B.columns than tasks --> split B.columns, as follows:
         * 
         * xx|xx|xxx B xx|xx|xxx xx|xx|xxx A xxx xx|xx|xxx C xxx xx|xx|xxx xxx
         * xx|xx|xxx xxx xx|xx|xxx xxx xx|xx|xxx
         * 
         * if less B.columns than tasks --> split A.rows, as follows:
         * 
         * xxxxxxx B xxxxxxx xxxxxxx A xxx xxxxxxx C xxx xxxxxxx --- ------- xxx
         * xxxxxxx xxx xxxxxxx --- ------- xxx xxxxxxx
         */
        if (transposeA)
            return viewDice().zMult(B, C, alpha, beta, false, transposeB);
        if (B instanceof SparseFloatMatrix2D || B instanceof RCFloatMatrix2D) {
            // exploit quick sparse mult
            // A*B = (B' * A')'
            if (C == null) {
                return B.zMult(this, null, alpha, beta, !transposeB, true).viewDice();
            } else {
                B.zMult(this, C.viewDice(), alpha, beta, !transposeB, true);
                return C;
            }
        }
        if (transposeB)
            return this.zMult(B.viewDice(), C, alpha, beta, transposeA, false);

        if (!(C instanceof DenseFloatMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);

        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + this.toStringShort() + ", " + B.toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibe result matrix: " + this.toStringShort() + ", " + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        long flops = 2L * m * n * p;
        int noOfTasks = (int) Math.min(flops / 30000, ConcurrencyUtils.getNumberOfProcessors()); // each
        /* thread should process at least 30000 flops */
        boolean splitB = (p >= noOfTasks);
        int width = splitB ? p : m;
        noOfTasks = Math.min(width, noOfTasks);

        if (noOfTasks < 2) { /*
                              * parallelization doesn't pay off (too much start
                              * up overhead)
                              */
            return this.zMultSeq(B, C, alpha, beta, transposeA, transposeB);
        }

        // set up concurrent tasks
        int span = width / noOfTasks;
        final Future[] subTasks = new Future[noOfTasks];
        for (int i = 0; i < noOfTasks; i++) {
            final int offset = i * span;
            if (i == noOfTasks - 1)
                span = width - span * i; // last span may be a bit larger

            final FloatMatrix2D AA, BB, CC;
            if (splitB) {
                // split B along columns into blocks
                AA = this;
                BB = B.viewPart(0, offset, n, span);
                CC = C.viewPart(0, offset, m, span);
            } else {
                // split A along rows into blocks
                AA = this.viewPart(offset, 0, span, n);
                BB = B;
                CC = C.viewPart(offset, 0, span, p);
            }

            subTasks[i] = ConcurrencyUtils.threadPool.submit(new Runnable() {
                public void run() {
                    ((DenseFloatMatrix2D) AA).zMultSeq(BB, CC, alpha, beta, transposeA, transposeB);
                }
            });
        }

        try {
            for (int j = 0; j < noOfTasks; j++) {
                subTasks[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return C;
    }

    public float zSum() {
        float sum = 0;
        if (elements == null)
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
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float sum = 0;
                        int idx = zero + startrow * rowStride;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                sum += elements[i];
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        return sum;
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    sum += (Float) futures[j].get();
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
                    sum += elements[i];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return sum;
    }

    protected boolean haveSharedCellsRaw(FloatMatrix2D other) {
        if (other instanceof SelectedDenseFloatMatrix2D) {
            SelectedDenseFloatMatrix2D otherMatrix = (SelectedDenseFloatMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFloatMatrix2D) {
            DenseFloatMatrix2D otherMatrix = (DenseFloatMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    protected FloatMatrix1D like1D(int size, int zero, int stride) {
        return new DenseFloatMatrix1D(size, this.elements, zero, stride);
    }

    protected FloatMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseFloatMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }

    protected FloatMatrix2D zMultSeq(FloatMatrix2D B, FloatMatrix2D C, float alpha, float beta, boolean transposeA, boolean transposeB) {
        if (transposeA)
            return viewDice().zMult(B, C, alpha, beta, false, transposeB);
        if (B instanceof SparseFloatMatrix2D || B instanceof RCFloatMatrix2D) {
            // exploit quick sparse mult
            // A*B = (B' * A')'
            if (C == null) {
                return B.zMult(this, null, alpha, beta, !transposeB, true).viewDice();
            } else {
                B.zMult(this, C.viewDice(), alpha, beta, !transposeB, true);
                return C;
            }
        }
        if (transposeB)
            return this.zMult(B.viewDice(), C, alpha, beta, transposeA, false);

        int m = rows;
        int n = columns;
        int p = B.columns();
        if (C == null)
            C = new DenseFloatMatrix2D(m, p);
        if (!(C instanceof DenseFloatMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + B.toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", " + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        DenseFloatMatrix2D BB = (DenseFloatMatrix2D) B;
        DenseFloatMatrix2D CC = (DenseFloatMatrix2D) C;
        final float[] AElems = this.elements;
        final float[] BElems = BB.elements;
        final float[] CElems = CC.elements;
        if (AElems == null || BElems == null || CElems == null)
            throw new InternalError();

        int cA = this.columnStride;
        int cB = BB.columnStride;
        int cC = CC.columnStride;

        int rA = this.rowStride;
        int rB = BB.rowStride;
        int rC = CC.rowStride;

        /*
         * A is blocked to hide memory latency xxxxxxx B xxxxxxx xxxxxxx A xxx
         * xxxxxxx C xxx xxxxxxx --- ------- xxx xxxxxxx xxx xxxxxxx --- -------
         * xxx xxxxxxx
         */
        final int BLOCK_SIZE = 30000; // * 8 == Level 2 cache in bytes
        int m_optimal = (BLOCK_SIZE - n) / (n + 1);
        if (m_optimal <= 0)
            m_optimal = 1;
        int blocks = m / m_optimal;
        int rr = 0;
        if (m % m_optimal != 0)
            blocks++;
        for (; --blocks >= 0;) {
            int jB = BB.index(0, 0);
            int indexA = index(rr, 0);
            int jC = CC.index(rr, 0);
            rr += m_optimal;
            if (blocks == 0)
                m_optimal += m - rr;

            for (int j = p; --j >= 0;) {
                int iA = indexA;
                int iC = jC;
                for (int i = m_optimal; --i >= 0;) {
                    int kA = iA;
                    int kB = jB;
                    float s = 0;

                    // loop unrolled
                    kA -= cA;
                    kB -= rB;

                    for (int k = n % 4; --k >= 0;) {
                        s += AElems[kA += cA] * BElems[kB += rB];
                    }
                    for (int k = n / 4; --k >= 0;) {
                        s += AElems[kA += cA] * BElems[kB += rB] + AElems[kA += cA] * BElems[kB += rB] + AElems[kA += cA] * BElems[kB += rB] + AElems[kA += cA] * BElems[kB += rB];
                    }

                    CElems[iC] = alpha * s + beta * CElems[iC];
                    iA += rA;
                    iC += rC;
                }
                jB += cB;
                jC += cC;
            }
        }
        return C;
    }
}
