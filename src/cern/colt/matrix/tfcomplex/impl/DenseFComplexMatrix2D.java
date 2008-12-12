/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.impl;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import cern.jet.math.tfcomplex.FComplex;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>complex</tt> elements. <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array, addressed in row
 * major. Complex data is represented by 2 float values in sequence, i.e.
 * elements[idx] constitute the real part and elements[idx+1] constitute the
 * imaginary part, where idx = index(0,0) + row * rowStride + column *
 * columnStride. Note that this implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*rows()*2*columns()</tt>. Thus, a 1000*1000 matrix uses
 * 16 MB.
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
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseFComplexMatrix2D extends FComplexMatrix2D {
    static final long serialVersionUID = 1020177651L;

    private FloatFFT_2D fft2;

    /**
     * The elements of this matrix. elements are stored in row major. Complex
     * data is represented by 2 float values in sequence, i.e. elements[idx]
     * constitute the real part and elements[idx+1] constitute the imaginary
     * part, where idx = index(0,0) + row * rowStride + column * columnStride.
     */
    protected float[] elements;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form
     * <tt>re = values[row][2*column]; im = values[row][2*column+1]</tt> and
     * have exactly the same number of rows and columns as the receiver. Due to
     * the fact that complex data is represented by 2 float values in sequence:
     * the real and imaginary parts, the new matrix will be of the size
     * values.length by values[0].length / 2.
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
    public DenseFComplexMatrix2D(float[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length / 2);
        assign(values);
    }

    /**
     * Constructs a complex matrix with the same size as <tt>realPart</tt>
     * matrix and fills the real part of this matrix with elements of
     * <tt>realPart</tt>.
     * 
     * @param realPart
     *            a real matrix whose elements become a real part of this matrix
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public DenseFComplexMatrix2D(FloatMatrix2D realPart) {
        this(realPart.rows(), realPart.columns());
        assignReal(realPart);
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
    public DenseFComplexMatrix2D(int rows, int columns) {
        setUp(rows, columns, 0, 0, 2 * columns, 2);
        this.elements = new float[rows * 2 * columns];
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
    public DenseFComplexMatrix2D(int rows, int columns, float[] elements, int rowZero, int columnZero, int rowStride, int columnStride) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = false;
    }

    public float[] aggregate(final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr, final cern.colt.function.tfcomplex.FComplexFComplexFunction f) {
        float[] b = new float[2];
        if (size() == 0) {
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        final int zero = index(0, 0);
        float[] a = null;
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
                        int idx = zero + startrow * rowStride;
                        float[] a = f.apply(new float[] { elements[idx], elements[idx + 1] });
                        int d = 1;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = d; c < columns; c++) {
                                idx = zero + r * rowStride + c * columnStride;
                                a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (float[]) futures[j].get();
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
            a = f.apply(new float[] { elements[zero], elements[zero + 1] });
            int d = 1; // first cell already done
            int idx;
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    idx = zero + r * rowStride + c * columnStride;
                    a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }));
                }
                d = 0;
            }
        }
        return a;
    }

    public float[] aggregate(final FComplexMatrix2D other, final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction aggr, final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction f) {
        if (!(other instanceof DenseFComplexMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        float[] b = new float[2];
        if (size() == 0) {
            b[0] = Float.NaN;
            b[1] = Float.NaN;
            return b;
        }
        final int zero = index(0, 0);
        final int zeroOther = other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final float[] elemsOther = (float[]) other.elements();
        float[] a = null;
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
                        int idx = zero + startrow * rowStride;
                        int idxOther = zeroOther + startrow * rowStrideOther;
                        float[] a = f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] });
                        int d = 1;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = d; c < columns; c++) {
                                idx = zero + r * rowStride + c * columnStride;
                                idxOther = zeroOther + r * rowStrideOther + c * colStrideOther;
                                a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] }));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            try {
                for (int j = 0; j < np; j++) {
                    results[j] = (float[]) futures[j].get();
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
            int idx;
            int idxOther;
            a = f.apply(new float[] { elements[zero], elements[zero + 1] }, new float[] { elemsOther[zeroOther], elemsOther[zeroOther + 1] });
            int d = 1; // first cell already done
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    idx = zero + r * rowStride + c * columnStride;
                    idxOther = zeroOther + r * rowStrideOther + c * colStrideOther;
                    a = aggr.apply(a, f.apply(new float[] { elements[idx], elements[idx + 1] }, new float[] { elemsOther[idxOther], elemsOther[idxOther + 1] }));
                }
                d = 0;
            }
        }
        return a;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
        final int zero = index(0, 0);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tfcomplex.FComplexMult) {
                float[] multiplicator = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
                if (multiplicator[0] == 1 && multiplicator[1] == 0)
                    return this;
                if (multiplicator[0] == 0 && multiplicator[1] == 0)
                    return assign(0, 0);
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
                        float[] tmp = new float[2];
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                tmp[0] = elements[i];
                                tmp[1] = elements[i + 1];
                                tmp = function.apply(tmp);
                                elements[i] = tmp[0];
                                elements[i + 1] = tmp[1];
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
            float[] tmp = new float[2];
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    tmp[0] = elements[i];
                    tmp[1] = elements[i + 1];
                    tmp = function.apply(tmp);
                    elements[i] = tmp[0];
                    elements[i + 1] = tmp[1];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond, final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
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
                        float[] elem = new float[2];
                        int idx = zero + startrow * rowStride;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem[0] = elements[i];
                                elem[1] = elements[i + 1];
                                if (cond.apply(elem) == true) {
                                    elem = function.apply(elem);
                                    elements[i] = elem[0];
                                    elements[i + 1] = elem[1];
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
            float[] elem = new float[2];
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem[0] = elements[i];
                    elem[1] = elements[i + 1];
                    if (cond.apply(elem) == true) {
                        elem = function.apply(elem);
                        elements[i] = elem[0];
                        elements[i + 1] = elem[1];
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexProcedure cond, final float[] value) {
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
                        float[] elem = new float[2];
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elem[0] = elements[i];
                                elem[1] = elements[i + 1];
                                if (cond.apply(elem) == true) {
                                    elements[i] = value[0];
                                    elements[i + 1] = value[1];
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
            float[] elem = new float[2];
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    elem[0] = elements[i];
                    elem[1] = elements[i + 1];
                    if (cond.apply(elem) == true) {
                        elements[i] = value[0];
                        elements[i + 1] = value[1];
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexRealFunction function) {
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
                        float[] tmp = new float[2];
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                tmp[0] = elements[i];
                                tmp[1] = elements[i + 1];
                                tmp[0] = function.apply(tmp);
                                elements[i] = tmp[0];
                                elements[i + 1] = 0;
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
            float[] tmp = new float[2];
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    tmp[0] = elements[i];
                    tmp[1] = elements[i + 1];
                    tmp[0] = function.apply(tmp);
                    elements[i] = tmp[0];
                    elements[i + 1] = 0;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final FComplexMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof DenseFComplexMatrix2D)) {
            super.assign(source);
            return this;
        }
        final DenseFComplexMatrix2D other_final = (DenseFComplexMatrix2D) source;
        if (other_final == this)
            return this; // nothing to do
        checkShape(other_final);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if (this.isNoView && other_final.isNoView) { // quickest
            System.arraycopy(other_final.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }

        DenseFComplexMatrix2D other = (DenseFComplexMatrix2D) source;
        if (haveSharedCells(other)) {
            FComplexMatrix2D c = other.copy();
            if (!(c instanceof DenseFComplexMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseFComplexMatrix2D) c;
        }

        final float[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        final int zeroOther = other.index(0, 0);
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
                        int idxOther = zeroOther + startrow * rowStrideOther;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elements[i] = elemsOther[j];
                                elements[i + 1] = elemsOther[j + 1];
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
                    elements[i + 1] = elemsOther[j + 1];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final FComplexMatrix2D y, final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseFComplexMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        checkShape(y);
        final float[] elemsOther = ((DenseFComplexMatrix2D) y).elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int columnStrideOther = y.columnStride();
        final int rowStrideOther = y.rowStride();
        final int zeroOther = y.index(0, 0);
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
                        int idxOther = zeroOther + startrow * rowStrideOther;
                        float[] tmp1 = new float[2];
                        float[] tmp2 = new float[2];
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                tmp1[0] = elements[i];
                                tmp1[1] = elements[i + 1];
                                tmp2[0] = elemsOther[j];
                                tmp2[1] = elemsOther[j + 1];
                                tmp1 = function.apply(tmp1, tmp2);
                                elements[i] = tmp1[0];
                                elements[i + 1] = tmp1[1];
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
            float[] tmp1 = new float[2];
            float[] tmp2 = new float[2];
            int idx = zero;
            int idxOther = zeroOther;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                    tmp1[0] = elements[i];
                    tmp1[1] = elements[i + 1];
                    tmp2[0] = elemsOther[j];
                    tmp2[1] = elemsOther[j + 1];
                    tmp1 = function.apply(tmp1, tmp2);
                    elements[i] = tmp1[0];
                    elements[i + 1] = tmp1[1];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final float re, final float im) {
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
                                elements[i] = re;
                                elements[i + 1] = im;
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
                    elements[i] = re;
                    elements[i + 1] = im;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final float[] values) {
        if (values.length != rows * 2 * columns)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + "rows()*2*columns()=" + rows() * 2 * columns());
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if (this.isNoView) {
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            final int zero = index(0, 0);
            if ((np > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future[] futures = new Future[np];
                int k = rows / np;
                for (int j = 0; j < np; j++) {
                    final int startrow = j * k;
                    final int stoprow;
                    final int glob_idxOther = j * k * 2 * columns;
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
                                    elements[i + 1] = values[idxOther++];
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
                        elements[i + 1] = values[idxOther++];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final float[][] values) {
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
                            int idx = 2 * columns;
                            int i = startrow * rowStride;
                            for (int r = startrow; r < stoprow; r++) {
                                float[] currentRow = values[r];
                                if (currentRow.length != idx)
                                    throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "2*columns()=" + idx);
                                System.arraycopy(currentRow, 0, elements, i, idx);
                                i += idx;
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
                int idx = 2 * columns;
                int i = 0;
                for (int r = 0; r < rows; r++) {
                    float[] currentRow = values[r];
                    if (currentRow.length != idx)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "2*columns()=" + idx);
                    System.arraycopy(currentRow, 0, this.elements, i, idx);
                    i += idx;
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
                                if (currentRow.length != 2 * columns)
                                    throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "2*columns()=" + 2 * columns());
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = currentRow[2 * c];
                                    elements[i + 1] = currentRow[2 * c + 1];
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
                    if (currentRow.length != 2 * columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns=" + currentRow.length + "2*columns()=" + 2 * columns());
                    for (int i = idx, c = 0; c < columns; c++) {
                        elements[i] = currentRow[2 * c];
                        elements[i + 1] = currentRow[2 * c + 1];
                        i += columnStride;
                    }
                    idx += rowStride;
                }
            }
        }
        return this;
    }

    public FComplexMatrix2D assignImaginary(final FloatMatrix2D other) {
        checkShape(other);
        final int columnStrideOther = other.columnStride();
        final int rowStrideOther = other.rowStride();
        final int zeroOther = other.index(0, 0);
        final int zero = index(0, 0);
        final float[] elemsOther = ((DenseFloatMatrix2D) other).elements();
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                ;
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
                                elements[i + 1] = elemsOther[j];
                                elements[i] = 0;
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
                    elements[i + 1] = elemsOther[j];
                    elements[i] = 0;
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return this;
    }

    public FComplexMatrix2D assignReal(final FloatMatrix2D other) {
        checkShape(other);
        final int columnStrideOther = other.columnStride();
        final int rowStrideOther = other.rowStride();
        final int zeroOther = other.index(0, 0);
        final int zero = index(0, 0);
        final float[] elemsOther = ((DenseFloatMatrix2D) other).elements();
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            int k = rows / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                ;
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
                                elements[i + 1] = 0;
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
                    elements[i + 1] = 0;
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
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
                                if ((elements[i] != 0.0) || (elements[i + 1] != 0.0))
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
                    if ((elements[i] != 0.0) || (elements[i + 1] != 0.0))
                        cardinality++;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return cardinality;
    }

    public void fft2() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.complexForward(elements);
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void fftColumns() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
        FComplexMatrix1D column;
        for (int c = 0; c < columns; c++) {
            column = viewColumn(c).copy();
            column.fft();
            viewColumn(c).assign(column);
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void fftRows() {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
    	FComplexMatrix1D row;
        for (int r = 0; r < rows; r++) {
            row = viewRow(r).copy();
            row.fft();
            viewRow(r).assign(row);
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public FComplexMatrix2D forEachNonZero(final cern.colt.function.tfcomplex.IntIntFComplexFunction function) {
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
                        float[] value = new float[2];
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                value[0] = elements[i];
                                value[1] = elements[i + 1];
                                if (value[0] != 0 || value[1] != 0) {
                                    float[] v = function.apply(r, c, value);
                                    elements[i] = v[0];
                                    elements[i + 1] = v[1];
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
            float[] value = new float[2];
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    value[0] = elements[i];
                    value[1] = elements[i + 1];
                    if (value[0] != 0 || value[1] != 0) {
                        float[] v = function.apply(r, c, value);
                        elements[i] = v[0];
                        elements[i + 1] = v[1];
                    }
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return this;
    }

    public FComplexMatrix2D getConjugateTranspose() {
        FComplexMatrix2D transpose = this.viewDice().copy();
        final float[] elemsOther = ((DenseFComplexMatrix2D) transpose).elements;
        final int zeroOther = transpose.index(0, 0);
        final int columnStrideOther = transpose.columnStride();
        final int rowStrideOther = transpose.rowStride();
        int np = ConcurrencyUtils.getNumberOfProcessors();
        final int columnsOther = transpose.columns();
        final int rowsOther = transpose.rows();
        if ((np > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            Future[] futures = new Future[np];
            int k = rowsOther / np;
            for (int j = 0; j < np; j++) {
                final int startrow = j * k;
                final int stoprow;
                if (j == np - 1) {
                    stoprow = rowsOther;
                } else {
                    stoprow = startrow + k;
                }
                futures[j] = ConcurrencyUtils.threadPool.submit(new Runnable() {

                    public void run() {
                        int idxOther = zeroOther + startrow * rowStrideOther;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int c = 0; c < columnsOther; c++) {
                                elemsOther[idxOther + 1] = -elemsOther[idxOther + 1];
                                idxOther += columnStrideOther;
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
            int idxOther = zeroOther;
            for (int r = 0; r < rowsOther; r++) {
                for (int c = 0; c < columnsOther; c++) {
                    elemsOther[idxOther + 1] = -elemsOther[idxOther + 1];
                    idxOther += columnStrideOther;
                }
            }
        }
        return transpose;
    }

    public float[] elements() {
        return elements;
    }

    public FloatMatrix2D getImaginaryPart() {
        final DenseFloatMatrix2D Im = new DenseFloatMatrix2D(rows, columns);
        final float[] elemsOther = (float[]) Im.elements();
        final int columnStrideOther = Im.columnStride();
        final int rowStrideOther = Im.rowStride();
        final int zeroOther = Im.index(0, 0);
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
                        int idxOther = zeroOther + startrow * rowStrideOther;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elemsOther[j] = elements[i + 1];
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
                    elemsOther[j] = elements[i + 1];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return Im;
    }

    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList, final ArrayList<float[]> valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();

        int idx = index(0, 0);
        for (int r = 0; r < rows; r++) {
            for (int i = idx, c = 0; c < columns; c++) {
                float[] value = new float[2];
                value[0] = elements[i];
                value[1] = elements[i + 1];
                if (value[0] != 0 || value[1] != 0) {
                    synchronized (rowList) {
                        rowList.add(r);
                        columnList.add(c);
                        valueList.add(value);
                    }
                }
                i += columnStride;
            }
            idx += rowStride;
        }

    }

    public float[] getQuick(int row, int column) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        return new float[] { elements[idx], elements[idx + 1] };
    }

    public FloatMatrix2D getRealPart() {
        final DenseFloatMatrix2D R = new DenseFloatMatrix2D(rows, columns);
        final float[] elemsOther = (float[]) R.elements();
        final int columnStrideOther = R.columnStride();
        final int rowStrideOther = R.rowStride();
        final int zeroOther = R.index(0, 0);
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
                        int idxOther = zeroOther + startrow * rowStrideOther;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elemsOther[j] = elements[i];
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
                    elemsOther[j] = elements[i];
                    i += columnStride;
                    j += columnStrideOther;
                }
                idx += rowStride;
                idxOther += rowStrideOther;
            }
        }
        return R;
    }

    public void ifft2(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.complexInverse(elements, scale);
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void ifftColumns(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
    	FComplexMatrix1D column;
        for (int c = 0; c < columns; c++) {
            column = viewColumn(c).copy();
            column.ifft(scale);
            viewColumn(c).assign(column);
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public void ifftRows(boolean scale) {
    	int oldNp = ConcurrencyUtils.getNumberOfProcessors();
    	ConcurrencyUtils.setNumberOfProcessors(ConcurrencyUtils.nextPow2(oldNp));
    	FComplexMatrix1D row;
        for (int r = 0; r < rows; r++) {
            row = viewRow(r).copy();
            row.ifft(scale);
            viewRow(r).assign(row);
        }
        ConcurrencyUtils.setNumberOfProcessors(oldNp);
    }

    public FComplexMatrix2D like(int rows, int columns) {
        return new DenseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix1D like1D(int size) {
        return new DenseFComplexMatrix1D(size);
    }

    public void setQuick(int row, int column, float re, float im) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = re;
        elements[idx + 1] = im;
    }

    public void setQuick(int row, int column, float[] value) {
        int idx = rowZero + row * rowStride + columnZero + column * columnStride;
        elements[idx] = value[0];
        elements[idx + 1] = value[1];
    }

    public float[][] toArray() {
        final float[][] values = new float[rows][2 * columns];
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
                            for (int i = idx, c = 0; c < columns; c++) {
                                values[r][2 * c] = elements[i];
                                values[r][2 * c + 1] = elements[i + 1];
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
                    values[r][2 * c] = elements[i];
                    values[r][2 * c + 1] = elements[i + 1];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return values;
    }

    public FComplexMatrix1D vectorize() {
        final FComplexMatrix1D v = new DenseFComplexMatrix1D(size());
        final int zero = index(0, 0);
        final int zeroOther = v.index(0);
        final int strideOther = v.stride();
        final float[] elemsOther = (float[]) v.elements();
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
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
                                elemsOther[idxOther + 1] = elements[idx + 1];
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
            int idx = 0;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                idx = zero + c * columnStride;
                for (int r = 0; r < rows; r++) {
                    elemsOther[idxOther] = elements[idx];
                    elemsOther[idxOther + 1] = elements[idx + 1];
                    idx += rowStride;
                    idxOther += strideOther;
                }
            }
        }
        return v;
    }

    public FComplexMatrix1D zMult(final FComplexMatrix1D y, FComplexMatrix1D z, final float[] alpha, final float[] beta, boolean transposeA) {
        if (transposeA)
            return getConjugateTranspose().zMult(y, z, alpha, beta, false);
        final FComplexMatrix1D zLoc;
        if (z == null) {
            zLoc = new DenseFComplexMatrix1D(this.rows);
        } else {
            zLoc = z;
        }
        if (columns != y.size() || rows > zLoc.size())
            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort() + ", " + zLoc.toStringShort());
        final float[] elemsY = (float[]) y.elements();
        final float[] elemsZ = (float[]) zLoc.elements();
        if (elements == null || elemsY == null || elemsZ == null)
            throw new InternalError();
        final int strideY = y.stride();
        final int strideZ = zLoc.stride();
        final int zero = index(0, 0);
        final int zeroY = y.index(0);
        final int zeroZ = zLoc.index(0);
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
                        float[] sum = new float[2];
                        int idxZero = zero + startrow * rowStride;
                        int idxZeroZ = zeroZ + startrow * strideZ;
                        float[] elem = new float[2];
                        float[] elemY = new float[2];
                        float[] elemZ = new float[2];
                        float[] tmp = new float[2];
                        for (int r = startrow; r < stoprow; r++) {
                            sum[0] = 0;
                            sum[1] = 0;
                            int idx = idxZero;
                            int idxY = zeroY;
                            for (int c = 0; c < columns; c++) {
                                elem[0] = elements[idx];
                                elem[1] = elements[idx + 1];
                                elemY[0] = elemsY[idxY];
                                elemY[1] = elemsY[idxY + 1];
                                sum = FComplex.plus(sum, FComplex.mult(elem, elemY));
                                idx += columnStride;
                                idxY += strideY;
                            }
                            elemZ[0] = elemsZ[idxZeroZ];
                            elemZ[1] = elemsZ[idxZeroZ + 1];
                            tmp = FComplex.plus(FComplex.mult(sum, alpha), FComplex.mult(elemZ, beta));
                            elemsZ[idxZeroZ] = tmp[0];
                            elemsZ[idxZeroZ + 1] = tmp[1];
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
            float[] sum = new float[2];
            float[] elem = new float[2];
            float[] elemY = new float[2];
            float[] elemZ = new float[2];
            float[] tmp = new float[2];
            for (int r = 0; r < rows; r++) {
                sum[0] = 0;
                sum[1] = 0;
                int idx = idxZero;
                int idxY = zeroY;
                for (int c = 0; c < columns; c++) {
                    elem[0] = elements[idx];
                    elem[1] = elements[idx + 1];
                    elemY[0] = elemsY[idxY];
                    elemY[1] = elemsY[idxY + 1];
                    sum = FComplex.plus(sum, FComplex.mult(elem, elemY));
                    idx += columnStride;
                    idxY += strideY;
                }
                elemZ[0] = elemsZ[idxZeroZ];
                elemZ[1] = elemsZ[idxZeroZ + 1];
                tmp = FComplex.plus(FComplex.mult(sum, alpha), FComplex.mult(elemZ, beta));
                elemsZ[idxZeroZ] = tmp[0];
                elemsZ[idxZeroZ + 1] = tmp[1];
                idxZero += rowStride;
                idxZeroZ += strideZ;
            }
        }
        z = zLoc;
        return z;
    }

    public FComplexMatrix2D zMult(final FComplexMatrix2D B, FComplexMatrix2D C, final float[] alpha, final float[] beta, final boolean transposeA, final boolean transposeB) {
        if (transposeA)
            return getConjugateTranspose().zMult(B, C, alpha, beta, false, transposeB);
        if (transposeB)
            return this.zMult(B.getConjugateTranspose(), C, alpha, beta, transposeA, false);
        final int m = rows;
        final int n = columns;
        final int p = B.columns();
        if (C == null)
            C = new DenseFComplexMatrix2D(m, p);
        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + B.toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibe result matrix: " + toStringShort() + ", " + B.toStringShort() + ", " + C.toStringShort());
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

            final FComplexMatrix2D AA, BB, CC;
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
                    ((DenseFComplexMatrix2D) AA).zMultSeq(BB, CC, alpha, beta, transposeA, transposeB);
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

    protected FComplexMatrix2D zMultSeq(FComplexMatrix2D B, FComplexMatrix2D C, float[] alpha, float[] beta, boolean transposeA, boolean transposeB) {
        if (transposeA)
            return getConjugateTranspose().zMult(B, C, alpha, beta, false, transposeB);
        if (transposeB)
            return this.zMult(B.getConjugateTranspose(), C, alpha, beta, transposeA, false);
        int m = rows;
        int n = columns;
        int p = B.columns();
        if (C == null)
            C = new DenseFComplexMatrix2D(m, p);
        if (!(C instanceof DenseFComplexMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + B.toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", " + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        DenseFComplexMatrix2D BB = (DenseFComplexMatrix2D) B;
        DenseFComplexMatrix2D CC = (DenseFComplexMatrix2D) C;
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
        float[] s = new float[2];
        float[] elemA = new float[2];
        float[] elemB = new float[2];
        float[] elemC = new float[2];
        float[] tmp = new float[2];
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
                    s[0] = 0;
                    s[1] = 0;
                    // loop unrolled
                    kA -= cA;
                    kB -= rB;
                    for (int k = n % 4; --k >= 0;) {
                        kA += cA;
                        kB += rB;
                        elemA[0] = AElems[kA];
                        elemA[1] = AElems[kA + 1];
                        elemB[0] = BElems[kB];
                        elemB[1] = BElems[kB + 1];
                        s = FComplex.plus(s, FComplex.mult(elemA, elemB));
                    }
                    for (int k = n / 4; --k >= 0;) {
                        kA += cA;
                        kB += rB;
                        elemA[0] = AElems[kA];
                        elemA[1] = AElems[kA + 1];
                        elemB[0] = BElems[kB];
                        elemB[1] = BElems[kB + 1];
                        s = FComplex.plus(s, FComplex.mult(elemA, elemB));
                        kA += cA;
                        kB += rB;
                        elemA[0] = AElems[kA];
                        elemA[1] = AElems[kA + 1];
                        elemB[0] = BElems[kB];
                        elemB[1] = BElems[kB + 1];
                        s = FComplex.plus(s, FComplex.mult(elemA, elemB));
                        kA += cA;
                        kB += rB;
                        elemA[0] = AElems[kA];
                        elemA[1] = AElems[kA + 1];
                        elemB[0] = BElems[kB];
                        elemB[1] = BElems[kB + 1];
                        s = FComplex.plus(s, FComplex.mult(elemA, elemB));
                        kA += cA;
                        kB += rB;
                        elemA[0] = AElems[kA];
                        elemA[1] = AElems[kA + 1];
                        elemB[0] = BElems[kB];
                        elemB[1] = BElems[kB + 1];
                        s = FComplex.plus(s, FComplex.mult(elemA, elemB));
                    }
                    elemC[0] = CElems[iC];
                    elemC[1] = CElems[iC + 1];
                    tmp = FComplex.plus(FComplex.mult(alpha, s), FComplex.mult(beta, elemC));
                    CElems[iC] = tmp[0];
                    CElems[iC + 1] = tmp[1];
                    iA += rA;
                    iC += rC;
                }
                jB += cB;
                jC += cC;
            }
        }
        return C;
    }

    public float[] zSum() {
        float[] sum = new float[2];
        final int zero = index(0, 0);
        int np = ConcurrencyUtils.getNumberOfProcessors();
        if ((np > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
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
                futures[j] = ConcurrencyUtils.threadPool.submit(new Callable<float[]>() {

                    public float[] call() throws Exception {
                        float[] sum = new float[2];
                        int idx = zero + startrow * rowStride;
                        for (int r = startrow; r < stoprow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                sum[0] += elements[i];
                                sum[1] += elements[i + 1];
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        return sum;
                    }
                });
            }
            try {
                float[] tmp;
                for (int j = 0; j < np; j++) {
                    tmp = (float[]) futures[j].get();
                    sum[0] = sum[0] + tmp[0];
                    sum[1] = sum[1] + tmp[1];
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
                    sum[0] += elements[i];
                    sum[1] += elements[i + 1];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
        return sum;
    }

    protected boolean haveSharedCellsRaw(FComplexMatrix2D other) {
        if (other instanceof SelectedDenseFComplexMatrix2D) {
            SelectedDenseFComplexMatrix2D otherMatrix = (SelectedDenseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseFComplexMatrix2D) {
            DenseFComplexMatrix2D otherMatrix = (DenseFComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    public int index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    protected FComplexMatrix1D like1D(int size, int zero, int stride) {
        return new DenseFComplexMatrix1D(size, this.elements, zero, stride);
    }

    protected FComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseFComplexMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }
}
