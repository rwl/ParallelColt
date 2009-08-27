/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.io.MatrixInfo;
import cern.colt.matrix.io.MatrixSize;
import cern.colt.matrix.io.MatrixVectorReader;
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
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public DenseFloatMatrix2D(int rows, int columns) {
        setUp(rows, columns);
        this.elements = new float[rows * columns];
    }

    /**
     * Constructs a matrix with the given parameters.
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
     * @param isView
     *            if true then a matrix view is constructed
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    public DenseFloatMatrix2D(int rows, int columns, float[] elements, int rowZero, int columnZero, int rowStride,
            int columnStride, boolean isView) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = !isView;
    }

    /**
     * Constructs a matrix from MatrixVectorReader.
     * 
     * @param reader
     *            matrix reader
     * @throws IOException
     */
    public DenseFloatMatrix2D(MatrixVectorReader reader) throws IOException {
        MatrixInfo info;
        if (reader.hasInfo())
            info = reader.readMatrixInfo();
        else
            info = new MatrixInfo(true, MatrixInfo.MatrixField.Real, MatrixInfo.MatrixSymmetry.General);

        if (info.isPattern())
            throw new UnsupportedOperationException("Pattern matrices are not supported");
        if (info.isDense())
            throw new UnsupportedOperationException("Dense matrices are not supported");
        if (info.isComplex())
            throw new UnsupportedOperationException("Complex matrices are not supported");

        MatrixSize size = reader.readMatrixSize(info);
        setUp(size.numRows(), size.numColumns());
        this.elements = new float[rows * columns];
        int numEntries = size.numEntries();
        int[] columnIndexes = new int[numEntries];
        int[] rowIndexes = new int[numEntries];
        float[] values = new float[numEntries];
        reader.readCoordinate(rowIndexes, columnIndexes, values);
        for (int i = 0; i < numEntries; i++) {
            setQuick(rowIndexes[i], columnIndexes[i], values[i]);
        }
        if (info.isSymmetric()) {
            for (int i = 0; i < numEntries; i++) {
                if (rowIndexes[i] != columnIndexes[i]) {
                    setQuick(columnIndexes[i], rowIndexes[i], values[i]);
                }
            }
        } else if (info.isSkewSymmetric()) {
            for (int i = 0; i < numEntries; i++) {
                if (rowIndexes[i] != columnIndexes[i]) {
                    setQuick(columnIndexes[i], rowIndexes[i], -values[i]);
                }
            }
        }
    }

    public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr,
            final cern.colt.function.tfloat.FloatFunction f) {
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + (firstRow - 1) * rowStride + (columns - 1) * columnStride]);
                        int d = 1;
                        for (int r = firstRow; --r >= lastRow;) {
                            int ridx = zero + r * rowStride;
                            for (int c = columns - d; --c >= 0;) {
                                a = aggr.apply(a, f.apply(elements[ridx + c * columnStride]));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero + (rows - 1) * rowStride + (columns - 1) * columnStride]);
            int d = 1;
            for (int r = rows; --r >= 0;) {
                int ridx = zero + r * rowStride;
                for (int c = columns - d; --c >= 0;) {
                    a = aggr.apply(a, f.apply(elements[ridx + c * columnStride]));
                }
                d = 0;
            }
        }
        return a;
    }

    public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr,
            final cern.colt.function.tfloat.FloatFunction f, final cern.colt.function.tfloat.FloatProcedure cond) {
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float elem = elements[zero + firstRow * rowStride];
                        float a = 0;
                        if (cond.apply(elem) == true) {
                            a = f.apply(elem);
                        }
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
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
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
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

    public float aggregate(final cern.colt.function.tfloat.FloatFloatFunction aggr,
            final cern.colt.function.tfloat.FloatFunction f, final IntArrayList rowList, final IntArrayList columnList) {
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + rowElements[firstIdx] * rowStride + columnElements[firstIdx]
                                * columnStride]);
                        float elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            elem = elements[zero + rowElements[i] * rowStride + columnElements[i] * columnStride];
                            a = aggr.apply(a, f.apply(elem));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
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

    public float aggregate(final FloatMatrix2D other, final cern.colt.function.tfloat.FloatFloatFunction aggr,
            final cern.colt.function.tfloat.FloatFloatFunction f) {
        if (!(other instanceof DenseFloatMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int colStrideOther = other.columnStride();
        final float[] elementsOther = (float[]) other.elements();
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + firstRow * rowStride], elementsOther[zeroOther + firstRow
                                * rowStrideOther]);
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = d; c < columns; c++) {
                                a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride],
                                        elementsOther[zeroOther + r * rowStrideOther + c * colStrideOther]));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            int d = 1; // first cell already done
            a = f.apply(elements[zero], elementsOther[zeroOther]);
            for (int r = 0; r < rows; r++) {
                for (int c = d; c < columns; c++) {
                    a = aggr.apply(a, f.apply(elements[zero + r * rowStride + c * columnStride],
                            elementsOther[zeroOther + r * rowStrideOther + c * colStrideOther]));
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
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] =
                // mult*x[i]
                float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                if (multiplicator == 1)
                    return this;
                if (multiplicator == 0)
                    return assign(0);
            }
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        // specialization for speed
                        if (function instanceof cern.jet.math.tfloat.FloatMult) {
                            // x[i] = mult*x[i]
                            float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                            if (multiplicator == 1)
                                return;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elems[i] *= multiplicator;
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        } else {
                            // the general case x[i] = f(x[i])
                            for (int r = firstRow; r < lastRow; r++) {
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
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            // specialization for speed
            if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] =
                // mult*x[i]
                float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                if (multiplicator == 1)
                    return this;
                if (multiplicator == 0)
                    return assign(0);
                for (int r = rows; --r >= 0;) { // the general case
                    for (int i = idx, c = columns; --c >= 0;) {
                        elems[i] *= multiplicator;
                        i -= columnStride;
                    }
                    idx -= rowStride;
                }
            } else { // the general case x[i] = f(x[i])
                for (int r = rows; --r >= 0;) {
                    for (int i = idx, c = columns; --c >= 0;) {
                        elems[i] = function.apply(elems[i]);
                        i -= columnStride;
                    }
                    idx -= rowStride;
                }
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final cern.colt.function.tfloat.FloatProcedure cond,
            final cern.colt.function.tfloat.FloatFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        float elem;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
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
            ConcurrencyUtils.waitForCompletion(futures);
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
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        float elem;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
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
            ConcurrencyUtils.waitForCompletion(futures);
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
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                elems[i] = value;
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
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
            throw new IllegalArgumentException("Must have same length: length=" + values.length + " rows()*columns()="
                    + rows() * columns());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            final int zero = (int) index(0, 0);
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, rows);
                Future<?>[] futures = new Future[nthreads];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            int idxOther = firstRow * columns;
                            int idx = zero + firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = values[idxOther++];
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
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
            throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "rows()="
                    + rows());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView) {
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, rows);
                Future<?>[] futures = new Future[nthreads];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int i = firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                float[] currentRow = values[r];
                                if (currentRow.length != columns)
                                    throw new IllegalArgumentException(
                                            "Must have same number of columns in every row: columns="
                                                    + currentRow.length + "columns()=" + columns());
                                System.arraycopy(currentRow, 0, elements, i, columns);
                                i += columns;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int i = 0;
                for (int r = 0; r < rows; r++) {
                    float[] currentRow = values[r];
                    if (currentRow.length != columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "columns()=" + columns());
                    System.arraycopy(currentRow, 0, this.elements, i, columns);
                    i += columns;
                }
            }
        } else {
            final int zero = (int) index(0, 0);
            if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, rows);
                Future<?>[] futures = new Future[nthreads];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            int idx = zero + firstRow * rowStride;
                            for (int r = firstRow; r < lastRow; r++) {
                                float[] currentRow = values[r];
                                if (currentRow.length != columns)
                                    throw new IllegalArgumentException(
                                            "Must have same number of columns in every row: columns="
                                                    + currentRow.length + "columns()=" + columns());
                                for (int i = idx, c = 0; c < columns; c++) {
                                    elements[i] = currentRow[c];
                                    i += columnStride;
                                }
                                idx += rowStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = zero;
                for (int r = 0; r < rows; r++) {
                    float[] currentRow = values[r];
                    if (currentRow.length != columns)
                        throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                                + currentRow.length + "columns()=" + columns());
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
        DenseFloatMatrix2D other = (DenseFloatMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            FloatMatrix2D c = other.copy();
            if (!(c instanceof DenseFloatMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseFloatMatrix2D) c;
        }

        final float[] elementsOther = other.elements;
        if (elements == null || elementsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        int idxOther = zeroOther + firstRow * rowStrideOther;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                elements[i] = elementsOther[j];
                                i += columnStride;
                                j += columnStrideOther;
                            }
                            idx += rowStride;
                            idxOther += rowStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                    elements[i] = elementsOther[j];
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
        final float[] elementsOther = other.elements;
        if (elements == null || elementsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) {
                float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
                if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                    return this;
                }
            }
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx;
                        int idxOther;
                        // specialized for speed
                        if (function == cern.jet.math.tfloat.FloatFunctions.mult) {
                            // x[i] = x[i]*y[i]
                            idx = zero + firstRow * rowStride;
                            idxOther = zeroOther + firstRow * rowStrideOther;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] *= elementsOther[j];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else if (function == cern.jet.math.tfloat.FloatFunctions.div) {
                            // x[i] = x[i] / y[i]
                            idx = zero + firstRow * rowStride;
                            idxOther = zeroOther + firstRow * rowStrideOther;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] /= elementsOther[j];
                                    i += columnStride;
                                    j += columnStrideOther;
                                }
                                idx += rowStride;
                                idxOther += rowStrideOther;
                            }
                        } else if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) {
                            float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
                            if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                idx = zero + firstRow * rowStride;
                                idxOther = zeroOther + firstRow * rowStrideOther;
                                for (int r = firstRow; r < lastRow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] += elementsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = x[i] - y[i]
                                idx = zero + firstRow * rowStride;
                                idxOther = zeroOther + firstRow * rowStrideOther;
                                for (int r = firstRow; r < lastRow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] -= elementsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            } else { // the general case
                                // x[i] = x[i] + mult*y[i]
                                idx = zero + firstRow * rowStride;
                                idxOther = zeroOther + firstRow * rowStrideOther;
                                for (int r = firstRow; r < lastRow; r++) {
                                    for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                        elements[i] += multiplicator * elementsOther[j];
                                        i += columnStride;
                                        j += columnStrideOther;
                                    }
                                    idx += rowStride;
                                    idxOther += rowStrideOther;
                                }
                            }
                        } else { // the general case x[i] = f(x[i],y[i])
                            idx = zero + firstRow * rowStride;
                            idxOther = zeroOther + firstRow * rowStrideOther;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                                    elements[i] = function.apply(elements[i], elementsOther[j]);
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
            ConcurrencyUtils.waitForCompletion(futures);
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
                        elements[i] *= elementsOther[j];
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
                        elements[i] /= elementsOther[j];
                        i += columnStride;
                        j += columnStrideOther;
                    }
                    idx += rowStride;
                    idxOther += rowStrideOther;
                }
            } else if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) {
                float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
                if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                    return this;
                } else if (multiplicator == 1) { // x[i] = x[i] + y[i]
                    idx = zero;
                    idxOther = zeroOther;
                    for (int r = 0; r < rows; r++) {
                        for (int i = idx, j = idxOther, c = 0; c < columns; c++) {
                            elements[i] += elementsOther[j];
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
                            elements[i] -= elementsOther[j];
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
                            elements[i] += multiplicator * elementsOther[j];
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
                        elements[i] = function.apply(elements[i], elementsOther[j]);
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

    public FloatMatrix2D assign(final FloatMatrix2D y, final cern.colt.function.tfloat.FloatFloatFunction function,
            IntArrayList rowList, IntArrayList columnList) {
        checkShape(y);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final float[] elementsOther = (float[]) y.elements();
        final int zeroOther = (int) y.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = y.columnStride();
        final int rowStrideOther = y.rowStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx;
                        int idxOther;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                            idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i]
                                    * columnStrideOther;
                            elements[idx] = function.apply(elements[idx], elementsOther[idxOther]);
                        }
                    }

                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int i = 0; i < size; i++) {
                idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i] * columnStrideOther;
                elements[idx] = function.apply(elements[idx], elementsOther[idxOther]);
            }
        }
        return this;
    }

    public int cardinality() {
        int cardinality = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
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
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Integer) futures[j].get();
                }
                cardinality = results[0].intValue();
                for (int j = 1; j < nthreads; j++) {
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

    /**
     * Computes the 2D discrete cosine transform (DCT-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dct2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the discrete cosine transform (DCT-II) of each column of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dctColumns(final boolean scale) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFloatMatrix1D) viewColumn(c)).dct(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFloatMatrix1D) viewColumn(c)).dct(scale);
            }
        }
    }

    /**
     * Computes the discrete cosine transform (DCT-II) of each row of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dctRows(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFloatMatrix1D) viewRow(r)).dct(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFloatMatrix1D) viewRow(r)).dct(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D discrete Hartley transform (DHT) of this matrix.
     * 
     */
    public void dht2() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the discrete Hartley transform (DHT) of each column of this
     * matrix.
     * 
     */
    public void dhtColumns() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFloatMatrix1D) viewColumn(c)).dht();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFloatMatrix1D) viewColumn(c)).dht();
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the discrete Hartley transform (DHT) of each row of this matrix.
     * 
     */
    public void dhtRows() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFloatMatrix1D) viewRow(r)).dht();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFloatMatrix1D) viewRow(r)).dht();
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D discrete sine transform (DST-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dst2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the discrete sine transform (DST-II) of each column of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dstColumns(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFloatMatrix1D) viewColumn(c)).dst(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFloatMatrix1D) viewColumn(c)).dst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the discrete sine transform (DST-II) of each row of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dstRows(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFloatMatrix1D) viewRow(r)).dst(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFloatMatrix1D) viewRow(r)).dst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public float[] elements() {
        return elements;
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of this matrix. The
     * physical layout of the output data is as follows:
     * 
     * <pre>
     * this[k1][2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2], 
     * this[k1][2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2], 
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2, 
     * this[0][2*k2] = Re[0][k2] = Re[0][columns-k2], 
     * this[0][2*k2+1] = Im[0][k2] = -Im[0][columns-k2], 
     *       0&lt;k2&lt;columns/2, 
     * this[k1][0] = Re[k1][0] = Re[rows-k1][0], 
     * this[k1][1] = Im[k1][0] = -Im[rows-k1][0], 
     * this[rows-k1][1] = Re[k1][columns/2] = Re[rows-k1][columns/2], 
     * this[rows-k1][0] = -Im[k1][columns/2] = Im[rows-k1][columns/2], 
     *       0&lt;k1&lt;rows/2, 
     * this[0][0] = Re[0][0], 
     * this[0][1] = Re[0][columns/2], 
     * this[rows/2][0] = Re[rows/2][0], 
     * this[rows/2][1] = Re[rows/2][columns/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>getFft2</code>. To get back the original
     * data, use <code>ifft2</code>.
     * 
     * @throws IllegalArgumentException
     *             if the row size or the column size of this matrix is not a
     *             power of 2 number.
     * 
     */
    public void fft2() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public FloatMatrix2D forEachNonZero(final cern.colt.function.tfloat.IntIntFloatFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
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
            ConcurrencyUtils.waitForCompletion(futures);
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

    /**
     * Returns a new matrix that has the same elements as this matrix, but they
     * are addressed internally in column major. This method creates a new
     * object (not a view), so changes in the returned matrix are NOT reflected
     * in this matrix.
     * 
     * @return this matrix with elements addressed internally in column major
     */
    public DenseColumnFloatMatrix2D getColumnMajor() {
        DenseColumnFloatMatrix2D R = new DenseColumnFloatMatrix2D(rows, columns);
        final int zeroR = (int) R.index(0, 0);
        final int rowStrideR = R.rowStride();
        final int columnStrideR = R.columnStride();
        final float[] elementsR = R.elements();
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == (nthreads - 1)) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + (firstRow - 1) * rowStride;
                        int idxR = zeroR + (firstRow - 1) * rowStrideR;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, j = idxR, c = 0; r < columns; c++) {
                                elementsR[j] = elements[i];
                                i += rowStride;
                                j += rowStrideR;
                            }
                            idx += columnStride;
                            idxR += columnStrideR;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxR = zeroR;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, j = idxR, c = 0; r < columns; c++) {
                    elementsR[j] = elements[i];
                    i += rowStride;
                    j += rowStrideR;
                }
                idx += columnStride;
                idxR += columnStrideR;
            }
        }
        return R;
    }

    /**
     * Returns new complex matrix which is the 2D discrete Fourier transform
     * (DFT) of this matrix.
     * 
     * @return the 2D discrete Fourier transform (DFT) of this matrix.
     * 
     */
    public DenseFComplexMatrix2D getFft2() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        final float[] elementsA;
        if (isNoView == true) {
            elementsA = elements;
        } else {
            elementsA = (float[]) this.copy().elements();
        }
        DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        final float[] elementsC = (C).elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            System.arraycopy(elementsA, r * columns, elementsC, r * columns, columns);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                System.arraycopy(elementsA, r * columns, elementsC, r * columns, columns);
            }
        }
        fft2.realForwardFull(elementsC);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the discrete Fourier transform (DFT)
     * of each column of this matrix.
     * 
     * @return the discrete Fourier transform (DFT) of each column of this
     *         matrix.
     */
    public DenseFComplexMatrix2D getFftColumns() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        final DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            C.viewColumn(c).assign(((DenseFloatMatrix1D) viewColumn(c)).getFft());
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                C.viewColumn(c).assign(((DenseFloatMatrix1D) viewColumn(c)).getFft());
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the discrete Fourier transform (DFT)
     * of each row of this matrix.
     * 
     * @return the discrete Fourier transform (DFT) of each row of this matrix.
     */
    public DenseFComplexMatrix2D getFftRows() {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        final DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            C.viewRow(r).assign(((DenseFloatMatrix1D) viewRow(r)).getFft());
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                C.viewRow(r).assign(((DenseFloatMatrix1D) viewRow(r)).getFft());
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the 2D inverse of the discrete
     * Fourier transform (IDFT) of this matrix.
     * 
     * @return the 2D inverse of the discrete Fourier transform (IDFT) of this
     *         matrix.
     */
    public DenseFComplexMatrix2D getIfft2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        final float[] elementsC = (C).elements();
        final float[] elementsA;
        if (isNoView == true) {
            elementsA = elements;
        } else {
            elementsA = (float[]) this.copy().elements();
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow;
                if (j == nthreads - 1) {
                    lastRow = rows;
                } else {
                    lastRow = firstRow + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            System.arraycopy(elementsA, r * columns, elementsC, r * columns, columns);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = 0; r < rows; r++) {
                System.arraycopy(elementsA, r * columns, elementsC, r * columns, columns);
            }
        }
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.realInverseFull(elementsC, scale);
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the inverse of the discrete Fourier
     * transform (IDFT) of each column of this matrix.
     * 
     * @return the inverse of the discrete Fourier transform (IDFT) of each
     *         column of this matrix.
     */
    public DenseFComplexMatrix2D getIfftColumns(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        final DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            C.viewColumn(c).assign(((DenseFloatMatrix1D) viewColumn(c)).getIfft(scale));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                C.viewColumn(c).assign(((DenseFloatMatrix1D) viewColumn(c)).getIfft(scale));
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    /**
     * Returns new complex matrix which is the inverse of the discrete Fourier
     * transform (IDFT) of each row of this matrix.
     * 
     * @return the inverse of the discrete Fourier transform (IDFT) of each row
     *         of this matrix.
     */
    public DenseFComplexMatrix2D getIfftRows(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        final DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            C.viewRow(r).assign(((DenseFloatMatrix1D) viewRow(r)).getIfft(scale));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                C.viewRow(r).assign(((DenseFloatMatrix1D) viewRow(r)).getIfft(scale));
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    public float[] getMaxLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = (int) index(0, 0);
        float maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            float[][] results = new float[nthreads][2];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float maxValue = elements[zero + firstRow * rowStride];
                        int rowLocation = firstRow;
                        int colLocation = 0;
                        float elem;
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
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
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (float[]) futures[j].get();
                }
                maxValue = results[0][0];
                rowLocation = (int) results[0][1];
                columnLocation = (int) results[0][2];
                for (int j = 1; j < nthreads; j++) {
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
        final int zero = (int) index(0, 0);
        float minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            float[][] results = new float[nthreads][2];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        int rowLocation = firstRow;
                        int columnLocation = 0;
                        float minValue = elements[zero + firstRow * rowStride];
                        float elem;
                        int d = 1;
                        for (int r = firstRow; r < lastRow; r++) {
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
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (float[]) futures[j].get();
                }
                minValue = results[0][0];
                rowLocation = (int) results[0][1];
                columnLocation = (int) results[0][2];
                for (int j = 1; j < nthreads; j++) {
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

    public void getNegativeValues(final IntArrayList rowList, final IntArrayList columnList,
            final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
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
        int idx = (int) index(0, 0);
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

    public void getPositiveValues(final IntArrayList rowList, final IntArrayList columnList,
            final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
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

    /**
     * Computes the 2D inverse of the discrete cosine transform (DCT-III) of
     * this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idct2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete cosine transform (DCT-III) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idctColumns(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFloatMatrix1D) viewColumn(c)).idct(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFloatMatrix1D) viewColumn(c)).idct(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete cosine transform (DCT-III) of each
     * row of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idctRows(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFloatMatrix1D) viewRow(r)).idct(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFloatMatrix1D) viewRow(r)).idct(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete Hartley transform (IDHT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idht2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete Hartley transform (IDHT) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idhtColumns(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFloatMatrix1D) viewColumn(c)).idht(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFloatMatrix1D) viewColumn(c)).idht(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete Hartley transform (IDHT) of each row
     * of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idhtRows(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFloatMatrix1D) viewRow(r)).idht(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFloatMatrix1D) viewRow(r)).idht(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete sine transform (DST-III) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idst2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete sine transform (DST-III) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idstColumns(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            ((DenseFloatMatrix1D) viewColumn(c)).idst(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = 0; c < columns; c++) {
                ((DenseFloatMatrix1D) viewColumn(c)).idst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the inverse of the discrete sine transform (DST-III) of each row
     * of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idstRows(final boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            ((DenseFloatMatrix1D) viewRow(r)).idst(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = 0; r < rows; r++) {
                ((DenseFloatMatrix1D) viewRow(r)).idst(scale);
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    /**
     * Computes the 2D inverse of the discrete Fourier transform (IDFT) of this
     * matrix. The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * this[k1][2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2], 
     * this[k1][2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2], 
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2, 
     * this[0][2*k2] = Re[0][k2] = Re[0][columns-k2], 
     * this[0][2*k2+1] = Im[0][k2] = -Im[0][columns-k2], 
     *       0&lt;k2&lt;columns/2, 
     * this[k1][0] = Re[k1][0] = Re[rows-k1][0], 
     * this[k1][1] = Im[k1][0] = -Im[rows-k1][0], 
     * this[rows-k1][1] = Re[k1][columns/2] = Re[rows-k1][columns/2], 
     * this[rows-k1][0] = -Im[k1][columns/2] = Im[rows-k1][columns/2], 
     *       0&lt;k1&lt;rows/2, 
     * this[0][0] = Re[0][0], 
     * this[0][1] = Re[0][columns/2], 
     * this[rows/2][0] = Re[rows/2][0], 
     * this[rows/2][1] = Re[rows/2][columns/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>getIfft2</code>.
     * 
     * @throws IllegalArgumentException
     *             if the row size or the column size of this matrix is not a
     *             power of 2 number.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void ifft2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
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
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public long index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    public FloatMatrix2D like(int rows, int columns) {
        return new DenseFloatMatrix2D(rows, columns);
    }

    public FloatMatrix1D like1D(int size) {
        return new DenseFloatMatrix1D(size);
    }

    public void setQuick(int row, int column, float value) {
        elements[rowZero + row * rowStride + columnZero + column * columnStride] = value;
    }

    public float[][] toArray() {
        final float[][] values = new float[rows][columns];
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
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
            ConcurrencyUtils.waitForCompletion(futures);
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
        final DenseFloatMatrix1D v = new DenseFloatMatrix1D((int) size());
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) v.index(0);
        final int strideOther = v.stride();
        final float[] elementsOther = v.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = 0;
                        int idxOther = zeroOther + firstColumn * rows;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            idx = zero + c * columnStride;
                            for (int r = 0; r < rows; r++) {
                                elementsOther[idxOther] = elements[idx];
                                idx += rowStride;
                                idxOther += strideOther;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int c = 0; c < columns; c++) {
                idx = zero + c * columnStride;
                for (int r = 0; r < rows; r++) {
                    elementsOther[idxOther] = elements[idx];
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

        int A_index = (int) index(1, 1);
        int B_index = (int) BB.index(1, 1);
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

    public FloatMatrix1D zMult(final FloatMatrix1D y, FloatMatrix1D z, final float alpha, final float beta,
            final boolean transposeA) {
        if (transposeA)
            return viewDice().zMult(y, z, alpha, beta, false);
        if (z == null) {
            z = new DenseFloatMatrix1D(rows);
        }
        if (!(y instanceof DenseFloatMatrix1D && z instanceof DenseFloatMatrix1D))
            return super.zMult(y, z, alpha, beta, transposeA);

        if (columns != y.size() || rows > z.size())
            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort()
                    + ", " + z.toStringShort());

        final float[] elemsY = (float[]) y.elements();
        final float[] elemsZ = (float[]) z.elements();
        if (elements == null || elemsY == null || elemsZ == null)
            throw new InternalError();
        final int strideY = y.stride();
        final int strideZ = z.stride();
        final int zero = (int) index(0, 0);
        final int zeroY = (int) y.index(0);
        final int zeroZ = (int) z.index(0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idxZero = zero + firstRow * rowStride;
                        int idxZeroZ = zeroZ + firstRow * strideZ;
                        for (int r = firstRow; r < lastRow; r++) {
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
            ConcurrencyUtils.waitForCompletion(futures);
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
        return z;
    }

    public FloatMatrix2D zMult(final FloatMatrix2D B, FloatMatrix2D C, final float alpha, final float beta,
            final boolean transposeA, final boolean transposeB) {
        final int rowsA = rows;
        final int columnsA = columns;
        final int rowsB = B.rows();
        final int columnsB = B.columns();
        final int rowsC = transposeA ? columnsA : rowsA;
        final int columnsC = transposeB ? rowsB : columnsB;

        if (C == null) {
            C = new DenseFloatMatrix2D(rowsC, columnsC);
        }

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
        if (B instanceof SparseFloatMatrix2D || B instanceof SparseRCFloatMatrix2D) {
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

        if (B.rows() != columnsA)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + this.toStringShort() + ", "
                    + B.toStringShort());
        if (C.rows() != rowsA || C.columns() != columnsB)
            throw new IllegalArgumentException("Incompatibe result matrix: " + this.toStringShort() + ", "
                    + B.toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        long flops = 2L * rowsA * columnsA * columnsB;
        int noOfTasks = (int) Math.min(flops / 30000, ConcurrencyUtils.getNumberOfThreads()); // each
        /* thread should process at least 30000 flops */
        boolean splitB = (columnsB >= noOfTasks);
        int width = splitB ? columnsB : rowsA;
        noOfTasks = Math.min(width, noOfTasks);

        if (noOfTasks < 2) { //parallelization doesn't pay off (too much start up overhead)
            return this.zMultSequential(B, C, alpha, beta, transposeA, transposeB);
        }

        // set up concurrent tasks
        int span = width / noOfTasks;
        final Future<?>[] subTasks = new Future[noOfTasks];
        for (int i = 0; i < noOfTasks; i++) {
            final int offset = i * span;
            if (i == noOfTasks - 1)
                span = width - span * i; // last span may be a bit larger

            final FloatMatrix2D AA, BB, CC;
            if (splitB) {
                // split B along columns into blocks
                AA = this;
                BB = B.viewPart(0, offset, columnsA, span);
                CC = C.viewPart(0, offset, rowsA, span);
            } else {
                // split A along rows into blocks
                AA = this.viewPart(offset, 0, span, columnsA);
                BB = B;
                CC = C.viewPart(offset, 0, span, columnsB);
            }

            subTasks[i] = ConcurrencyUtils.submit(new Runnable() {
                public void run() {
                    ((DenseFloatMatrix2D) AA).zMultSequential(BB, CC, alpha, beta, transposeA, transposeB);
                }
            });
        }

        ConcurrencyUtils.waitForCompletion(subTasks);
        return C;
    }

    public float zSum() {
        float sum = 0;
        if (elements == null)
            throw new InternalError();
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float sum = 0;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
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
                for (int j = 0; j < nthreads; j++) {
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

    private FloatMatrix2D zMultSequential(FloatMatrix2D B, FloatMatrix2D C, float alpha, float beta,
            boolean transposeA, boolean transposeB) {
        if (transposeA)
            return viewDice().zMult(B, C, alpha, beta, false, transposeB);
        if (B instanceof SparseFloatMatrix2D || B instanceof SparseRCFloatMatrix2D
                || B instanceof SparseCCFloatMatrix2D) {
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

        int rowsA = rows;
        int columnsA = columns;
        int p = B.columns();
        if (C == null) {
            C = new DenseFloatMatrix2D(rowsA, p);
        }
        if (!(B instanceof DenseFloatMatrix2D) || !(C instanceof DenseFloatMatrix2D))
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        if (B.rows() != columnsA)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + B.toStringShort());
        if (C.rows() != rowsA || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", "
                    + B.toStringShort() + ", " + C.toStringShort());
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
        int m_optimal = (BLOCK_SIZE - columnsA) / (columnsA + 1);
        if (m_optimal <= 0)
            m_optimal = 1;
        int blocks = rowsA / m_optimal;
        int rr = 0;
        if (rowsA % m_optimal != 0)
            blocks++;
        for (; --blocks >= 0;) {
            int jB = (int) BB.index(0, 0);
            int indexA = (int) index(rr, 0);
            int jC = (int) CC.index(rr, 0);
            rr += m_optimal;
            if (blocks == 0)
                m_optimal += rowsA - rr;

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

                    for (int k = columnsA % 4; --k >= 0;) {
                        s += AElems[kA += cA] * BElems[kB += rB];
                    }
                    for (int k = columnsA / 4; --k >= 0;) {
                        s += AElems[kA += cA] * BElems[kB += rB] + AElems[kA += cA] * BElems[kB += rB]
                                + AElems[kA += cA] * BElems[kB += rB] + AElems[kA += cA] * BElems[kB += rB];
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
        return new DenseFloatMatrix1D(size, this.elements, zero, stride, true);
    }

    protected FloatMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseFloatMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }
}
