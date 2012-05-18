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

import org.netlib.blas.BLAS;

import cern.colt.function.tfloat.FloatFloatFunction;
import cern.colt.function.tfloat.FloatFunction;
import cern.colt.function.tfloat.FloatProcedure;
import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.Transpose;
import cern.colt.matrix.io.MatrixInfo;
import cern.colt.matrix.io.MatrixSize;
import cern.colt.matrix.io.MatrixVectorReader;
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
 * Internally holds one single contigous one-dimensional array, addressed in
 * column major. Note that this implementation is not synchronized.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 * <p>
 * Cells are internally addressed in column-major. Applications demanding utmost
 * speed can exploit this fact. Setting/getting values in a loop
 * column-by-column is quicker than row-by-row. Thus
 * 
 * <pre>
 * for (int column = 0; column &lt; columns; column++) {
 *     for (int row = 0; row &lt; rows; row++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * 
 * </pre>
 * 
 * is quicker than
 * 
 * <pre>
 * for (int row = 0; row &lt; rows; row++) {
 *     for (int column = 0; column &lt; columns; column++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * 
 * </pre>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseColumnFloatMatrix2D extends FloatMatrix2D {
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
    public DenseColumnFloatMatrix2D(float[][] values) {
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
    public DenseColumnFloatMatrix2D(int rows, int columns) {
        setUp(rows, columns, 0, 0, 1, rows);
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
    public DenseColumnFloatMatrix2D(int rows, int columns, float[] elements, int rowZero, int columnZero,
            int rowStride, int columnStride, boolean isView) {
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
    public DenseColumnFloatMatrix2D(MatrixVectorReader reader) throws IOException {
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

    public float aggregate(final FloatFloatFunction aggr, final FloatFunction f) {
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride]);
                        int d = 1;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            int cidx = zero + c * columnStride;
                            for (int r = rows - d; --r >= 0;) {
                                a = aggr.apply(a, f.apply(elements[r * rowStride + cidx]));
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
            for (int c = columns; --c >= 0;) {
                int cidx = zero + c * columnStride;
                for (int r = rows - d; --r >= 0;) {
                    a = aggr.apply(a, f.apply(elements[r * rowStride + cidx]));
                }
                d = 0;
            }
        }
        return a;
    }

    public float aggregate(final FloatFloatFunction aggr, final FloatFunction f, final FloatProcedure cond) {
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float elem = elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride];
                        float a = 0;
                        if (cond.apply(elem) == true) {
                            a = f.apply(elem);
                        }
                        int d = 1;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            int cidx = zero + c * columnStride;
                            for (int r = rows - d; --r >= 0;) {
                                elem = elements[r * rowStride + cidx];
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
            float elem = elements[zero + (rows - 1) * rowStride + (columns - 1) * columnStride];
            if (cond.apply(elem) == true) {
                a = f.apply(elem);
            }
            int d = 1;
            for (int c = columns; --c >= 0;) {
                int cidx = zero + c * columnStride;
                for (int r = rows - d; --r >= 0;) {
                    elem = elements[r * rowStride + cidx];
                    if (cond.apply(elem) == true) {
                        a = aggr.apply(a, f.apply(elem));
                    }
                }
                d = 0;
            }
        }
        return a;
    }

    public float aggregate(final FloatFloatFunction aggr, final FloatFunction f, final IntArrayList rowList,
            final IntArrayList columnList) {
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = size - j * k;
                final int lastIdx = (j == (nthreads - 1)) ? 0 : firstIdx - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + rowElements[firstIdx - 1] * rowStride
                                + columnElements[firstIdx - 1] * columnStride]);
                        for (int i = firstIdx - 1; --i >= lastIdx;) {
                            a = aggr.apply(a, f.apply(elements[zero + rowElements[i] * rowStride + columnElements[i]
                                    * columnStride]));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero + rowElements[size - 1] * rowStride + columnElements[size - 1] * columnStride]);
            for (int i = size - 1; --i >= 0;) {
                a = aggr.apply(a, f
                        .apply(elements[zero + rowElements[i] * rowStride + columnElements[i] * columnStride]));
            }
        }
        return a;
    }

    public float aggregate(final FloatMatrix2D other, final FloatFloatFunction aggr, final FloatFloatFunction f) {
        if (!(other instanceof DenseColumnFloatMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            return Float.NaN;
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int columnStrideOther = other.columnStride();
        final float[] otherElements = (float[]) other.elements();
        float a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float a = f.apply(elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride],
                                otherElements[zeroOther + (rows - 1) * rowStrideOther + (firstColumn - 1)
                                        * columnStrideOther]);
                        int d = 1;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            int cidx = zero + c * columnStride;
                            int cidxOther = zeroOther + c * columnStrideOther;
                            for (int r = rows - d; --r >= 0;) {
                                a = aggr.apply(a, f.apply(elements[r * rowStride + cidx], otherElements[r
                                        * rowStrideOther + cidxOther]));
                            }
                            d = 0;
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            int d = 1;
            a = f.apply(elements[zero + (rows - 1) * rowStride + (columns - 1) * columnStride], otherElements[zeroOther
                    + (rows - 1) * rowStrideOther + (columns - 1) * columnStrideOther]);
            for (int c = columns; --c >= 0;) {
                int cidx = zero + c * columnStride;
                int cidxOther = zeroOther + c * columnStrideOther;
                for (int r = rows - d; --r >= 0;) {
                    a = aggr.apply(a, f.apply(elements[r * rowStride + cidx], otherElements[r * rowStrideOther
                            + cidxOther]));
                }
                d = 0;
            }
        }
        return a;
    }

    public FloatMatrix2D assign(final FloatFunction function) {
        if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] = mult*x[i]
            float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
            if (multiplicator == 1)
                return this;
            if (multiplicator == 0)
                return assign(0);
        }
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        // specialization for speed
                        if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] = mult*x[i]
                            float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                            for (int c = firstColumn; --c >= lastColumn;) {
                                for (int i = idx, r = rows; --r >= 0;) {
                                    elements[i] *= multiplicator;
                                    i -= rowStride;
                                }
                                idx -= columnStride;
                            }
                        } else { // the general case x[i] = f(x[i])                            
                            for (int c = firstColumn; --c >= lastColumn;) {
                                for (int i = idx, r = rows; --r >= 0;) {
                                    elements[i] = function.apply(elements[i]);
                                    i -= rowStride;
                                }
                                idx -= columnStride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] = mult*x[i]
                float multiplicator = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
                for (int c = columns; --c >= 0;) {
                    for (int i = idx, r = rows; --r >= 0;) {
                        elements[i] *= multiplicator;
                        i -= rowStride;
                    }
                    idx -= columnStride;
                }
            } else { // the general case x[i] = f(x[i])
                for (int c = columns; --c >= 0;) {
                    for (int i = idx, r = rows; --r >= 0;) {
                        elements[i] = function.apply(elements[i]);
                        i -= rowStride;
                    }
                    idx -= columnStride;
                }
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final FloatProcedure cond, final FloatFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        float elem;
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, r = rows; --r >= 0;) {
                                elem = elements[i];
                                if (cond.apply(elem) == true) {
                                    elements[i] = function.apply(elem);
                                }
                                i -= rowStride;
                            }
                            idx -= columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float elem;
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, r = rows; --r >= 0;) {
                    elem = elements[i];
                    if (cond.apply(elem) == true) {
                        elements[i] = function.apply(elem);
                    }
                    i -= rowStride;
                }
                idx -= columnStride;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final FloatProcedure cond, final float value) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        float elem;
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, r = rows; --r >= 0;) {
                                elem = elements[i];
                                if (cond.apply(elem) == true) {
                                    elements[i] = value;
                                }
                                i -= rowStride;
                            }
                            idx -= columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            float elem;
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, r = rows; --r >= 0;) {
                    elem = elements[i];
                    if (cond.apply(elem) == true) {
                        elements[i] = value;
                    }
                    i -= rowStride;
                }
                idx -= columnStride;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final float value) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, r = rows; --r >= 0;) {
                                elements[i] = value;
                                i -= rowStride;
                            }
                            idx -= columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, r = rows; --r >= 0;) {
                    elements[i] = value;
                    i -= rowStride;
                }
                idx -= columnStride;
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
                nthreads = Math.min(nthreads, columns);
                Future<?>[] futures = new Future[nthreads];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = columns - j * k;
                    final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                            int idxOther = (rows - 1) + (firstColumn - 1) * rows;
                            for (int c = firstColumn; --c >= lastColumn;) {
                                for (int i = idx, r = rows; --r >= 0;) {
                                    elements[i] = values[idxOther--];
                                    i -= rowStride;
                                }
                                idx -= columnStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
                int idxOther = values.length - 1;
                for (int c = columns; --c >= 0;) {
                    for (int i = idx, r = rows; --r >= 0;) {
                        elements[i] = values[idxOther--];
                        i -= rowStride;
                    }
                    idx -= columnStride;
                }
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final float[][] values) {
        if (values.length != rows)
            throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "columns()="
                    + rows());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + (firstRow - 1) * rowStride + (columns - 1) * columnStride;
                        for (int r = firstRow; --r >= lastRow;) {
                            float[] currentRow = values[r];
                            if (currentRow.length != columns)
                                throw new IllegalArgumentException(
                                        "Must have same number of columns in every row: column=" + currentRow.length
                                                + "columns()=" + columns());
                            for (int i = idx, c = columns; --c >= 0;) {
                                elements[i] = currentRow[c];
                                i -= columnStride;
                            }
                            idx -= rowStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int r = rows; --r >= 0;) {
                float[] currentRow = values[r];
                if (currentRow.length != columns)
                    throw new IllegalArgumentException("Must have same number of columns in every row: column="
                            + currentRow.length + "columns()=" + columns());
                for (int i = idx, c = columns; --c >= 0;) {
                    elements[i] = currentRow[c];
                    i -= columnStride;
                }
                idx -= rowStride;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final FloatMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof DenseColumnFloatMatrix2D)) {
            super.assign(source);
            return this;
        }
        DenseColumnFloatMatrix2D other = (DenseColumnFloatMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, elements, 0, elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            FloatMatrix2D c = other.copy();
            if (!(c instanceof DenseColumnFloatMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseColumnFloatMatrix2D) c;
        }

        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        final float[] otherElements = other.elements;
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        int idxOther = zeroOther + (rows - 1) * rowStrideOther + (firstColumn - 1) * columnStrideOther;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                elements[i] = otherElements[j];
                                i -= rowStride;
                                j -= rowStrideOther;
                            }
                            idx -= columnStride;
                            idxOther -= columnStrideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            int idxOther = zeroOther + (rows - 1) * rowStrideOther + (columns - 1) * columnStrideOther;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                    elements[i] = otherElements[j];
                    i -= rowStride;
                    j -= rowStrideOther;
                }
                idx -= columnStride;
                idxOther -= columnStrideOther;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final FloatMatrix2D y, final FloatFloatFunction function) {
        if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) {
            float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
            if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                return this;
            }
        }
        if (function instanceof cern.jet.math.tfloat.FloatPlusMultFirst) {
            float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultFirst) function).multiplicator;
            if (multiplicator == 0) { // x[i] = 0*x[i] + y[i]
                return assign(y);
            }
        }
        if (!(y instanceof DenseColumnFloatMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        DenseColumnFloatMatrix2D other = (DenseColumnFloatMatrix2D) y;
        checkShape(y);
        final float[] otherElements = other.elements;
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        int idxOther = zeroOther + (rows - 1) * rowStrideOther + (firstColumn - 1) * columnStrideOther;
                        if (function == cern.jet.math.tfloat.FloatFunctions.mult) {
                            // x[i] = x[i]*y[i]                            
                            for (int c = firstColumn; --c >= lastColumn;) {
                                for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                    elements[i] *= otherElements[j];
                                    i -= rowStride;
                                    j -= rowStrideOther;
                                }
                                idx -= columnStride;
                                idxOther -= columnStrideOther;
                            }
                        } else if (function == cern.jet.math.tfloat.FloatFunctions.div) {
                            // x[i] = x[i] / y[i]
                            for (int c = firstColumn; --c >= lastColumn;) {
                                for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                    elements[i] /= otherElements[j];
                                    i -= rowStride;
                                    j -= rowStrideOther;
                                }
                                idx -= columnStride;
                                idxOther -= columnStrideOther;
                            }
                        } else if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) {
                            float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
                            if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                for (int c = firstColumn; --c >= lastColumn;) {
                                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                        elements[i] += otherElements[j];
                                        i -= rowStride;
                                        j -= rowStrideOther;
                                    }
                                    idx -= columnStride;
                                    idxOther -= columnStrideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = x[i] - y[i]
                                for (int c = firstColumn; --c >= lastColumn;) {
                                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                        elements[i] -= otherElements[j];
                                        i -= rowStride;
                                        j -= rowStrideOther;
                                    }
                                    idx -= columnStride;
                                    idxOther -= columnStrideOther;
                                }
                            } else { // the general case
                                // x[i] = x[i] + mult*y[i]
                                for (int c = firstColumn; --c >= lastColumn;) {
                                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                        elements[i] += multiplicator * otherElements[j];
                                        i -= rowStride;
                                        j -= rowStrideOther;
                                    }
                                    idx -= columnStride;
                                    idxOther -= columnStrideOther;
                                }
                            }
                        } else if (function instanceof cern.jet.math.tfloat.FloatPlusMultFirst) {
                            float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultFirst) function).multiplicator;
                            if (multiplicator == 1) {
                                // x[i] = x[i] + y[i]
                                for (int c = firstColumn; --c >= lastColumn;) {
                                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                        elements[i] += otherElements[j];
                                        i -= rowStride;
                                        j -= rowStrideOther;
                                    }
                                    idx -= columnStride;
                                    idxOther -= columnStrideOther;
                                }
                            } else if (multiplicator == -1) {
                                // x[i] = -x[i] + y[i]
                                for (int c = firstColumn; --c >= lastColumn;) {
                                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                        elements[i] = otherElements[j] - elements[i];
                                        i -= rowStride;
                                        j -= rowStrideOther;
                                    }
                                    idx -= columnStride;
                                    idxOther -= columnStrideOther;
                                }
                            } else { // the general case
                                // x[i] = mult*x[i] + y[i]
                                for (int c = firstColumn; --c >= lastColumn;) {
                                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                        elements[i] = multiplicator * elements[i] + otherElements[j];
                                        i -= rowStride;
                                        j -= rowStrideOther;
                                    }
                                    idx -= columnStride;
                                    idxOther -= columnStrideOther;
                                }
                            }
                        } else { // the general case x[i] = f(x[i],y[i])
                            for (int c = firstColumn; --c >= lastColumn;) {
                                for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                    elements[i] = function.apply(elements[i], otherElements[j]);
                                    i -= rowStride;
                                    j -= rowStrideOther;
                                }
                                idx -= columnStride;
                                idxOther -= columnStrideOther;
                            }
                        }

                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            int idxOther = zeroOther + (rows - 1) * rowStrideOther + (columns - 1) * columnStrideOther;
            if (function == cern.jet.math.tfloat.FloatFunctions.mult) {
                // x[i] = x[i]*y[i]                            
                for (int c = columns; --c >= 0;) {
                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                        elements[i] *= otherElements[j];
                        i -= rowStride;
                        j -= rowStrideOther;
                    }
                    idx -= columnStride;
                    idxOther -= columnStrideOther;
                }
            } else if (function == cern.jet.math.tfloat.FloatFunctions.div) {
                // x[i] = x[i] / y[i]
                for (int c = columns; --c >= 0;) {
                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                        elements[i] /= otherElements[j];
                        i -= rowStride;
                        j -= rowStrideOther;
                    }
                    idx -= columnStride;
                    idxOther -= columnStrideOther;
                }
            } else if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) {
                float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
                if (multiplicator == 1) {
                    // x[i] = x[i] + y[i]
                    for (int c = columns; --c >= 0;) {
                        for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                            elements[i] += otherElements[j];
                            i -= rowStride;
                            j -= rowStrideOther;
                        }
                        idx -= columnStride;
                        idxOther -= columnStrideOther;
                    }
                } else if (multiplicator == -1) {
                    // x[i] = x[i] - y[i]
                    for (int c = columns; --c >= 0;) {
                        for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                            elements[i] -= otherElements[j];
                            i -= rowStride;
                            j -= rowStrideOther;
                        }
                        idx -= columnStride;
                        idxOther -= columnStrideOther;
                    }
                } else { // the general case
                    // x[i] = x[i] + mult*y[i]
                    for (int c = columns; --c >= 0;) {
                        for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                            elements[i] += multiplicator * otherElements[j];
                            i -= rowStride;
                            j -= rowStrideOther;
                        }
                        idx -= columnStride;
                        idxOther -= columnStrideOther;
                    }
                }
            } else if (function instanceof cern.jet.math.tfloat.FloatPlusMultFirst) {
                float multiplicator = ((cern.jet.math.tfloat.FloatPlusMultFirst) function).multiplicator;
                if (multiplicator == 1) {
                    // x[i] = x[i] + y[i]
                    for (int c = columns; --c >= 0;) {
                        for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                            elements[i] += otherElements[j];
                            i -= rowStride;
                            j -= rowStrideOther;
                        }
                        idx -= columnStride;
                        idxOther -= columnStrideOther;
                    }
                } else if (multiplicator == -1) {
                    // x[i] = -x[i] + y[i]
                    for (int c = columns; --c >= 0;) {
                        for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                            elements[i] = otherElements[j] - elements[i];
                            i -= rowStride;
                            j -= rowStrideOther;
                        }
                        idx -= columnStride;
                        idxOther -= columnStrideOther;
                    }
                } else { // the general case
                    // x[i] = mult*x[i] + y[i]
                    for (int c = columns; --c >= 0;) {
                        for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                            elements[i] = multiplicator * elements[i] + otherElements[j];
                            i -= rowStride;
                            j -= rowStrideOther;
                        }
                        idx -= columnStride;
                        idxOther -= columnStrideOther;
                    }
                }
            } else { // the general case x[i] = f(x[i],y[i])
                for (int c = columns; --c >= 0;) {
                    for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                        elements[i] = function.apply(elements[i], otherElements[j]);
                        i -= rowStride;
                        j -= rowStrideOther;
                    }
                    idx -= columnStride;
                    idxOther -= columnStrideOther;
                }
            }
        }
        return this;
    }

    public FloatMatrix2D assign(final FloatMatrix2D y, final FloatFloatFunction function, IntArrayList rowList,
            IntArrayList columnList) {
        checkShape(y);
        if (!(y instanceof DenseColumnFloatMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        DenseColumnFloatMatrix2D other = (DenseColumnFloatMatrix2D) y;
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final float[] otherElements = other.elements();
        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride();
        final int rowStrideOther = other.rowStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = size - j * k;
                final int lastIdx = (j == (nthreads - 1)) ? 0 : firstIdx - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx;
                        int idxOther;
                        for (int i = firstIdx; --i >= lastIdx;) {
                            idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                            idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i]
                                    * columnStrideOther;
                            elements[idx] = function.apply(elements[idx], otherElements[idxOther]);
                        }
                    }

                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx;
            int idxOther;
            for (int i = size; --i >= 0;) {
                idx = zero + rowElements[i] * rowStride + columnElements[i] * columnStride;
                idxOther = zeroOther + rowElements[i] * rowStrideOther + columnElements[i] * columnStrideOther;
                elements[idx] = function.apply(elements[idx], otherElements[idxOther]);
            }
        }
        return this;
    }

    public int cardinality() {
        int cardinality = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, r = rows; --r >= 0;) {
                                if (elements[i] != 0)
                                    cardinality++;
                                i -= rowStride;
                            }
                            idx -= columnStride;
                        }
                        return cardinality;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Integer) futures[j].get();
                }
                cardinality = results[0];
                for (int j = 1; j < nthreads; j++) {
                    cardinality += results[j];
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, r = rows; --r >= 0;) {
                    if (elements[i] != 0)
                        cardinality++;
                    i -= rowStride;
                }
                idx -= columnStride;
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dct2 == null) {
            dct2 = new FloatDCT_2D(rows, columns);
        }
        dct2.forward((float[]) transpose.elements(), scale);
        this.assign(transpose.viewDice().copy());
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
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            ((DenseFloatMatrix1D) viewColumn(c)).dct(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dht2 == null) {
            dht2 = new FloatDHT_2D(rows, columns);
        }
        dht2.forward((float[]) transpose.elements());
        this.assign(transpose.viewDice().copy());
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
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            ((DenseFloatMatrix1D) viewColumn(c)).dht();
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dst2 == null) {
            dst2 = new FloatDST_2D(rows, columns);
        }
        dst2.forward((float[]) transpose.elements(), scale);
        this.assign(transpose.viewDice().copy());
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
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            ((DenseFloatMatrix1D) viewColumn(c)).dst(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
                ((DenseFloatMatrix1D) viewColumn(c)).dst(scale);
            }
        }
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.realForward((float[]) transpose.elements());
        this.assign(transpose.viewDice().copy());
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public FloatMatrix2D forEachNonZero(final cern.colt.function.tfloat.IntIntFloatFunction function) {
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, r = rows; --r >= 0;) {
                                float value = elements[i];
                                if (value != 0) {
                                    elements[i] = function.apply(r, c, value);
                                }
                                i -= rowStride;
                            }
                            idx -= columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, r = rows; --r >= 0;) {
                    float value = elements[i];
                    if (value != 0) {
                        elements[i] = function.apply(r, c, value);
                    }
                    i -= rowStride;
                }
                idx -= columnStride;
            }
        }
        return this;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but they
     * are addressed internally in row major. This method creates a new object
     * (not a view), so changes in the returned matrix are NOT reflected in this
     * matrix.
     * 
     * @return this matrix with elements addressed internally in row major
     */
    public DenseFloatMatrix2D getRowMajor() {
        DenseFloatMatrix2D R = new DenseFloatMatrix2D(rows, columns);
        final int zeroR = (int) R.index(0, 0);
        final int rowStrideR = R.rowStride();
        final int columnStrideR = R.columnStride();
        final float[] elementsR = R.elements();
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        int idxR = zeroR + (rows - 1) * rowStrideR + (firstColumn - 1) * columnStrideR;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, j = idxR, r = rows; --r >= 0;) {
                                elementsR[j] = elements[i];
                                i -= rowStride;
                                j -= rowStrideR;
                            }
                            idx -= columnStride;
                            idxR -= columnStrideR;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            int idxR = zeroR + (rows - 1) * rowStrideR + (columns - 1) * columnStrideR;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, j = idxR, r = rows; --r >= 0;) {
                    elementsR[j] = elements[i];
                    i -= rowStride;
                    j -= rowStrideR;
                }
                idx -= columnStride;
                idxR -= columnStrideR;
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
        DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        final float[] elementsC = (C).elements();
        final int zero = (int) index(0, 0);
        final int zeroC = (int) C.index(0, 0);
        final int rowStrideC = C.rowStride() / 2;
        final int columnStrideC = 1;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        int idxOther = zeroC + (rows - 1) * rowStrideC + (firstColumn - 1) * columnStrideC;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                elementsC[j] = elements[i];
                                i -= rowStride;
                                j -= rowStrideC;
                            }
                            idx -= columnStride;
                            idxOther -= columnStrideC;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            int idxOther = zeroC + (rows - 1) * rowStrideC + (columns - 1) * columnStrideC;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                    elementsC[j] = elements[i];
                    i -= rowStride;
                    j -= rowStrideC;
                }
                idx -= columnStride;
                idxOther -= columnStrideC;
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
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            C.viewColumn(c).assign(((DenseFloatMatrix1D) viewColumn(c)).getFft());
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
                            C.viewRow(r).assign(((DenseFloatMatrix1D) viewRow(r)).getFft());
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = rows; --r >= 0;) {
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
    public FComplexMatrix2D getIfft2(boolean scale) {
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        DenseFComplexMatrix2D C = new DenseFComplexMatrix2D(rows, columns);
        final float[] elementsC = (C).elements();
        final int zero = (int) index(0, 0);
        final int zeroC = (int) C.index(0, 0);
        final int rowStrideC = C.rowStride() / 2;
        final int columnStrideC = 1;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        int idxOther = zeroC + (rows - 1) * rowStrideC + (firstColumn - 1) * columnStrideC;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                                elementsC[j] = elements[i];
                                i -= rowStride;
                                j -= rowStrideC;
                            }
                            idx -= columnStride;
                            idxOther -= columnStrideC;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            int idxOther = zeroC + (rows - 1) * rowStrideC + (columns - 1) * columnStrideC;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, j = idxOther, r = rows; --r >= 0;) {
                    elementsC[j] = elements[i];
                    i -= rowStride;
                    j -= rowStrideC;
                }
                idx -= columnStride;
                idxOther -= columnStrideC;
            }
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
    public FComplexMatrix2D getIfftColumns(final boolean scale) {
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
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            C.viewColumn(c).assign(((DenseFloatMatrix1D) viewColumn(c)).getIfft(scale));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
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
    public FComplexMatrix2D getIfftRows(final boolean scale) {
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
                            C.viewRow(r).assign(((DenseFloatMatrix1D) viewRow(r)).getIfft(scale));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int r = rows; --r >= 0;) {
                C.viewRow(r).assign(((DenseFloatMatrix1D) viewRow(r)).getIfft(scale));
            }
        }
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
        return C;
    }

    public void getNegativeValues(final IntArrayList rowList, final IntArrayList columnList,
            final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int c = 0; c < columns; c++) {
            for (int i = idx, r = 0; r < rows; r++) {
                float value = elements[i];
                if (value < 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += rowStride;
            }
            idx += columnStride;
        }
    }

    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList, final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int c = 0; c < columns; c++) {
            for (int i = idx, r = 0; r < rows; r++) {
                float value = elements[i];
                if (value != 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += rowStride;
            }
            idx += columnStride;
        }
    }

    public void getPositiveValues(final IntArrayList rowList, final IntArrayList columnList,
            final FloatArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int c = 0; c < columns; c++) {
            for (int i = idx, r = 0; r < rows; r++) {
                float value = elements[i];
                if (value > 0) {
                    rowList.add(r);
                    columnList.add(c);
                    valueList.add(value);
                }
                i += rowStride;
            }
            idx += columnStride;
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dct2 == null) {
            dct2 = new FloatDCT_2D(rows, columns);
        }
        dct2.inverse((float[]) transpose.elements(), scale);
        this.assign(transpose.viewDice().copy());
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
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            ((DenseFloatMatrix1D) viewColumn(c)).idct(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
                ((DenseFloatMatrix1D) viewColumn(c)).idct(scale);
            }
        }
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dht2 == null) {
            dht2 = new FloatDHT_2D(rows, columns);
        }
        dht2.inverse((float[]) transpose.elements(), scale);
        this.assign(transpose.viewDice().copy());
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
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            ((DenseFloatMatrix1D) viewColumn(c)).idht(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (dst2 == null) {
            dst2 = new FloatDST_2D(rows, columns);
        }
        dst2.inverse((float[]) transpose.elements(), scale);
        this.assign(transpose.viewDice().copy());
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
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int c = firstColumn; --c >= lastColumn;) {
                            ((DenseFloatMatrix1D) viewColumn(c)).idst(scale);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN_FFT();
        } else {
            for (int c = columns; --c >= 0;) {
                ((DenseFloatMatrix1D) viewColumn(c)).idst(scale);
            }
        }
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
                final int firstRow = rows - j * k;
                final int lastRow = (j == (nthreads - 1)) ? 0 : firstRow - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; --r >= lastRow;) {
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
        FloatMatrix2D transpose = viewDice().copy();
        int oldNthreads = ConcurrencyUtils.getNumberOfThreads();
        ConcurrencyUtils.setNumberOfThreads(ConcurrencyUtils.nextPow2(oldNthreads));
        if (fft2 == null) {
            fft2 = new FloatFFT_2D(rows, columns);
        }
        fft2.realInverse((float[]) transpose.elements(), scale);
        this.assign(transpose.viewDice().copy());
        ConcurrencyUtils.setNumberOfThreads(oldNthreads);
    }

    public long index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    public FloatMatrix2D like(int rows, int columns) {
        return new DenseColumnFloatMatrix2D(rows, columns);
    }

    public FloatMatrix1D like1D(int size) {
        return new DenseFloatMatrix1D(size);
    }

    public float[] getMaxLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = (int) index(0, 0);
        float maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            float[][] results = new float[nthreads][3];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float maxValue = elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride];
                        int rowLocation = rows - 1;
                        int columnLocation = firstColumn - 1;
                        float elem;
                        int d = 1;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            int cidx = zero + c * columnStride;
                            for (int r = rows - d; --r >= 0;) {
                                elem = elements[r * rowStride + cidx];
                                if (maxValue < elem) {
                                    maxValue = elem;
                                    rowLocation = r;
                                    columnLocation = c;
                                }
                            }
                            d = 0;
                        }
                        return new float[] { maxValue, rowLocation, columnLocation };
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
            maxValue = elements[zero + (rows - 1) * rowStride + (columns - 1) * columnStride];
            rowLocation = rows - 1;
            columnLocation = columns - 1;
            float elem;
            int d = 1;
            for (int c = columns; --c >= 0;) {
                int cidx = zero + c * columnStride;
                for (int r = rows - d; --r >= 0;) {
                    elem = elements[r * rowStride + cidx];
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
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            float[][] results = new float[nthreads][3];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<float[]>() {
                    public float[] call() throws Exception {
                        float minValue = elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride];
                        int rowLocation = rows - 1;
                        int columnLocation = firstColumn - 1;
                        float elem;
                        int d = 1;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            int cidx = zero + c * columnStride;
                            for (int r = rows - d; --r >= 0;) {
                                elem = elements[r * rowStride + cidx];
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
            minValue = elements[zero + (rows - 1) * rowStride + (columns - 1) * columnStride];
            rowLocation = rows - 1;
            columnLocation = columns - 1;
            float elem;
            int d = 1;
            for (int c = columns; --c >= 0;) {
                int cidx = zero + c * columnStride;
                for (int r = rows - d; --r >= 0;) {
                    elem = elements[r * rowStride + cidx];
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
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        final int zero = (int) index(0, 0);
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, r = rows; --r >= 0;) {
                                values[r][c] = elements[i];
                                i -= rowStride;
                            }
                            idx -= columnStride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, r = rows; --r >= 0;) {
                    values[r][c] = elements[i];
                    i -= rowStride;
                }
                idx -= columnStride;
            }
        }
        return values;
    }

    public FloatMatrix1D vectorize() {
        final int size = (int) size();
        FloatMatrix1D v = new DenseFloatMatrix1D(size);
        if (isNoView == true) {
            System.arraycopy(elements, 0, v.elements(), 0, size);
        } else {
            final int zero = (int) index(0, 0);
            final int zeroOther = (int) v.index(0);
            final int strideOther = v.stride();
            final float[] elementsOther = (float[]) v.elements();
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, columns);
                Future<?>[] futures = new Future[nthreads];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = columns - j * k;
                    final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                    final int firstIdxOther = size - j * k * rows;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                            int idxOther = zeroOther + (firstIdxOther - 1) * strideOther;
                            for (int c = firstColumn; --c >= lastColumn;) {
                                for (int i = idx, r = rows; --r >= 0;) {
                                    elementsOther[idxOther] = elements[i];
                                    i -= rowStride;
                                    idxOther -= strideOther;
                                }
                                idx -= columnStride;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
                int idxOther = zeroOther + size - 1;
                for (int c = columns; --c >= 0;) {
                    for (int i = idx, r = rows; --r >= 0;) {
                        elementsOther[idxOther] = elements[i];
                        i -= rowStride;
                        idxOther--;
                    }
                    idx -= columnStride;
                }
            }
        }
        return v;
    }

    public FloatMatrix1D zMult(final FloatMatrix1D y, FloatMatrix1D z, final float alpha, final float beta,
            final boolean transposeA) {
        if (z == null) {
            z = new DenseFloatMatrix1D(transposeA ? columns : rows);
        }

        if ((transposeA ? rows : columns) != y.size() || (transposeA ? columns : rows) > z.size())
            throw new IllegalArgumentException("Incompatible args: " + toStringShort() + ", " + y.toStringShort()
                    + ", " + z.toStringShort());

        if (!(y instanceof DenseFloatMatrix1D) || !(z instanceof DenseFloatMatrix1D) || this.isView() || y.isView()
                || z.isView())
            return super.zMult(y, z, alpha, beta, transposeA);

        float[] yElements = (float[]) y.elements();
        float[] zElements = (float[]) z.elements();
        Transpose transA = transposeA ? Transpose.Transpose : Transpose.NoTranspose;

        BLAS.getInstance().sgemv(transA.netlib(), rows, columns, alpha, elements, Math.max(rows, 1), yElements, 1,
                beta, zElements, 1);
        return z;
    }

    public FloatMatrix2D zMult(final FloatMatrix2D B, FloatMatrix2D C, final float alpha, final float beta,
            final boolean transposeA, final boolean transposeB) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;
        final int rowsB = transposeB ? B.columns() : B.rows();
        final int columnsB = transposeB ? B.rows() : B.columns();
        final int rowsC = rowsA;
        final int columnsC = columnsB;

        if (columnsA != rowsB) {
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + this.toStringShort() + ", "
                    + B.toStringShort());
        }

        if (C == null) {
            C = new DenseColumnFloatMatrix2D(rowsC, columnsC);
        } else {
            if (rowsA != C.rows() || columnsB != C.columns()) {
                throw new IllegalArgumentException("Incompatibe result matrix: " + this.toStringShort() + ", "
                        + B.toStringShort() + ", " + C.toStringShort());
            }
        }
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!(B instanceof DenseColumnFloatMatrix2D) || !(C instanceof DenseColumnFloatMatrix2D) || this.isView()
                || B.isView() || C.isView())
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);

        Transpose transA = transposeA ? Transpose.Transpose : Transpose.NoTranspose;
        Transpose transB = transposeB ? Transpose.Transpose : Transpose.NoTranspose;
        float[] elementsA = elements;
        float[] elementsB = (float[]) B.elements();
        float[] elementsC = (float[]) C.elements();

        int lda = transposeA ? Math.max(1, columnsA) : Math.max(1, rowsA);
        int ldb = transposeB ? Math.max(1, columnsB) : Math.max(1, rowsB);
        int ldc = Math.max(1, rowsA);

        BLAS.getInstance().sgemm(transA.netlib(), transB.netlib(), rowsA, columnsB, columnsA, alpha, elementsA, lda,
                elementsB, ldb, beta, elementsC, ldc);

        return C;
    }

    public float zSum() {
        float sum = 0;
        if (elements == null)
            throw new InternalError();
        final int zero = (int) index(0, 0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {

                    public Float call() throws Exception {
                        float sum = 0;
                        int idx = zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride;
                        for (int c = firstColumn; --c >= lastColumn;) {
                            for (int i = idx, r = rows; --r >= 0;) {
                                sum += elements[i];
                                i -= rowStride;
                            }
                            idx -= columnStride;
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
            int idx = zero + (rows - 1) * rowStride + (columns - 1) * columnStride;
            for (int c = columns; --c >= 0;) {
                for (int i = idx, r = rows; --r >= 0;) {
                    sum += elements[i];
                    i -= rowStride;
                }
                idx -= columnStride;
            }
        }
        return sum;
    }

    protected boolean haveSharedCellsRaw(FloatMatrix2D other) {
        if (other instanceof SelectedDenseColumnFloatMatrix2D) {
            SelectedDenseColumnFloatMatrix2D otherMatrix = (SelectedDenseColumnFloatMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseColumnFloatMatrix2D) {
            DenseColumnFloatMatrix2D otherMatrix = (DenseColumnFloatMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    protected FloatMatrix1D like1D(int size, int zero, int stride) {
        return new DenseFloatMatrix1D(size, this.elements, zero, stride, true);
    }

    protected FloatMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseColumnFloatMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }
}
