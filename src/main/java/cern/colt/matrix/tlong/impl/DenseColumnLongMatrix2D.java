/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong.impl;

import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.function.tlong.LongFunction;
import cern.colt.function.tlong.LongLongFunction;
import cern.colt.function.tlong.LongProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.matrix.io.MatrixInfo;
import cern.colt.matrix.io.MatrixSize;
import cern.colt.matrix.io.MatrixVectorReader;
import cern.colt.matrix.tlong.LongMatrix1D;
import cern.colt.matrix.tlong.LongMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 2-d matrix holding <tt>long</tt> elements. First see the <a
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
public class DenseColumnLongMatrix2D extends LongMatrix2D {
    static final long serialVersionUID = 1020177651L;
    protected long[] elements;

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
    public DenseColumnLongMatrix2D(long[][] values) {
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
    public DenseColumnLongMatrix2D(int rows, int columns) {
        setUp(rows, columns, 0, 0, 1, rows);
        this.elements = new long[rows * columns];
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
    public DenseColumnLongMatrix2D(int rows, int columns, long[] elements, int rowZero, int columnZero, int rowStride,
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
    public DenseColumnLongMatrix2D(MatrixVectorReader reader) throws IOException {
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
        this.elements = new long[rows * columns];
        int numEntries = size.numEntries();
        int[] columnIndexes = new int[numEntries];
        int[] rowIndexes = new int[numEntries];
        long[] values = new long[numEntries];
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

    public long aggregate(final LongLongFunction aggr, final LongFunction f) {
        if (size() == 0)
            throw new IllegalArgumentException("size() = 0");
        final int zero = (int) index(0, 0);
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride]);
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

    public long aggregate(final LongLongFunction aggr, final LongFunction f, final LongProcedure cond) {
        if (size() == 0)
            throw new IllegalArgumentException("size() = 0");
        final int zero = (int) index(0, 0);
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long elem = elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride];
                        long a = 0;
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
            long elem = elements[zero + (rows - 1) * rowStride + (columns - 1) * columnStride];
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

    public long aggregate(final LongLongFunction aggr, final LongFunction f, final IntArrayList rowList,
            final IntArrayList columnList) {
        if (size() == 0)
            throw new IllegalArgumentException("size() = 0");
        final int zero = (int) index(0, 0);
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = size - j * k;
                final int lastIdx = (j == (nthreads - 1)) ? 0 : firstIdx - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + rowElements[firstIdx - 1] * rowStride
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

    public long aggregate(final LongMatrix2D other, final LongLongFunction aggr, final LongLongFunction f) {
        if (!(other instanceof DenseColumnLongMatrix2D)) {
            return super.aggregate(other, aggr, f);
        }
        checkShape(other);
        if (size() == 0)
            throw new IllegalArgumentException("size() = 0");
        final int zero = (int) index(0, 0);
        final int zeroOther = (int) other.index(0, 0);
        final int rowStrideOther = other.rowStride();
        final int columnStrideOther = other.columnStride();
        final long[] otherElements = (long[]) other.elements();
        long a = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long a = f.apply(elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride],
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

    public LongMatrix2D assign(final LongFunction function) {
        if (function instanceof cern.jet.math.tlong.LongMult) { // x[i] = mult*x[i]
            long multiplicator = ((cern.jet.math.tlong.LongMult) function).multiplicator;
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
                        if (function instanceof cern.jet.math.tlong.LongMult) { // x[i] = mult*x[i]
                            long multiplicator = ((cern.jet.math.tlong.LongMult) function).multiplicator;
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
            if (function instanceof cern.jet.math.tlong.LongMult) { // x[i] = mult*x[i]
                long multiplicator = ((cern.jet.math.tlong.LongMult) function).multiplicator;
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

    public LongMatrix2D assign(final LongProcedure cond, final LongFunction function) {
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
                        long elem;
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
            long elem;
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

    public LongMatrix2D assign(final LongProcedure cond, final long value) {
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
                        long elem;
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
            long elem;
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

    public LongMatrix2D assign(final long value) {
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

    public LongMatrix2D assign(final long[] values) {
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

    public LongMatrix2D assign(final long[][] values) {
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
                            long[] currentRow = values[r];
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
                long[] currentRow = values[r];
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

    public LongMatrix2D assign(final LongMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof DenseColumnLongMatrix2D)) {
            super.assign(source);
            return this;
        }
        DenseColumnLongMatrix2D other = (DenseColumnLongMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (this.isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, elements, 0, elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            LongMatrix2D c = other.copy();
            if (!(c instanceof DenseColumnLongMatrix2D)) { // should not happen
                super.assign(other);
                return this;
            }
            other = (DenseColumnLongMatrix2D) c;
        }

        final int zeroOther = (int) other.index(0, 0);
        final int zero = (int) index(0, 0);
        final int columnStrideOther = other.columnStride;
        final int rowStrideOther = other.rowStride;
        final long[] otherElements = other.elements;
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

    public LongMatrix2D assign(final LongMatrix2D y, final LongLongFunction function) {
        if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
            long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
            if (multiplicator == 0) { // x[i] = x[i] + 0*y[i]
                return this;
            }
        }
        if (function instanceof cern.jet.math.tlong.LongPlusMultFirst) {
            long multiplicator = ((cern.jet.math.tlong.LongPlusMultFirst) function).multiplicator;
            if (multiplicator == 0) { // x[i] = 0*x[i] + y[i]
                return assign(y);
            }
        }
        if (!(y instanceof DenseColumnLongMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        DenseColumnLongMatrix2D other = (DenseColumnLongMatrix2D) y;
        checkShape(y);
        final long[] otherElements = other.elements;
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
                        if (function == cern.jet.math.tlong.LongFunctions.mult) {
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
                        } else if (function == cern.jet.math.tlong.LongFunctions.div) {
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
                        } else if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
                            long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
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
                        } else if (function instanceof cern.jet.math.tlong.LongPlusMultFirst) {
                            long multiplicator = ((cern.jet.math.tlong.LongPlusMultFirst) function).multiplicator;
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
            if (function == cern.jet.math.tlong.LongFunctions.mult) {
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
            } else if (function == cern.jet.math.tlong.LongFunctions.div) {
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
            } else if (function instanceof cern.jet.math.tlong.LongPlusMultSecond) {
                long multiplicator = ((cern.jet.math.tlong.LongPlusMultSecond) function).multiplicator;
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
            } else if (function instanceof cern.jet.math.tlong.LongPlusMultFirst) {
                long multiplicator = ((cern.jet.math.tlong.LongPlusMultFirst) function).multiplicator;
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

    public LongMatrix2D assign(final LongMatrix2D y, final LongLongFunction function, IntArrayList rowList,
            IntArrayList columnList) {
        checkShape(y);
        if (!(y instanceof DenseColumnLongMatrix2D)) {
            super.assign(y, function);
            return this;
        }
        DenseColumnLongMatrix2D other = (DenseColumnLongMatrix2D) y;
        final int size = rowList.size();
        final int[] rowElements = rowList.elements();
        final int[] columnElements = columnList.elements();
        final long[] otherElements = other.elements();
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

    public long[] elements() {
        return elements;
    }

    public LongMatrix2D forEachNonZero(final cern.colt.function.tlong.IntIntLongFunction function) {
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
                                long value = elements[i];
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
                    long value = elements[i];
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
    public DenseLongMatrix2D getRowMajor() {
        DenseLongMatrix2D R = new DenseLongMatrix2D(rows, columns);
        final int zeroR = (int) R.index(0, 0);
        final int rowStrideR = R.rowStride();
        final int columnStrideR = R.columnStride();
        final long[] elementsR = R.elements();
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

    public void getNegativeValues(final IntArrayList rowList, final IntArrayList columnList,
            final LongArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int c = 0; c < columns; c++) {
            for (int i = idx, r = 0; r < rows; r++) {
                long value = elements[i];
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

    public void getNonZeros(final IntArrayList rowList, final IntArrayList columnList, final LongArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int c = 0; c < columns; c++) {
            for (int i = idx, r = 0; r < rows; r++) {
                long value = elements[i];
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
            final LongArrayList valueList) {
        rowList.clear();
        columnList.clear();
        valueList.clear();
        int idx = (int) index(0, 0);
        for (int c = 0; c < columns; c++) {
            for (int i = idx, r = 0; r < rows; r++) {
                long value = elements[i];
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

    public long getQuick(int row, int column) {
        return elements[rowZero + row * rowStride + columnZero + column * columnStride];
    }

    public long index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    public LongMatrix2D like(int rows, int columns) {
        return new DenseColumnLongMatrix2D(rows, columns);
    }

    public LongMatrix1D like1D(int size) {
        return new DenseLongMatrix1D(size);
    }

    public long[] getMaxLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = (int) index(0, 0);
        long maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][3];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        long maxValue = elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride];
                        int rowLocation = rows - 1;
                        int columnLocation = firstColumn - 1;
                        long elem;
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
                        return new long[] { maxValue, rowLocation, columnLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
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
            long elem;
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
        return new long[] { maxValue, rowLocation, columnLocation };
    }

    public long[] getMinLocation() {
        int rowLocation = 0;
        int columnLocation = 0;
        final int zero = (int) index(0, 0);
        long minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            long[][] results = new long[nthreads][3];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = columns - j * k;
                final int lastColumn = (j == (nthreads - 1)) ? 0 : firstColumn - k;
                futures[j] = ConcurrencyUtils.submit(new Callable<long[]>() {
                    public long[] call() throws Exception {
                        long minValue = elements[zero + (rows - 1) * rowStride + (firstColumn - 1) * columnStride];
                        int rowLocation = rows - 1;
                        int columnLocation = firstColumn - 1;
                        long elem;
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
                        return new long[] { minValue, rowLocation, columnLocation };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (long[]) futures[j].get();
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
            long elem;
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
        return new long[] { minValue, rowLocation, columnLocation };
    }

    public void setQuick(int row, int column, long value) {
        elements[rowZero + row * rowStride + columnZero + column * columnStride] = value;
    }

    public long[][] toArray() {
        final long[][] values = new long[rows][columns];
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

    public LongMatrix1D vectorize() {
        final int size = (int) size();
        LongMatrix1D v = new DenseLongMatrix1D(size);
        if (isNoView == true) {
            System.arraycopy(elements, 0, v.elements(), 0, size);
        } else {
            final int zero = (int) index(0, 0);
            final int zeroOther = (int) v.index(0);
            final int strideOther = v.stride();
            final long[] elementsOther = (long[]) v.elements();
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

    public long zSum() {
        long sum = 0;
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
                futures[j] = ConcurrencyUtils.submit(new Callable<Long>() {

                    public Long call() throws Exception {
                        long sum = 0;
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
                    sum += (Long) futures[j].get();
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

    protected boolean haveSharedCellsRaw(LongMatrix2D other) {
        if (other instanceof SelectedDenseColumnLongMatrix2D) {
            SelectedDenseColumnLongMatrix2D otherMatrix = (SelectedDenseColumnLongMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseColumnLongMatrix2D) {
            DenseColumnLongMatrix2D otherMatrix = (DenseColumnLongMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    protected LongMatrix1D like1D(int size, int zero, int stride) {
        return new DenseLongMatrix1D(size, this.elements, zero, stride, true);
    }

    protected LongMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedDenseColumnLongMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }
}
