/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Diagonal 2-d matrix holding <tt>int</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DiagonalIntMatrix2D extends WrapperIntMatrix2D {
    private static final long serialVersionUID = 1L;

    /*
     * The non zero elements of the matrix.
     */
    protected int[] elements;

    /*
     * Length of the diagonal
     */
    protected int dlength;

    /*
     * An m-by-n matrix A has m+n-1 diagonals. Since the DiagonalIntMatrix2D can have only one
     * diagonal, dindex is a value from interval [-m+1, n-1] that denotes which diagonal is stored.
     */
    protected int dindex;

    /**
     * Constructs a matrix with a copy of the given values. <tt>values</tt> is
     * required to have the form <tt>values[row][column]</tt> and have exactly
     * the same number of columns in every row. Only the values on the main
     * diagonal, i.e. values[i][i] are used.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     * @param dindex
     *            index of the diagonal.
     * @throws IllegalArgumentException
     *             if
     * 
     *             <tt>for any 1 &lt;= row &lt; values.length: values[row].length != values[row-1].length || index < -rows+1 || index > columns - 1</tt>
     *             .
     */
    public DiagonalIntMatrix2D(int[][] values, int dindex) {
        this(values.length, values.length == 0 ? 0 : values[0].length, dindex);
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
     * @param dindex
     *            index of the diagonal.
     * @throws IllegalArgumentException
     *             if <tt>size<0 (int)size > Integer.MAX_VALUE</tt>.
     */
    public DiagonalIntMatrix2D(int rows, int columns, int dindex) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        if ((dindex < -rows + 1) || (dindex > columns - 1)) {
            throw new IllegalArgumentException("index is out of bounds");
        } else {
            this.dindex = dindex;
        }
        if (dindex == 0) {
            dlength = Math.min(rows, columns);
        } else if (dindex > 0) {
            if (rows >= columns) {
                dlength = columns - dindex;
            } else {
                int diff = columns - rows;
                if (dindex <= diff) {
                    dlength = rows;
                } else {
                    dlength = rows - (dindex - diff);
                }
            }
        } else {
            if (rows >= columns) {
                int diff = rows - columns;
                if (-dindex <= diff) {
                    dlength = columns;
                } else {
                    dlength = columns + dindex + diff;
                }
            } else {
                dlength = rows + dindex;
            }
        }
        elements = new int[dlength];
    }

    public IntMatrix2D assign(final cern.colt.function.tint.IntFunction function) {
        if (function instanceof cern.jet.math.tint.IntMult) { // x[i] = mult*x[i]
            final int alpha = ((cern.jet.math.tint.IntMult) function).multiplicator;
            if (alpha == 1)
                return this;
            if (alpha == 0)
                return assign(0);
            if (alpha != alpha)
                return assign(alpha); // the funny definition of isNaN(). This should better not happen.
            for (int j = dlength; --j >= 0;) {
                elements[j] *= alpha;
            }
        } else {
            for (int j = dlength; --j >= 0;) {
                elements[j] = function.apply(elements[j]);
            }
        }
        return this;
    }

    public IntMatrix2D assign(int value) {
        for (int i = dlength; --i >= 0;)
            elements[i] = value;
        return this;
    }

    public IntMatrix2D assign(final int[] values) {
        if (values.length != dlength)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + " dlength=" + dlength);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            elements[r] = values[r];
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = dlength; --r >= 0;) {
                elements[r] = values[r];
            }
        }
        return this;
    }

    public IntMatrix2D assign(final int[][] values) {
        if (values.length != rows)
            throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "rows()="
                    + rows());
        int r, c;
        if (dindex >= 0) {
            r = 0;
            c = dindex;
        } else {
            r = -dindex;
            c = 0;
        }
        for (int i = 0; i < dlength; i++) {
            if (values[i].length != columns) {
                throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                        + values[r].length + "columns()=" + columns());
            }
            elements[i] = values[r++][c++];
        }
        return this;
    }

    public IntMatrix2D assign(IntMatrix2D source) {
        // overriden for performance only
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof DiagonalIntMatrix2D) {
            DiagonalIntMatrix2D other = (DiagonalIntMatrix2D) source;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("source is DiagonalIntMatrix2D with different diagonal stored.");
            }
            // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        } else {
            return super.assign(source);
        }
    }

    public IntMatrix2D assign(final IntMatrix2D y, final cern.colt.function.tint.IntIntFunction function) {
        checkShape(y);
        if (y instanceof DiagonalIntMatrix2D) {
            DiagonalIntMatrix2D other = (DiagonalIntMatrix2D) y;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("y is DiagonalIntMatrix2D with different diagonal stored.");
            }
            if (function instanceof cern.jet.math.tint.IntPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                final int alpha = ((cern.jet.math.tint.IntPlusMultSecond) function).multiplicator;
                if (alpha == 0) {
                    return this; // nothing to do
                }
            }
            final int[] otherElements = other.elements;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, dlength);
                Future<?>[] futures = new Future[nthreads];
                int k = dlength / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            if (function instanceof cern.jet.math.tint.IntPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                                final int alpha = ((cern.jet.math.tint.IntPlusMultSecond) function).multiplicator;
                                if (alpha == 1) {
                                    for (int j = firstRow; j < lastRow; j++) {
                                        elements[j] += otherElements[j];
                                    }
                                } else {
                                    for (int j = firstRow; j < lastRow; j++) {
                                        elements[j] = elements[j] + alpha * otherElements[j];
                                    }
                                }
                            } else if (function == cern.jet.math.tint.IntFunctions.mult) { // x[i] = x[i] * y[i]
                                for (int j = firstRow; j < lastRow; j++) {
                                    elements[j] = elements[j] * otherElements[j];
                                }
                            } else if (function == cern.jet.math.tint.IntFunctions.div) { // x[i] = x[i] /  y[i]
                                for (int j = firstRow; j < lastRow; j++) {
                                    elements[j] = elements[j] / otherElements[j];
                                }
                            } else {
                                for (int j = firstRow; j < lastRow; j++) {
                                    elements[j] = function.apply(elements[j], otherElements[j]);
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                if (function instanceof cern.jet.math.tint.IntPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                    final int alpha = ((cern.jet.math.tint.IntPlusMultSecond) function).multiplicator;
                    if (alpha == 1) {
                        for (int j = dlength; --j >= 0;) {
                            elements[j] += otherElements[j];
                        }
                    } else {
                        for (int j = dlength; --j >= 0;) {
                            elements[j] = elements[j] + alpha * otherElements[j];
                        }
                    }
                } else if (function == cern.jet.math.tint.IntFunctions.mult) { // x[i] = x[i] * y[i]
                    for (int j = dlength; --j >= 0;) {
                        elements[j] = elements[j] * otherElements[j];
                    }
                } else if (function == cern.jet.math.tint.IntFunctions.div) { // x[i] = x[i] /  y[i]
                    for (int j = dlength; --j >= 0;) {
                        elements[j] = elements[j] / otherElements[j];
                    }
                } else {
                    for (int j = dlength; --j >= 0;) {
                        elements[j] = function.apply(elements[j], otherElements[j]);
                    }
                }
            }
            return this;
        } else {
            return super.assign(y, function);
        }
    }

    public int cardinality() {
        int cardinality = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            Integer[] results = new Integer[nthreads];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        for (int r = firstRow; r < lastRow; r++) {
                            if (elements[r] != 0)
                                cardinality++;
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
            for (int r = 0; r < dlength; r++) {
                if (elements[r] != 0)
                    cardinality++;
            }
        }
        return cardinality;
    }

    public int[] elements() {
        return elements;
    }

    public boolean equals(int value) {
        for (int r = 0; r < dlength; r++) {
            int x = elements[r];
            int diff = value - x;
            if (diff != 0) {
                return false;
            }
        }
        return true;
    }

    public boolean equals(Object obj) {
        if (obj instanceof DiagonalIntMatrix2D) {
            DiagonalIntMatrix2D other = (DiagonalIntMatrix2D) obj;
            if (this == obj)
                return true;
            if (!(this != null && obj != null))
                return false;
            final int rows = this.rows();
            final int columns = this.columns();
            if (columns != other.columns() || rows != other.rows())
                return false;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                return false;
            }
            int[] otherElements = other.elements;
            for (int r = 0; r < dlength; r++) {
                int x = elements[r];
                int value = otherElements[r];
                int diff = value - x;
                if (diff != 0) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(obj);
        }
    }

    public IntMatrix2D forEachNonZero(final cern.colt.function.tint.IntIntIntFunction function) {
        for (int j = dlength; --j >= 0;) {
            int value = elements[j];
            if (value != 0) {
                elements[j] = function.apply(j, j, value);
            }
        }
        return this;
    }

    /**
     * Returns the length of the diagonal
     * 
     * @return the length of the diagonal
     */
    public int diagonalLength() {
        return dlength;
    }

    /**
     * Returns the index of the diagonal
     * 
     * @return the index of the diagonal
     */
    public int diagonalIndex() {
        return dindex;
    }

    public int[] getMaxLocation() {
        int location = 0;
        int maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            int[][] results = new int[nthreads][2];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<int[]>() {
                    public int[] call() throws Exception {
                        int location = firstRow;
                        int maxValue = elements[location];
                        int elem;
                        for (int r = firstRow + 1; r < lastRow; r++) {
                            elem = elements[r];
                            if (maxValue < elem) {
                                maxValue = elem;
                                location = r;
                            }
                        }
                        return new int[] { maxValue, location, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (int[]) futures[j].get();
                }
                maxValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < nthreads; j++) {
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
            maxValue = elements[0];
            int elem;
            for (int r = 1; r < dlength; r++) {
                elem = elements[r];
                if (maxValue < elem) {
                    maxValue = elem;
                    location = r;
                }
            }
        }
        int rowLocation;
        int columnLocation;
        if (dindex > 0) {
            rowLocation = location;
            columnLocation = location + dindex;
        } else if (dindex < 0) {
            rowLocation = location - dindex;
            columnLocation = location;
        } else {
            rowLocation = location;
            columnLocation = location;
        }
        return new int[] { maxValue, rowLocation, columnLocation };
    }

    public int[] getMinLocation() {
        int location = 0;
        int minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            int[][] results = new int[nthreads][2];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<int[]>() {
                    public int[] call() throws Exception {
                        int location = firstRow;
                        int minValue = elements[location];
                        int elem;
                        for (int r = firstRow + 1; r < lastRow; r++) {
                            elem = elements[r];
                            if (minValue > elem) {
                                minValue = elem;
                                location = r;
                            }
                        }
                        return new int[] { minValue, location, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (int[]) futures[j].get();
                }
                minValue = results[0][0];
                location = (int) results[0][1];
                for (int j = 1; j < nthreads; j++) {
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
            minValue = elements[0];
            int elem;
            for (int r = 1; r < dlength; r++) {
                elem = elements[r];
                if (minValue > elem) {
                    minValue = elem;
                    location = r;
                }
            }
        }
        int rowLocation;
        int columnLocation;
        if (dindex > 0) {
            rowLocation = location;
            columnLocation = location + dindex;
        } else if (dindex < 0) {
            rowLocation = location - dindex;
            columnLocation = location;
        } else {
            rowLocation = location;
            columnLocation = location;
        }
        return new int[] { minValue, rowLocation, columnLocation };
    }

    public int getQuick(int row, int column) {
        if (dindex >= 0) {
            if (column < dindex) {
                return 0;
            } else {
                if ((row < dlength) && (row + dindex == column)) {
                    return elements[row];
                } else {
                    return 0;
                }
            }
        } else {
            if (row < -dindex) {
                return 0;
            } else {
                if ((column < dlength) && (row + dindex == column)) {
                    return elements[column];
                } else {
                    return 0;
                }
            }
        }
    }

    public IntMatrix2D like(int rows, int columns) {
        return new SparseIntMatrix2D(rows, columns);
    }

    public IntMatrix1D like1D(int size) {
        return new SparseIntMatrix1D(size);
    }

    public void setQuick(int row, int column, int value) {
        if (dindex >= 0) {
            if (column < dindex) {
                //do nothing
            } else {
                if ((row < dlength) && (row + dindex == column)) {
                    elements[row] = value;
                } else {
                    // do nothing
                }
            }
        } else {
            if (row < -dindex) {
                //do nothing
            } else {
                if ((column < dlength) && (row + dindex == column)) {
                    elements[column] = value;
                } else {
                    //do nothing;
                }
            }
        }
    }

    public IntMatrix1D zMult(IntMatrix1D y, IntMatrix1D z, int alpha, int beta, final boolean transposeA) {
        int rowsA = rows;
        int columnsA = columns;
        if (transposeA) {
            rowsA = columns;
            columnsA = rows;
        }

        boolean ignore = (z == null);
        if (z == null)
            z = new DenseIntMatrix1D(rowsA);

        if (!(this.isNoView && y instanceof DenseIntMatrix1D && z instanceof DenseIntMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        if ((!ignore) && ((beta) != 1))
            z.assign(cern.jet.math.tint.IntFunctions.mult(beta));

        DenseIntMatrix1D zz = (DenseIntMatrix1D) z;
        final int[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) z.index(0);

        DenseIntMatrix1D yy = (DenseIntMatrix1D) y;
        final int[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) y.index(0);

        if (elementsY == null || elementsZ == null)
            throw new InternalError();
        if (!transposeA) {
            if (dindex >= 0) {
                for (int i = dlength; --i >= 0;) {
                    elementsZ[zeroZ + strideZ * i] += alpha * elements[i] * elementsY[dindex + zeroY + strideY * i];
                }
            } else {
                for (int i = dlength; --i >= 0;) {
                    elementsZ[-dindex + zeroZ + strideZ * i] += alpha * elements[i] * elementsY[zeroY + strideY * i];
                }
            }
        } else {
            if (dindex >= 0) {
                for (int i = dlength; --i >= 0;) {
                    elementsZ[dindex + zeroZ + strideZ * i] += alpha * elements[i] * elementsY[zeroY + strideY * i];
                }
            } else {
                for (int i = dlength; --i >= 0;) {
                    elementsZ[zeroZ + strideZ * i] += alpha * elements[i] * elementsY[-dindex + zeroY + strideY * i];
                }
            }

        }
        return z;
    }

    protected IntMatrix2D getContent() {
        return this;
    }
}
