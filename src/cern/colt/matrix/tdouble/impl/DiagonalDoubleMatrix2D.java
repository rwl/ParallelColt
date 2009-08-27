/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Diagonal 2-d matrix holding <tt>double</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DiagonalDoubleMatrix2D extends WrapperDoubleMatrix2D {
    private static final long serialVersionUID = 1L;

    /*
     * The non zero elements of the matrix.
     */
    protected double[] elements;

    /*
     * Length of the diagonal
     */
    protected int dlength;

    /*
     * An m-by-n matrix A has m+n-1 diagonals. Since the DiagonalDoubleMatrix2D can have only one
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
    public DiagonalDoubleMatrix2D(double[][] values, int dindex) {
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
     *             if <tt>size<0 (double)size > Integer.MAX_VALUE</tt>.
     */
    public DiagonalDoubleMatrix2D(int rows, int columns, int dindex) {
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
        elements = new double[dlength];
    }

    public DoubleMatrix2D assign(final cern.colt.function.tdouble.DoubleFunction function) {
        if (function instanceof cern.jet.math.tdouble.DoubleMult) { // x[i] = mult*x[i]
            final double alpha = ((cern.jet.math.tdouble.DoubleMult) function).multiplicator;
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

    public DoubleMatrix2D assign(double value) {
        for (int i = dlength; --i >= 0;)
            elements[i] = value;
        return this;
    }

    public DoubleMatrix2D assign(final double[] values) {
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

    public DoubleMatrix2D assign(final double[][] values) {
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

    public DoubleMatrix2D assign(DoubleMatrix2D source) {
        // overriden for performance only
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof DiagonalDoubleMatrix2D) {
            DiagonalDoubleMatrix2D other = (DiagonalDoubleMatrix2D) source;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("source is DiagonalDoubleMatrix2D with different diagonal stored.");
            }
            // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        } else {
            return super.assign(source);
        }
    }

    public DoubleMatrix2D assign(final DoubleMatrix2D y, final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        checkShape(y);
        if (y instanceof DiagonalDoubleMatrix2D) {
            DiagonalDoubleMatrix2D other = (DiagonalDoubleMatrix2D) y;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("y is DiagonalDoubleMatrix2D with different diagonal stored.");
            }
            if (function instanceof cern.jet.math.tdouble.DoublePlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                final double alpha = ((cern.jet.math.tdouble.DoublePlusMultSecond) function).multiplicator;
                if (alpha == 0) {
                    return this; // nothing to do
                }
            }
            final double[] otherElements = other.elements;
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
                            if (function instanceof cern.jet.math.tdouble.DoublePlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                                final double alpha = ((cern.jet.math.tdouble.DoublePlusMultSecond) function).multiplicator;
                                if (alpha == 1) {
                                    for (int j = firstRow; j < lastRow; j++) {
                                        elements[j] += otherElements[j];
                                    }
                                } else {
                                    for (int j = firstRow; j < lastRow; j++) {
                                        elements[j] = elements[j] + alpha * otherElements[j];
                                    }
                                }
                            } else if (function == cern.jet.math.tdouble.DoubleFunctions.mult) { // x[i] = x[i] * y[i]
                                for (int j = firstRow; j < lastRow; j++) {
                                    elements[j] = elements[j] * otherElements[j];
                                }
                            } else if (function == cern.jet.math.tdouble.DoubleFunctions.div) { // x[i] = x[i] /  y[i]
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
                if (function instanceof cern.jet.math.tdouble.DoublePlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                    final double alpha = ((cern.jet.math.tdouble.DoublePlusMultSecond) function).multiplicator;
                    if (alpha == 1) {
                        for (int j = dlength; --j >= 0;) {
                            elements[j] += otherElements[j];
                        }
                    } else {
                        for (int j = dlength; --j >= 0;) {
                            elements[j] = elements[j] + alpha * otherElements[j];
                        }
                    }
                } else if (function == cern.jet.math.tdouble.DoubleFunctions.mult) { // x[i] = x[i] * y[i]
                    for (int j = dlength; --j >= 0;) {
                        elements[j] = elements[j] * otherElements[j];
                    }
                } else if (function == cern.jet.math.tdouble.DoubleFunctions.div) { // x[i] = x[i] /  y[i]
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

    public double[] elements() {
        return elements;
    }

    public boolean equals(double value) {
        double epsilon = cern.colt.matrix.tdouble.algo.DoubleProperty.DEFAULT.tolerance();
        for (int r = 0; r < dlength; r++) {
            double x = elements[r];
            double diff = Math.abs(value - x);
            if ((diff != diff) && ((value != value && x != x) || value == x))
                diff = 0;
            if (!(diff <= epsilon)) {
                return false;
            }
        }
        return true;
    }

    public boolean equals(Object obj) {
        if (obj instanceof DiagonalDoubleMatrix2D) {
            DiagonalDoubleMatrix2D other = (DiagonalDoubleMatrix2D) obj;
            double epsilon = cern.colt.matrix.tdouble.algo.DoubleProperty.DEFAULT.tolerance();
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
            double[] otherElements = other.elements;
            for (int r = 0; r < dlength; r++) {
                double x = elements[r];
                double value = otherElements[r];
                double diff = Math.abs(value - x);
                if ((diff != diff) && ((value != value && x != x) || value == x))
                    diff = 0;
                if (!(diff <= epsilon)) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(obj);
        }
    }

    public DoubleMatrix2D forEachNonZero(final cern.colt.function.tdouble.IntIntDoubleFunction function) {
        for (int j = dlength; --j >= 0;) {
            double value = elements[j];
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

    public double[] getMaxLocation() {
        int location = 0;
        double maxValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            double[][] results = new double[nthreads][2];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int location = firstRow;
                        double maxValue = elements[location];
                        double elem;
                        for (int r = firstRow + 1; r < lastRow; r++) {
                            elem = elements[r];
                            if (maxValue < elem) {
                                maxValue = elem;
                                location = r;
                            }
                        }
                        return new double[] { maxValue, location, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (double[]) futures[j].get();
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
            double elem;
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
        return new double[] { maxValue, rowLocation, columnLocation };
    }

    public double[] getMinLocation() {
        int location = 0;
        double minValue = 0;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            double[][] results = new double[nthreads][2];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<double[]>() {
                    public double[] call() throws Exception {
                        int location = firstRow;
                        double minValue = elements[location];
                        double elem;
                        for (int r = firstRow + 1; r < lastRow; r++) {
                            elem = elements[r];
                            if (minValue > elem) {
                                minValue = elem;
                                location = r;
                            }
                        }
                        return new double[] { minValue, location, location };
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (double[]) futures[j].get();
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
            double elem;
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
        return new double[] { minValue, rowLocation, columnLocation };
    }

    public double getQuick(int row, int column) {
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

    public DoubleMatrix2D like(int rows, int columns) {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    public DoubleMatrix1D like1D(int size) {
        return new SparseDoubleMatrix1D(size);
    }

    public void setQuick(int row, int column, double value) {
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

    public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, double alpha, double beta, final boolean transposeA) {
        int rowsA = rows;
        int columnsA = columns;
        if (transposeA) {
            rowsA = columns;
            columnsA = rows;
        }

        boolean ignore = (z == null);
        if (z == null)
            z = new DenseDoubleMatrix1D(rowsA);

        if (!(this.isNoView && y instanceof DenseDoubleMatrix1D && z instanceof DenseDoubleMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        if ((!ignore) && ((beta) != 1))
            z.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta));

        DenseDoubleMatrix1D zz = (DenseDoubleMatrix1D) z;
        final double[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) z.index(0);

        DenseDoubleMatrix1D yy = (DenseDoubleMatrix1D) y;
        final double[] elementsY = yy.elements;
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

    protected DoubleMatrix2D getContent() {
        return this;
    }
}
