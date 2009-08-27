/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.jet.math.tfcomplex.FComplex;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Diagonal 2-d matrix holding <tt>complex</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DiagonalFComplexMatrix2D extends WrapperFComplexMatrix2D {
    private static final long serialVersionUID = 1L;

    /*
     * The non zero elements of the matrix.
     */
    protected float[] elements;

    /*
     * Length of the diagonal
     */
    protected int dlength;

    /*
     * An m-by-n matrix A has m+n-1 diagonals. Since the DiagonalFComplexMatrix2D can have only one
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
    public DiagonalFComplexMatrix2D(float[][] values, int dindex) {
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
     *             if <tt>size<0 (float)size > Integer.MAX_VALUE</tt>.
     */
    public DiagonalFComplexMatrix2D(int rows, int columns, int dindex) {
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
        elements = new float[2 * dlength];
    }

    public FComplexMatrix2D assign(final cern.colt.function.tfcomplex.FComplexFComplexFunction function) {
        if (function instanceof cern.jet.math.tfcomplex.FComplexMult) { // x[i] = mult*x[i]
            final float[] alpha = ((cern.jet.math.tfcomplex.FComplexMult) function).multiplicator;
            if (alpha[0] == 1 && alpha[1] == 0)
                return this;
            if (alpha[0] == 0 && alpha[1] == 0)
                return assign(alpha);
            if (alpha[0] != alpha[0] || alpha[1] != alpha[1])
                return assign(alpha); // the funny definition of isNaN(). This should better not happen.
            float[] elem = new float[2];
            for (int j = 0; j < dlength; j++) {
                elem[0] = elements[2 * j];
                elem[1] = elements[2 * j + 1];
                elem = FComplex.mult(elem, alpha);
                elements[2 * j] = elem[0];
                elements[2 * j + 1] = elem[1];
            }
        } else {
            float[] elem = new float[2];
            for (int j = 0; j < dlength; j++) {
                elem[0] = elements[2 * j];
                elem[1] = elements[2 * j + 1];
                elem = function.apply(elem);
                elements[2 * j] = elem[0];
                elements[2 * j + 1] = elem[1];
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(float re, float im) {
        for (int j = 0; j < dlength; j++) {
            elements[2 * j] = re;
            elements[2 * j + 1] = im;
        }
        return this;
    }

    public FComplexMatrix2D assign(final float[] values) {
        if (values.length != 2 * dlength)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + " 2*dlength=" + 2
                    * dlength);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? dlength : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            elements[2 * i] = values[2 * i];
                            elements[2 * i + 1] = values[2 * i + 1];
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = 0; i < dlength; i++) {
                elements[2 * i] = values[2 * i];
                elements[2 * i + 1] = values[2 * i + 1];
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(final float[][] values) {
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
            if (values[i].length != 2 * columns) {
                throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                        + values[r].length + "2 * columns()=" + 2 * columns());
            }
            elements[2 * i] = values[r][2 * c];
            elements[2 * i + 1] = values[r][2 * c + 1];
            c++;
            r++;
        }
        return this;
    }

    public FComplexMatrix2D assign(FComplexMatrix2D source) {
        // overriden for performance only
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof DiagonalFComplexMatrix2D) {
            DiagonalFComplexMatrix2D other = (DiagonalFComplexMatrix2D) source;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("source is DiagonalFComplexMatrix2D with different diagonal stored.");
            }
            // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        } else {
            return super.assign(source);
        }
    }

    public FComplexMatrix2D assign(final FComplexMatrix2D y,
            final cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction function) {
        checkShape(y);
        if (y instanceof DiagonalFComplexMatrix2D) {
            DiagonalFComplexMatrix2D other = (DiagonalFComplexMatrix2D) y;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("y is DiagonalFComplexMatrix2D with different diagonal stored.");
            }
            if (function instanceof cern.jet.math.tfcomplex.FComplexPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                final float[] alpha = ((cern.jet.math.tfcomplex.FComplexPlusMultSecond) function).multiplicator;
                if (alpha[0] == 0 && alpha[1] == 0) {
                    return this; // nothing to do
                }
            }
            final float[] otherElements = other.elements;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, dlength);
                Future<?>[] futures = new Future[nthreads];
                int k = dlength / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == nthreads - 1) ? dlength : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            if (function instanceof cern.jet.math.tfcomplex.FComplexPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                                final float alpha[] = ((cern.jet.math.tfcomplex.FComplexPlusMultSecond) function).multiplicator;
                                if (alpha[0] == 1 && alpha[1] == 0) {
                                    for (int j = firstIdx; j < lastIdx; j++) {
                                        elements[2 * j] += otherElements[2 * j];
                                        elements[2 * j + 1] += otherElements[2 * j + 1];
                                    }
                                } else {
                                    float[] elem = new float[2];
                                    for (int j = firstIdx; j < lastIdx; j++) {
                                        elem[0] = otherElements[2 * j];
                                        elem[1] = otherElements[2 * j + 1];
                                        elem = FComplex.mult(alpha, elem);
                                        elements[2 * j] += elem[0];
                                        elements[2 * j + 1] += elem[1];
                                    }
                                }
                            } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.mult) { // x[i] = x[i] * y[i]
                                float[] elem = new float[2];
                                float[] otherElem = new float[2];
                                for (int j = firstIdx; j < lastIdx; j++) {
                                    otherElem[0] = otherElements[2 * j];
                                    otherElem[1] = otherElements[2 * j + 1];
                                    elem[0] = elements[2 * j];
                                    elem[1] = elements[2 * j + 1];
                                    elem = FComplex.mult(elem, otherElem);
                                    elements[2 * j] = elem[0];
                                    elements[2 * j + 1] = elem[1];
                                }
                            } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.div) { // x[i] = x[i] /  y[i]
                                float[] elem = new float[2];
                                float[] otherElem = new float[2];
                                for (int j = firstIdx; j < lastIdx; j++) {
                                    otherElem[0] = otherElements[2 * j];
                                    otherElem[1] = otherElements[2 * j + 1];
                                    elem[0] = elements[2 * j];
                                    elem[1] = elements[2 * j + 1];
                                    elem = FComplex.div(elem, otherElem);
                                    elements[2 * j] = elem[0];
                                    elements[2 * j + 1] = elem[1];
                                }
                            } else {
                                float[] elem = new float[2];
                                float[] otherElem = new float[2];
                                for (int j = firstIdx; j < lastIdx; j++) {
                                    otherElem[0] = otherElements[2 * j];
                                    otherElem[1] = otherElements[2 * j + 1];
                                    elem[0] = elements[2 * j];
                                    elem[1] = elements[2 * j + 1];
                                    elem = function.apply(elem, otherElem);
                                    elements[2 * j] = elem[0];
                                    elements[2 * j + 1] = elem[1];
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                if (function instanceof cern.jet.math.tfcomplex.FComplexPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
                    final float alpha[] = ((cern.jet.math.tfcomplex.FComplexPlusMultSecond) function).multiplicator;
                    if (alpha[0] == 1 && alpha[1] == 0) {
                        for (int j = 0; j < dlength; j++) {
                            elements[2 * j] += otherElements[2 * j];
                            elements[2 * j + 1] += otherElements[2 * j + 1];
                        }
                    } else {
                        float[] elem = new float[2];
                        for (int j = 0; j < dlength; j++) {
                            elem[0] = otherElements[2 * j];
                            elem[1] = otherElements[2 * j + 1];
                            elem = FComplex.mult(alpha, elem);
                            elements[2 * j] += elem[0];
                            elements[2 * j + 1] += elem[1];
                        }
                    }
                } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.mult) { // x[i] = x[i] * y[i]
                    float[] elem = new float[2];
                    float[] otherElem = new float[2];
                    for (int j = 0; j < dlength; j++) {
                        otherElem[0] = otherElements[2 * j];
                        otherElem[1] = otherElements[2 * j + 1];
                        elem[0] = elements[2 * j];
                        elem[1] = elements[2 * j + 1];
                        elem = FComplex.mult(elem, otherElem);
                        elements[2 * j] = elem[0];
                        elements[2 * j + 1] = elem[1];
                    }
                } else if (function == cern.jet.math.tfcomplex.FComplexFunctions.div) { // x[i] = x[i] /  y[i]
                    float[] elem = new float[2];
                    float[] otherElem = new float[2];
                    for (int j = 0; j < dlength; j++) {
                        otherElem[0] = otherElements[2 * j];
                        otherElem[1] = otherElements[2 * j + 1];
                        elem[0] = elements[2 * j];
                        elem[1] = elements[2 * j + 1];
                        elem = FComplex.div(elem, otherElem);
                        elements[2 * j] = elem[0];
                        elements[2 * j + 1] = elem[1];
                    }
                } else {
                    float[] elem = new float[2];
                    float[] otherElem = new float[2];
                    for (int j = 0; j < dlength; j++) {
                        otherElem[0] = otherElements[2 * j];
                        otherElem[1] = otherElements[2 * j + 1];
                        elem[0] = elements[2 * j];
                        elem[1] = elements[2 * j + 1];
                        elem = function.apply(elem, otherElem);
                        elements[2 * j] = elem[0];
                        elements[2 * j + 1] = elem[1];
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
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? dlength : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        for (int i = firstIdx; i < lastIdx; i++) {
                            if (elements[2 * i] != 0 || elements[2 * i + 1] != 0)
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
            for (int i = 0; i < dlength; i++) {
                if (elements[2 * i] != 0 || elements[2 * i + 1] != 0)
                    cardinality++;
            }
        }
        return cardinality;
    }

    public float[] elements() {
        return elements;
    }

    public boolean equals(float[] value) {
        float epsilon = cern.colt.matrix.tfcomplex.algo.FComplexProperty.DEFAULT.tolerance();
        float[] x = new float[2];
        float[] diff = new float[2];
        for (int i = 0; i < dlength; i++) {
            x[0] = elements[2 * i];
            x[1] = elements[2 * i + 1];
            diff[0] = Math.abs(value[0] - x[0]);
            diff[1] = Math.abs(value[1] - x[1]);
            if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                    && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                    || (FComplex.isEqual(value, x, epsilon))) {
                diff[0] = 0;
                diff[1] = 0;
            }
            if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                return false;
            }
        }
        return true;
    }

    public boolean equals(Object obj) {
        if (obj instanceof DiagonalFComplexMatrix2D) {
            DiagonalFComplexMatrix2D other = (DiagonalFComplexMatrix2D) obj;
            float epsilon = cern.colt.matrix.tfcomplex.algo.FComplexProperty.DEFAULT.tolerance();
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
            float[] otherElements = other.elements;
            float[] x = new float[2];
            float[] value = new float[2];
            float[] diff = new float[2];
            for (int i = 0; i < dlength; i++) {
                x[0] = elements[2 * i];
                x[1] = elements[2 * i + 1];
                value[0] = otherElements[2 * i];
                value[1] = otherElements[2 * i + 1];
                diff[0] = Math.abs(value[0] - x[0]);
                diff[1] = Math.abs(value[1] - x[1]);
                if (((diff[0] != diff[0]) || (diff[1] != diff[1]))
                        && ((((value[0] != value[0]) || (value[1] != value[1])) && ((x[0] != x[0]) || (x[1] != x[1]))))
                        || (FComplex.isEqual(value, x, epsilon))) {
                    diff[0] = 0;
                    diff[1] = 0;
                }
                if ((diff[0] > epsilon) || (diff[1] > epsilon)) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(obj);
        }
    }

    public FComplexMatrix2D forEachNonZero(final cern.colt.function.tfcomplex.IntIntFComplexFunction function) {
        float[] value = new float[2];
        for (int i = 0; i < dlength; i++) {
            value[0] = elements[2 * i];
            value[1] = elements[2 * i + 1];
            if (value[0] != 0 || value[1] != 0) {
                value = function.apply(i, i, value);
                elements[2 * i] = value[0];
                elements[2 * i + 1] = value[1];
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

    public float[] getQuick(int row, int column) {
        if (dindex >= 0) {
            if (column < dindex) {
                return new float[2];
            } else {
                if ((row < dlength) && (row + dindex == column)) {
                    return new float[] { elements[2 * row], elements[2 * row + 1] };
                } else {
                    return new float[2];
                }
            }
        } else {
            if (row < -dindex) {
                return new float[2];
            } else {
                if ((column < dlength) && (row + dindex == column)) {
                    return new float[] { elements[2 * column], elements[2 * column + 1] };
                } else {
                    return new float[2];
                }
            }
        }
    }

    public FComplexMatrix2D like(int rows, int columns) {
        return new SparseFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix1D like1D(int size) {
        return new SparseFComplexMatrix1D(size);
    }

    public void setQuick(int row, int column, float[] value) {
        if (dindex >= 0) {
            if (column < dindex) {
                //do nothing
            } else {
                if ((row < dlength) && (row + dindex == column)) {
                    elements[2 * row] = value[0];
                    elements[2 * row + 1] = value[1];
                } else {
                    // do nothing
                }
            }
        } else {
            if (row < -dindex) {
                //do nothing
            } else {
                if ((column < dlength) && (row + dindex == column)) {
                    elements[2 * column] = value[0];
                    elements[2 * column + 1] = value[1];
                } else {
                    //do nothing;
                }
            }
        }
    }

    public void setQuick(int row, int column, float re, float im) {
        if (dindex >= 0) {
            if (column < dindex) {
                //do nothing
            } else {
                if ((row < dlength) && (row + dindex == column)) {
                    elements[2 * row] = re;
                    elements[2 * row + 1] = im;
                } else {
                    // do nothing
                }
            }
        } else {
            if (row < -dindex) {
                //do nothing
            } else {
                if ((column < dlength) && (row + dindex == column)) {
                    elements[2 * column] = re;
                    elements[2 * column + 1] = im;
                } else {
                    //do nothing;
                }
            }
        }
    }

    public FComplexMatrix1D zMult(FComplexMatrix1D y, FComplexMatrix1D z, float[] alpha, float[] beta,
            final boolean transposeA) {
        int rowsA = rows;
        int columnsA = columns;
        if (transposeA) {
            rowsA = columns;
            columnsA = rows;
        }

        boolean ignore = (z == null);
        if (z == null)
            z = new DenseFComplexMatrix1D(rowsA);

        if (!(this.isNoView && y instanceof DenseFComplexMatrix1D && z instanceof DenseFComplexMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        if ((!ignore) && !((beta[0] == 1) && (beta[1] == 0)))
            z.assign(cern.jet.math.tfcomplex.FComplexFunctions.mult(beta));

        DenseFComplexMatrix1D zz = (DenseFComplexMatrix1D) z;
        final float[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) z.index(0);

        DenseFComplexMatrix1D yy = (DenseFComplexMatrix1D) y;
        final float[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) y.index(0);

        if (elementsY == null || elementsZ == null)
            throw new InternalError();
        float[] elemA = new float[2];
        float[] elemY = new float[2];
        if (!transposeA) {
            if (dindex >= 0) {
                for (int i = 0; i < dlength; i++) {
                    elemA[0] = elements[2 * i];
                    elemA[1] = elements[2 * i + 1];
                    elemY[0] = elementsY[2 * dindex + zeroY + strideY * i];
                    elemY[1] = elementsY[2 * dindex + zeroY + strideY * i + 1];
                    elemA = FComplex.mult(elemA, elemY);
                    elemA = FComplex.mult(alpha, elemA);
                    elementsZ[zeroZ + strideZ * i] += elemA[0];
                    elementsZ[zeroZ + strideZ * i + 1] += elemA[1];
                }
            } else {
                for (int i = 0; i < dlength; i++) {
                    elemA[0] = elements[2 * i];
                    elemA[1] = elements[2 * i + 1];
                    elemY[0] = elementsY[zeroY + strideY * i];
                    elemY[1] = elementsY[zeroY + strideY * i + 1];
                    elemA = FComplex.mult(elemA, elemY);
                    elemA = FComplex.mult(alpha, elemA);
                    elementsZ[-2 * dindex + zeroZ + strideZ * i] += elemA[0];
                    elementsZ[-2 * dindex + zeroZ + strideZ * i + 1] += elemA[1];
                }
            }
        } else {
            if (dindex >= 0) {
                for (int i = 0; i < dlength; i++) {
                    elemA[0] = elements[2 * i];
                    elemA[1] = -elements[2 * i + 1];
                    elemY[0] = elementsY[zeroY + strideY * i];
                    elemY[1] = elementsY[zeroY + strideY * i + 1];
                    elemA = FComplex.mult(elemA, elemY);
                    elemA = FComplex.mult(alpha, elemA);
                    elementsZ[2 * dindex + zeroZ + strideZ * i] += elemA[0];
                    elementsZ[2 * dindex + zeroZ + strideZ * i + 1] += elemA[1];
                }
            } else {
                for (int i = 0; i < dlength; i++) {
                    elemA[0] = elements[2 * i];
                    elemA[1] = -elements[2 * i + 1];
                    elemY[0] = elementsY[-2 * dindex + zeroY + strideY * i];
                    elemY[1] = elementsY[-2 * dindex + zeroY + strideY * i + 1];
                    elemA = FComplex.mult(elemA, elemY);
                    elemA = FComplex.mult(alpha, elemA);
                    elementsZ[zeroZ + strideZ * i] += elemA[0];
                    elementsZ[zeroZ + strideZ * i + 1] += elemA[1];
                }
            }

        }
        return z;
    }

    protected FComplexMatrix2D getContent() {
        return this;
    }
}
