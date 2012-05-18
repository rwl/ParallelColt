/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.impl;

import java.util.Arrays;
import java.util.concurrent.Future;

import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse row-compressed 2-d matrix holding <tt>complex</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally uses the standard sparse row-compressed format<br>
 * Note that this implementation is not synchronized.
 * <p>
 * Cells that
 * <ul>
 * <li>are never set to non-zero values do not use any memory.
 * <li>switch from zero to non-zero state do use memory.
 * <li>switch back from non-zero to zero state also do use memory. Their memory
 * is <i>not</i> automatically reclaimed (because of the lists vs. arrays).
 * Reclamation can be triggered via {@link #trimToSize()}.
 * </ul>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseRCFComplexMatrix2D extends WrapperFComplexMatrix2D {
    private static final long serialVersionUID = 1L;

    private static int searchFromTo(int[] list, int key, int from, int to) {
        while (from <= to) {
            if (list[from] == key) {
                return from;
            } else {
                from++;
                continue;
            }
        }
        return -(from + 1); // key not found.
    }

    /*
     * The elements of the matrix.
     */
    protected int[] rowPointers;

    protected int[] columnIndexes;

    protected float[] values;

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
    public SparseRCFComplexMatrix2D(float[][] values) {
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
    public SparseRCFComplexMatrix2D(int rows, int columns) {
        this(rows, columns, (int) Math.min(10l * rows, Integer.MAX_VALUE));
    }

    /**
     * Constructs a matrix with a given number of rows and columns. All entries
     * are initially <tt>0</tt>.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param nzmax
     *            maximum number of nonzero elements
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseRCFComplexMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        columnIndexes = new int[nzmax];
        values = new float[2 * nzmax];
        rowPointers = new int[rows + 1];
    }

    /**
     * Constructs a matrix with indexes given in the coordinate format and
     * single value.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param rowIndexes
     *            row indexes
     * @param columnIndexes
     *            column indexes
     * @param re
     *            real part of numerical value
     * @param im
     *            imaginary part of numerical value
     * 
     * @param removeDuplicates
     *            if true, then duplicates (if any) are removed
     */
    public SparseRCFComplexMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, float re, float im,
            boolean removeDuplicates) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        if (rowIndexes.length != columnIndexes.length) {
            throw new IllegalArgumentException("rowIndexes.length != columnIndexes.length");
        }
        if (re == 0 && im == 0) {
            throw new IllegalArgumentException("value cannot be 0");
        }

        int nz = Math.max(rowIndexes.length, 1);
        this.columnIndexes = new int[nz];
        this.values = new float[2 * nz];
        this.rowPointers = new int[rows + 1];
        int[] w = new int[rows];
        int r;
        for (int k = 0; k < nz; k++) {
            w[rowIndexes[k]]++;
        }
        cumsum(this.rowPointers, w, rows);
        for (int k = 0; k < nz; k++) {
            this.columnIndexes[r = w[rowIndexes[k]]++] = columnIndexes[k];
            this.values[2 * r] = re;
            this.values[2 * r + 1] = im;
        }
        if (removeDuplicates) {
            removeDuplicates();
        }
    }

    /**
     * Constructs a matrix with indexes and values given in the coordinate
     * format.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param rowIndexes
     *            row indexes
     * @param columnIndexes
     *            column indexes
     * @param values
     *            numerical values
     * @param removeDuplicates
     *            if true, then duplicates (if any) are removed
     * @param removeZeroes
     *            if true, then zeroes (if any) are removed
     */
    public SparseRCFComplexMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, float[] values,
            boolean removeDuplicates, boolean removeZeroes) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        if (rowIndexes.length != columnIndexes.length) {
            throw new IllegalArgumentException("rowIndexes.length != columnIndexes.length");
        } else if (2 * rowIndexes.length != values.length) {
            throw new IllegalArgumentException("2 * rowIndexes.length != values.length");
        }
        int nz = Math.max(rowIndexes.length, 1);
        this.columnIndexes = new int[nz];
        this.values = new float[2 * nz];
        this.rowPointers = new int[rows + 1];
        int[] w = new int[rows];
        int r;
        for (int k = 0; k < nz; k++) {
            w[rowIndexes[k]]++;
        }
        cumsum(this.rowPointers, w, rows);
        for (int k = 0; k < nz; k++) {
            this.columnIndexes[r = w[rowIndexes[k]]++] = columnIndexes[k];
            this.values[2 * r] = values[2 * k];
            this.values[2 * r + 1] = values[2 * k + 1];
        }
        if (removeZeroes) {
            removeZeroes();
        }
        if (removeDuplicates) {
            removeDuplicates();
        }
    }

    /**
     * Constructs a matrix with given parameters. The arrays are not copied.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param rowPointers
     *            row pointers
     * @param columnIndexes
     *            column indexes
     * @param values
     *            numerical values
     */
    public SparseRCFComplexMatrix2D(int rows, int columns, int[] rowPointers, int[] columnIndexes, float[] values) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        if (rowPointers.length != rows + 1) {
            throw new IllegalArgumentException("rowPointers.length != rows + 1");
        }
        if (2 * columnIndexes.length != values.length) {
            throw new IllegalArgumentException("2 * columnIndexes.length != values.length");
        }
        this.rowPointers = rowPointers;
        this.columnIndexes = columnIndexes;
        this.values = values;
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
            int nz = cardinality();
            float[] elem = new float[2];
            for (int j = 0; j < nz; j++) {
                elem[0] = values[2 * j];
                elem[1] = values[2 * j + 1];
                elem = FComplex.mult(elem, alpha);
                values[2 * j] = elem[0];
                values[2 * j + 1] = elem[1];
            }
        } else {
            forEachNonZero(new cern.colt.function.tfcomplex.IntIntFComplexFunction() {
                public float[] apply(int i, int j, float[] value) {
                    return function.apply(value);
                }
            });
        }
        return this;
    }

    public FComplexMatrix2D assign(float re, float im) {
        if (re == 0 && im == 0) {
            Arrays.fill(rowPointers, 0);
            Arrays.fill(columnIndexes, 0);
            Arrays.fill(values, 0);
        } else {
            int nnz = cardinality();
            for (int i = 0; i < nnz; i++) {
                values[2 * i] = re;
                values[2 * i + 1] = im;
            }
        }
        return this;
    }

    public FComplexMatrix2D assign(FComplexMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof SparseRCFComplexMatrix2D) {
            SparseRCFComplexMatrix2D other = (SparseRCFComplexMatrix2D) source;
            System.arraycopy(other.rowPointers, 0, rowPointers, 0, rows + 1);
            int nzmax = other.columnIndexes.length;
            if (columnIndexes.length < nzmax) {
                columnIndexes = new int[nzmax];
                values = new float[2 * nzmax];
            }
            System.arraycopy(other.columnIndexes, 0, columnIndexes, 0, nzmax);
            System.arraycopy(other.values, 0, values, 0, other.values.length);
        } else if (source instanceof SparseCCFComplexMatrix2D) {
            SparseCCFComplexMatrix2D other = ((SparseCCFComplexMatrix2D) source).getConjugateTranspose();
            rowPointers = other.getColumnPointers();
            columnIndexes = other.getRowIndexes();
            values = other.getValues();
        } else {
            assign(0, 0);
            source.forEachNonZero(new cern.colt.function.tfcomplex.IntIntFComplexFunction() {
                public float[] apply(int i, int j, float[] value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
        }
        return this;
    }

    public FComplexMatrix2D assign(final FComplexMatrix2D y,
            cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction function) {
        checkShape(y);
        if ((y instanceof SparseRCFComplexMatrix2D) && (function == cern.jet.math.tfcomplex.FComplexFunctions.plus)) { // x[i] = x[i] + y[i] 
            SparseRCFComplexMatrix2D yy = (SparseRCFComplexMatrix2D) y;

            final int[] rowPointersY = yy.rowPointers;
            final int[] columnIndexesY = yy.columnIndexes;
            final float[] valuesY = yy.values;

            final int[] rowPointersC = new int[rows + 1];
            int cnz = Math.max(columnIndexes.length, (int) Math.min(Integer.MAX_VALUE, (long) rowPointers[rows]
                    + (long) rowPointersY[rows]));
            final int[] columnIndexesC = new int[cnz];
            final float[] valuesC = new float[2 * cnz];
            int nrow = rows;
            int ncol = columns;
            int nzmax = cnz;
            if (function == cern.jet.math.tfcomplex.FComplexFunctions.plus) { // x[i] = x[i] + y[i]
                int kc = 0;
                rowPointersC[0] = kc;
                int j1, j2;
                for (int i = 0; i < nrow; i++) {
                    int ka = rowPointers[i];
                    int kb = rowPointersY[i];
                    int kamax = rowPointers[i + 1] - 1;
                    int kbmax = rowPointersY[i + 1] - 1;
                    while (ka <= kamax || kb <= kbmax) {
                        if (ka <= kamax) {
                            j1 = columnIndexes[ka];
                        } else {
                            j1 = ncol + 1;
                        }
                        if (kb <= kbmax) {
                            j2 = columnIndexesY[kb];
                        } else {
                            j2 = ncol + 1;
                        }
                        if (j1 == j2) {
                            valuesC[2 * kc] = values[2 * ka] + valuesY[2 * kb];
                            valuesC[2 * kc + 1] = values[2 * ka + 1] + valuesY[2 * kb + 1];
                            columnIndexesC[kc] = j1;
                            ka++;
                            kb++;
                            kc++;
                        } else if (j1 < j2) {
                            columnIndexesC[kc] = j1;
                            valuesC[2 * kc] = values[2 * ka];
                            valuesC[2 * kc + 1] = values[2 * ka + 1];
                            ka++;
                            kc++;
                        } else if (j1 > j2) {
                            columnIndexesC[kc] = j2;
                            valuesC[2 * kc] = valuesY[2 * kb];
                            valuesC[2 * kc + 1] = valuesY[2 * kb + 1];
                            kb++;
                            kc++;
                        }
                        if (kc >= nzmax) {
                            throw new IllegalArgumentException("The number of elements in C exceeds nzmax");
                        }
                    }
                    rowPointersC[i + 1] = kc;
                }
                this.rowPointers = rowPointersC;
                this.columnIndexes = columnIndexesC;
                this.values = valuesC;
                return this;
            }
        }

        if (function instanceof cern.jet.math.tfcomplex.FComplexPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final float[] alpha = ((cern.jet.math.tfcomplex.FComplexPlusMultSecond) function).multiplicator;
            if (alpha[0] == 0 && alpha[1] == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tfcomplex.IntIntFComplexFunction() {
                public float[] apply(int i, int j, float[] value) {
                    setQuick(i, j, FComplex.plus(getQuick(i, j), FComplex.mult(alpha, value)));
                    return value;
                }
            });
            return this;
        }

        if (function instanceof cern.jet.math.tfcomplex.FComplexPlusMultFirst) { // x[i] = alpha*x[i] + y[i]
            final float[] alpha = ((cern.jet.math.tfcomplex.FComplexPlusMultFirst) function).multiplicator;
            if (alpha[0] == 0 && alpha[1] == 0)
                return assign(y);
            y.forEachNonZero(new cern.colt.function.tfcomplex.IntIntFComplexFunction() {
                public float[] apply(int i, int j, float[] value) {
                    setQuick(i, j, FComplex.plus(FComplex.mult(alpha, getQuick(i, j)), value));
                    return value;
                }
            });
            return this;
        }

        if (function == cern.jet.math.tfcomplex.FComplexFunctions.mult) { // x[i] = x[i] * y[i]
            float[] elem = new float[2];
            for (int i = 0; i < rows; i++) {
                int high = rowPointers[i + 1];
                for (int k = rowPointers[i]; k < high; k++) {
                    int j = columnIndexes[k];
                    elem[0] = values[2 * k];
                    elem[1] = values[2 * k + 1];
                    elem = FComplex.mult(elem, y.getQuick(i, j));
                    values[2 * k] = elem[0];
                    values[2 * k + 1] = elem[1];
                    if (values[2 * k] == 0 && values[2 * k + 1] == 0)
                        remove(i, j);
                }
            }
            return this;
        }

        if (function == cern.jet.math.tfcomplex.FComplexFunctions.div) { // x[i] = x[i] / y[i]
            float[] elem = new float[2];
            for (int i = 0; i < rows; i++) {
                int high = rowPointers[i + 1];
                for (int k = rowPointers[i]; k < high; k++) {
                    int j = columnIndexes[k];
                    elem[0] = values[2 * k];
                    elem[1] = values[2 * k + 1];
                    elem = FComplex.div(elem, y.getQuick(i, j));
                    values[2 * k] = elem[0];
                    values[2 * k + 1] = elem[1];
                    if (values[2 * k] == 0 && values[2 * k + 1] == 0)
                        remove(i, j);
                }
            }
            return this;
        }
        return super.assign(y, function);

    }

    public int cardinality() {
        return rowPointers[rows];
    }

    public FComplexMatrix2D forEachNonZero(final cern.colt.function.tfcomplex.IntIntFComplexFunction function) {
        float[] value = new float[2];
        for (int i = 0; i < rows; i++) {
            int high = rowPointers[i + 1];
            for (int k = rowPointers[i]; k < high; k++) {
                int j = columnIndexes[k];
                value[0] = values[2 * k];
                value[1] = values[2 * k + 1];
                float[] r = function.apply(i, j, value);
                if (r[0] != value[0] || r[1] != value[1]) {
                    values[2 * k] = r[0];
                    values[2 * k + 1] = r[1];
                }
            }
        }
        return this;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a column-compressed form. This method creates a new object (not a view),
     * so changes in the returned matrix are NOT reflected in this matrix.
     * 
     * @return this matrix in a column-compressed form
     */
    public SparseCCFComplexMatrix2D getColumnCompressed() {
        SparseRCFComplexMatrix2D tr = getConjugateTranspose();
        SparseCCFComplexMatrix2D cc = new SparseCCFComplexMatrix2D(rows, columns);
        cc.rowIndexes = tr.columnIndexes;
        cc.columnPointers = tr.rowPointers;
        cc.values = tr.values;
        return cc;
    }

    /**
     * Returns column indexes
     * 
     * @return column indexes
     */
    public int[] getColumnIndexes() {
        return columnIndexes;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a dense form. This method creates a new object (not a view), so changes
     * in the returned matrix are NOT reflected in this matrix.
     * 
     * @return this matrix in a dense form
     */
    public DenseFComplexMatrix2D getDense() {
        final DenseFComplexMatrix2D dense = new DenseFComplexMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tfcomplex.IntIntFComplexFunction() {
            public float[] apply(int i, int j, float[] value) {
                dense.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return dense;
    }

    public synchronized float[] getQuick(int row, int column) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        float[] v = new float[2];
        if (k >= 0) {
            v[0] = values[2 * k];
            v[1] = values[2 * k + 1];
        }
        return v;
    }

    /**
     * Returns row pointers
     * 
     * @return row pointers
     */
    public int[] getRowPointers() {
        return rowPointers;
    }

    public SparseRCFComplexMatrix2D getConjugateTranspose() {
        int nnz = rowPointers[rows];
        int[] w = new int[columns];
        int[] rowPointersT = new int[columns + 1];
        int[] columnIndexesT = new int[nnz];
        float[] valuesT = new float[2 * nnz];

        for (int p = 0; p < nnz; p++) {
            w[columnIndexes[p]]++;
        }
        cumsum(rowPointersT, w, columns);
        int q;
        for (int j = 0; j < rows; j++) {
            int high = rowPointers[j + 1];
            for (int p = rowPointers[j]; p < high; p++) {
                columnIndexesT[q = w[columnIndexes[p]]++] = j;
                valuesT[2 * q] = values[2 * p];
                valuesT[2 * q + 1] = -values[2 * p + 1];
            }
        }
        SparseRCFComplexMatrix2D T = new SparseRCFComplexMatrix2D(columns, rows);
        T.rowPointers = rowPointersT;
        T.columnIndexes = columnIndexesT;
        T.values = valuesT;
        return T;
    }

    /**
     * Returns numerical values
     * 
     * @return numerical values
     */
    public float[] getValues() {
        return values;
    }

    public FComplexMatrix2D like(int rows, int columns) {
        return new SparseRCFComplexMatrix2D(rows, columns);
    }

    public FComplexMatrix1D like1D(int size) {
        return new SparseFComplexMatrix1D(size);
    }

    /**
     * Removes (sums) duplicate entries (if any}
     */
    public void removeDuplicates() {
        int nz = 0;
        int q, i;
        int[] w = new int[columns]; /* get workspace */
        for (i = 0; i < columns; i++)
            w[i] = -1; /* column i not yet seen */
        for (int j = 0; j < rows; j++) {
            q = nz; /* row j will start at q */
            for (int p = rowPointers[j]; p < rowPointers[j + 1]; p++) {
                i = columnIndexes[p]; /* A(i,j) is nonzero */
                if (w[i] >= q) {
                    values[2 * w[i]] += values[2 * p]; /* A(i,j) is a duplicate */
                    values[2 * w[i] + 1] += values[2 * p + 1];
                } else {
                    w[i] = nz; /* record where column i occurs */
                    columnIndexes[nz] = i; /* keep A(i,j) */
                    values[2 * nz] = values[2 * p];
                    values[2 * nz + 1] = values[2 * p + 1];
                    nz++;
                }
            }
            rowPointers[j] = q; /* record start of row j */
        }
        rowPointers[rows] = nz; /* finalize A */
    }

    /**
     * Removes zero entries (if any)
     */
    public void removeZeroes() {
        int nz = 0;
        float eps = (float) Math.pow(2, -23);
        float[] elem = new float[2];
        for (int j = 0; j < rows; j++) {
            int p = rowPointers[j]; /* get current location of row j */
            rowPointers[j] = nz; /* record new location of row j */
            for (; p < rowPointers[j + 1]; p++) {
                elem[0] = values[2 * p];
                elem[1] = values[2 * p + 1];
                if (FComplex.abs(elem) > eps) {
                    values[2 * nz] = values[2 * p]; /* keep A(i,j) */
                    values[2 * nz + 1] = values[2 * p + 1];
                    columnIndexes[nz++] = columnIndexes[p];
                }
            }
        }
        rowPointers[rows] = nz; /* finalize A */
    }

    public synchronized void setQuick(int row, int column, float[] value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        if (k >= 0) { // found
            if (value[0] == 0 && value[1] == 0)
                remove(row, k);
            else {
                values[2 * k] = value[0];
                values[2 * k + 1] = value[1];
            }
            return;
        }

        if (value[0] != 0 || value[1] != 0) {
            k = -k - 1;
            insert(row, column, k, value);
        }
    }

    public synchronized void setQuick(int row, int column, float re, float im) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        if (k >= 0) { // found
            if (re == 0 && im == 0)
                remove(row, k);
            else {
                values[2 * k] = re;
                values[2 * k + 1] = im;
            }
            return;
        }

        if (re != 0 || im != 0) {
            k = -k - 1;
            insert(row, column, k, re, im);
        }
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(rows).append(" x ").append(columns).append(" sparse matrix, nnz = ").append(cardinality())
                .append('\n');
        for (int i = 0; i < rows; i++) {
            int high = rowPointers[i + 1];
            for (int j = rowPointers[i]; j < high; j++) {
                if (values[2 * j + 1] > 0) {
                    builder.append('(').append(i).append(',').append(columnIndexes[j]).append(')').append('\t').append(
                            values[2 * j]).append('+').append(values[2 * j + 1]).append('i').append('\n');
                } else if (values[2 * j + 1] == 0) {
                    builder.append('(').append(i).append(',').append(columnIndexes[j]).append(')').append('\t').append(
                            values[2 * j]).append('\n');
                } else {
                    builder.append('(').append(i).append(',').append(columnIndexes[j]).append(')').append('\t').append(
                            values[2 * j]).append('-').append(values[2 * j + 1]).append('i').append('\n');
                }
            }
        }
        return builder.toString();
    }

    public void trimToSize() {
        realloc(0);
    }

    public FComplexMatrix1D zMult(FComplexMatrix1D y, FComplexMatrix1D z, final float[] alpha, final float[] beta,
            final boolean transposeA) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;

        boolean ignore = (z == null || !transposeA);
        if (z == null)
            z = new DenseFComplexMatrix1D(rowsA);

        if (!(y instanceof DenseFComplexMatrix1D && z instanceof DenseFComplexMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        DenseFComplexMatrix1D zz = (DenseFComplexMatrix1D) z;
        final float[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) z.index(0);

        DenseFComplexMatrix1D yy = (DenseFComplexMatrix1D) y;
        final float[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) y.index(0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();

        if (transposeA) {
            if ((!ignore) && !((beta[0] == 1) && (beta[1] == 0)))
                z.assign(cern.jet.math.tfcomplex.FComplexFunctions.mult(beta));

            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = 2;
                Future<?>[] futures = new Future[nthreads];
                final float[] result = new float[2 * rowsA];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    final int threadID = j;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            float[] yElem = new float[2];
                            float[] val = new float[2];
                            if (threadID == 0) {
                                for (int i = firstRow; i < lastRow; i++) {
                                    int high = rowPointers[i + 1];
                                    yElem[0] = elementsY[zeroY + strideY * i];
                                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                                    yElem = FComplex.mult(alpha, yElem);
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = columnIndexes[k];
                                        val[0] = values[2 * k];
                                        val[1] = -values[2 * k + 1];
                                        val = FComplex.mult(val, yElem);
                                        elementsZ[zeroZ + strideZ * j] += val[0];
                                        elementsZ[zeroZ + strideZ * j + 1] += val[1];
                                    }
                                }
                            } else {
                                for (int i = firstRow; i < lastRow; i++) {
                                    int high = rowPointers[i + 1];
                                    yElem[0] = elementsY[zeroY + strideY * i];
                                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                                    yElem = FComplex.mult(alpha, yElem);
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = columnIndexes[k];
                                        val[0] = values[2 * k];
                                        val[1] = -values[2 * k + 1];
                                        val = FComplex.mult(val, yElem);
                                        result[2 * j] += val[0];
                                        result[2 * j + 1] += val[1];
                                    }
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
                for (int j = 0; j < rowsA; j++) {
                    elementsZ[zeroZ + j * strideZ] += result[2 * j];
                    elementsZ[zeroZ + j * strideZ + 1] += result[2 * j + 1];
                }
            } else {
                float[] yElem = new float[2];
                float[] val = new float[2];
                for (int i = 0; i < rows; i++) {
                    int high = rowPointers[i + 1];
                    yElem[0] = elementsY[zeroY + strideY * i];
                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                    yElem = FComplex.mult(alpha, yElem);
                    for (int k = rowPointers[i]; k < high; k++) {
                        int j = columnIndexes[k];
                        val[0] = values[2 * k];
                        val[1] = -values[2 * k + 1];
                        val = FComplex.mult(val, yElem);
                        elementsZ[zeroZ + strideZ * j] += val[0];
                        elementsZ[zeroZ + strideZ * j + 1] += val[1];
                    }
                }
            }

            return z;
        }

        if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int zidx = zeroZ + firstRow * strideZ;
                        float[] yElem = new float[2];
                        float[] val = new float[2];
                        if (beta[0] == 0.0 && beta[1] == 0) {
                            for (int i = firstRow; i < lastRow; i++) {
                                float[] sum = new float[2];
                                int high = rowPointers[i + 1];
                                for (int k = rowPointers[i]; k < high; k++) {
                                    yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                                    yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                                    val[0] = values[2 * k];
                                    val[1] = values[2 * k + 1];
                                    sum = FComplex.plus(sum, FComplex.mult(val, yElem));
                                }
                                sum = FComplex.mult(alpha, sum);
                                elementsZ[zidx] = sum[0];
                                elementsZ[zidx + 1] = sum[1];
                                zidx += strideZ;
                            }
                        } else {
                            float[] zElem = new float[2];
                            for (int i = firstRow; i < lastRow; i++) {
                                float[] sum = new float[2];
                                int high = rowPointers[i + 1];
                                for (int k = rowPointers[i]; k < high; k++) {
                                    yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                                    yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                                    val[0] = values[2 * k];
                                    val[1] = values[2 * k + 1];
                                    sum = FComplex.plus(sum, FComplex.mult(val, yElem));
                                }
                                sum = FComplex.mult(alpha, sum);
                                zElem[0] = elementsZ[zidx];
                                zElem[1] = elementsZ[zidx + 1];
                                zElem = FComplex.mult(beta, zElem);
                                elementsZ[zidx] = sum[0] + zElem[0];
                                elementsZ[zidx + 1] = sum[1] + zElem[1];
                                zidx += strideZ;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int zidx = zeroZ;
            float[] yElem = new float[2];
            float[] val = new float[2];
            if (beta[0] == 0.0 && beta[1] == 0) {
                for (int i = 0; i < rows; i++) {
                    float[] sum = new float[2];
                    int high = rowPointers[i + 1];
                    for (int k = rowPointers[i]; k < high; k++) {
                        yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                        yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                        val[0] = values[2 * k];
                        val[1] = values[2 * k + 1];
                        sum = FComplex.plus(sum, FComplex.mult(val, yElem));
                    }
                    sum = FComplex.mult(alpha, sum);
                    elementsZ[zidx] = sum[0];
                    elementsZ[zidx + 1] = sum[1];
                    zidx += strideZ;
                }
            } else {
                float[] zElem = new float[2];
                for (int i = 0; i < rows; i++) {
                    float[] sum = new float[2];
                    int high = rowPointers[i + 1];
                    for (int k = rowPointers[i]; k < high; k++) {
                        yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                        yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                        val[0] = values[2 * k];
                        val[1] = values[2 * k + 1];
                        sum = FComplex.plus(sum, FComplex.mult(val, yElem));
                    }
                    sum = FComplex.mult(alpha, sum);
                    zElem[0] = elementsZ[zidx];
                    zElem[1] = elementsZ[zidx + 1];
                    zElem = FComplex.mult(beta, zElem);
                    elementsZ[zidx] = sum[0] + zElem[0];
                    elementsZ[zidx + 1] = sum[1] + zElem[1];
                    zidx += strideZ;
                }
            }
        }
        return z;
    }

    public FComplexMatrix2D zMult(FComplexMatrix2D B, FComplexMatrix2D C, final float[] alpha, float[] beta,
            final boolean transposeA, boolean transposeB) {
        int rowsA = rows;
        int columnsA = columns;
        if (transposeA) {
            rowsA = columns;
            columnsA = rows;
        }
        int rowsB = B.rows();
        int columnsB = B.columns();
        if (transposeB) {
            rowsB = B.columns();
            columnsB = B.rows();
        }
        int p = columnsB;
        boolean ignore = (C == null);
        if (C == null) {
            if (B instanceof SparseRCFComplexMatrix2D) {
                C = new SparseRCFComplexMatrix2D(rowsA, p, (rowsA * p));
            } else {
                C = new DenseFComplexMatrix2D(rowsA, p);
            }
        }

        if (rowsB != columnsA)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows() != rowsA || C.columns() != p)
            throw new IllegalArgumentException("Incompatible result matrix: " + toStringShort() + ", "
                    + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!ignore && !(beta[0] == 1.0 && beta[1] == 0)) {
            C.assign(cern.jet.math.tfcomplex.FComplexFunctions.mult(beta));
        }

        if ((B instanceof DenseFComplexMatrix2D) && (C instanceof DenseFComplexMatrix2D)) {
            SparseRCFComplexMatrix2D AA;
            if (transposeA) {
                AA = getConjugateTranspose();
            } else {
                AA = this;
            }
            DenseFComplexMatrix2D BB;
            if (transposeB) {
                BB = (DenseFComplexMatrix2D) B.getConjugateTranspose();
            } else {
                BB = (DenseFComplexMatrix2D) B;
            }

            DenseFComplexMatrix2D CC = (DenseFComplexMatrix2D) C;
            int[] rowPointersA = AA.rowPointers;
            int[] columnIndexesA = AA.columnIndexes;
            float[] valuesA = AA.values;
            float[] valA = new float[2];
            for (int ii = 0; ii < rowsA; ii++) {
                int highA = rowPointersA[ii + 1];
                for (int ka = rowPointersA[ii]; ka < highA; ka++) {
                    valA[0] = valuesA[2 * ka];
                    valA[1] = valuesA[2 * ka + 1];
                    float[] scal = FComplex.mult(alpha, valA);
                    int jj = columnIndexesA[ka];
                    CC.viewRow(ii).assign(BB.viewRow(jj), FComplexFunctions.plusMultSecond(scal));
                }
            }
        } else if ((B instanceof SparseRCFComplexMatrix2D) && (C instanceof SparseRCFComplexMatrix2D)) {
            SparseRCFComplexMatrix2D AA;
            SparseRCFComplexMatrix2D BB;
            SparseRCFComplexMatrix2D CC = (SparseRCFComplexMatrix2D) C;
            if (transposeA) {
                AA = getConjugateTranspose();
            } else {
                AA = this;
            }
            if (transposeB) {
                BB = ((SparseRCFComplexMatrix2D) B).getConjugateTranspose();
            } else {
                BB = (SparseRCFComplexMatrix2D) B;
            }

            int[] rowPointersA = AA.rowPointers;
            int[] columnIndexesA = AA.columnIndexes;
            float[] valuesA = AA.values;

            int[] rowPointersB = BB.rowPointers;
            int[] columnIndexesB = BB.columnIndexes;
            float[] valuesB = BB.values;

            int[] rowPointersC = CC.rowPointers;
            int[] columnIndexesC = CC.columnIndexes;
            float[] valuesC = CC.values;
            int nzmax = columnIndexesC.length;

            int[] iw = new int[columnsB + 1];
            for (int i = 0; i < iw.length; i++) {
                iw[i] = -1;
            }
            int len = -1;
            float[] valA = new float[2];
            float[] valB = new float[2];
            float[] valC = new float[2];
            for (int ii = 0; ii < rowsA; ii++) {
                int highA = rowPointersA[ii + 1];
                for (int ka = rowPointersA[ii]; ka < highA; ka++) {
                    valA[0] = valuesA[2 * ka];
                    valA[1] = valuesA[2 * ka + 1];
                    float[] scal = FComplex.mult(alpha, valA);
                    int jj = columnIndexesA[ka];
                    int highB = rowPointersB[jj + 1];
                    for (int kb = rowPointersB[jj]; kb < highB; kb++) {
                        int jcol = columnIndexesB[kb];
                        int jpos = iw[jcol];
                        if (jpos == -1) {
                            len++;
                            if (len >= nzmax) {
                                throw new IllegalArgumentException(
                                        "The max number of nonzero elements in C is too small.");
                            }
                            columnIndexesC[len] = jcol;
                            iw[jcol] = len;
                            valB[0] = valuesB[2 * kb];
                            valB[1] = valuesB[2 * kb + 1];
                            valB = FComplex.mult(scal, valB);
                            valuesC[2 * len] = valB[0];
                            valuesC[2 * len + 1] = valB[1];
                        } else {
                            valB[0] = valuesB[2 * kb];
                            valB[1] = valuesB[2 * kb + 1];
                            valB = FComplex.mult(scal, valB);
                            valuesC[2 * jpos] += valB[0];
                            valuesC[2 * jpos + 1] += valB[1];
                        }
                    }
                }
                for (int k = rowPointersC[ii]; k < len + 1; k++) {
                    iw[columnIndexesC[k]] = -1;
                }
                rowPointersC[ii + 1] = len + 1;

                //                int length = rowPointersC[ii + 1] - rowPointersC[ii];
                //                IntMatrix1D columnIndexesCPart = columnIndexesC.viewPart(rowPointersC[ii], length);
                //                int[] indexes = cern.colt.matrix.tint.algo.IntSorting.quickSort.sortIndex(columnIndexesCPart);
                //                Arrays.sort(columnIndexesCElements, rowPointersC[ii], rowPointersC[ii + 1]);
                //                FComplexMatrix1D valuesCPart = valuesC.viewPart(rowPointersC[ii], length).viewSelection(indexes);
                //                valuesC.viewPart(rowPointersC[ii], length).assign(valuesCPart);
            }
            //            CC.columnIndexes.elements((int[]) columnIndexesC.elements());
            //            CC.columnIndexes.setSize(columnIndexesSize);
            //            CC.values.elements((float[]) valuesC.elements());
            //            CC.values.setSize(columnIndexesSize);
        } else {
            if (transposeB) {
                B = B.getConjugateTranspose();
            }
            // cache views
            final FComplexMatrix1D[] Brows = new FComplexMatrix1D[columnsA];
            for (int i = columnsA; --i >= 0;)
                Brows[i] = B.viewRow(i);
            final FComplexMatrix1D[] Crows = new FComplexMatrix1D[rowsA];
            for (int i = rowsA; --i >= 0;)
                Crows[i] = C.viewRow(i);

            final cern.jet.math.tfcomplex.FComplexPlusMultSecond fun = cern.jet.math.tfcomplex.FComplexPlusMultSecond
                    .plusMult(new float[2]);

            final int[] columnIndexesA = columnIndexes;
            final float[] valuesA = values;
            float[] valA = new float[2];
            for (int i = rows; --i >= 0;) {
                int low = rowPointers[i];
                for (int k = rowPointers[i + 1]; --k >= low;) {
                    int j = columnIndexesA[k];
                    valA[0] = valuesA[2 * k];
                    valA[1] = valuesA[2 * k + 1];
                    fun.multiplicator = FComplex.mult(valA, alpha);
                    if (!transposeA)
                        Crows[i].assign(Brows[j], fun);
                    else
                        Crows[j].assign(Brows[i], fun);
                }
            }
        }
        return C;
    }

    private float cumsum(int[] p, int[] c, int n) {
        int nz = 0;
        float nz2 = 0;
        for (int k = 0; k < n; k++) {
            p[k] = nz;
            nz += c[k];
            nz2 += c[k];
            c[k] = p[k];
        }
        p[n] = nz;
        return (nz2);
    }

    private void realloc(int nzmax) {
        if (nzmax <= 0)
            nzmax = rowPointers[rows];
        int[] columnIndexesNew = new int[nzmax];
        int length = Math.min(nzmax, columnIndexes.length);
        System.arraycopy(columnIndexes, 0, columnIndexesNew, 0, length);
        columnIndexes = columnIndexesNew;
        float[] valuesNew = new float[2 * nzmax];
        length = Math.min(nzmax, values.length);
        System.arraycopy(values, 0, valuesNew, 0, length);
        values = valuesNew;
    }

    protected FComplexMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, float[] value) {
        IntArrayList columnIndexesList = new IntArrayList(columnIndexes);
        columnIndexesList.setSizeRaw(rowPointers[rows]);
        FloatArrayList valuesList = new FloatArrayList(values);
        valuesList.setSizeRaw(2 * rowPointers[rows]);
        columnIndexesList.beforeInsert(index, column);
        valuesList.beforeInsert(2 * index, value[0]);
        valuesList.beforeInsert(2 * index + 1, value[1]);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]++;
        columnIndexes = columnIndexesList.elements();
        values = valuesList.elements();
    }

    protected void insert(int row, int column, int index, float re, float im) {
        IntArrayList columnIndexesList = new IntArrayList(columnIndexes);
        columnIndexesList.setSizeRaw(rowPointers[rows]);
        FloatArrayList valuesList = new FloatArrayList(values);
        valuesList.setSizeRaw(2 * rowPointers[rows]);
        columnIndexesList.beforeInsert(index, column);
        valuesList.beforeInsert(2 * index, re);
        valuesList.beforeInsert(2 * index + 1, im);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]++;
        columnIndexes = columnIndexesList.elements();
        values = valuesList.elements();
    }

    protected void remove(int row, int index) {
        IntArrayList columnIndexesList = new IntArrayList(columnIndexes);
        columnIndexesList.setSizeRaw(rowPointers[rows]);
        FloatArrayList valuesList = new FloatArrayList(values);
        valuesList.setSizeRaw(rowPointers[rows]);
        columnIndexesList.remove(index);
        valuesList.remove(2 * index);
        valuesList.remove(2 * index + 1);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]--;
        columnIndexes = columnIndexesList.elements();
        values = valuesList.elements();
    }

}
