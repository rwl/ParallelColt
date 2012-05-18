/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import java.util.Arrays;
import java.util.concurrent.Future;

import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
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
 * <b>Memory requirements:</b>
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
public class SparseRCDComplexMatrix2D extends WrapperDComplexMatrix2D {
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

    protected double[] values;

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
    public SparseRCDComplexMatrix2D(double[][] values) {
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
    public SparseRCDComplexMatrix2D(int rows, int columns) {
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
    public SparseRCDComplexMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        columnIndexes = new int[nzmax];
        values = new double[2 * nzmax];
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
    public SparseRCDComplexMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double re, double im,
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
        this.values = new double[2 * nz];
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
    public SparseRCDComplexMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double[] values,
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
        this.values = new double[2 * nz];
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
    public SparseRCDComplexMatrix2D(int rows, int columns, int[] rowPointers, int[] columnIndexes, double[] values) {
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

    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexDComplexFunction function) {
        if (function instanceof cern.jet.math.tdcomplex.DComplexMult) { // x[i] = mult*x[i]
            final double[] alpha = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
            if (alpha[0] == 1 && alpha[1] == 0)
                return this;
            if (alpha[0] == 0 && alpha[1] == 0)
                return assign(alpha);
            if (alpha[0] != alpha[0] || alpha[1] != alpha[1])
                return assign(alpha); // the funny definition of isNaN(). This should better not happen.
            int nz = cardinality();
            double[] elem = new double[2];
            for (int j = 0; j < nz; j++) {
                elem[0] = values[2 * j];
                elem[1] = values[2 * j + 1];
                elem = DComplex.mult(elem, alpha);
                values[2 * j] = elem[0];
                values[2 * j + 1] = elem[1];
            }
        } else {
            forEachNonZero(new cern.colt.function.tdcomplex.IntIntDComplexFunction() {
                public double[] apply(int i, int j, double[] value) {
                    return function.apply(value);
                }
            });
        }
        return this;
    }

    public DComplexMatrix2D assign(double re, double im) {
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

    public DComplexMatrix2D assign(DComplexMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof SparseRCDComplexMatrix2D) {
            SparseRCDComplexMatrix2D other = (SparseRCDComplexMatrix2D) source;
            System.arraycopy(other.rowPointers, 0, rowPointers, 0, rows + 1);
            int nzmax = other.columnIndexes.length;
            if (columnIndexes.length < nzmax) {
                columnIndexes = new int[nzmax];
                values = new double[2 * nzmax];
            }
            System.arraycopy(other.columnIndexes, 0, columnIndexes, 0, nzmax);
            System.arraycopy(other.values, 0, values, 0, other.values.length);
        } else if (source instanceof SparseCCDComplexMatrix2D) {
            SparseCCDComplexMatrix2D other = ((SparseCCDComplexMatrix2D) source).getConjugateTranspose();
            rowPointers = other.getColumnPointers();
            columnIndexes = other.getRowIndexes();
            values = other.getValues();
        } else {
            assign(0, 0);
            source.forEachNonZero(new cern.colt.function.tdcomplex.IntIntDComplexFunction() {
                public double[] apply(int i, int j, double[] value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
        }
        return this;
    }

    public DComplexMatrix2D assign(final DComplexMatrix2D y,
            cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction function) {
        checkShape(y);
        if ((y instanceof SparseRCDComplexMatrix2D) && (function == cern.jet.math.tdcomplex.DComplexFunctions.plus)) { // x[i] = x[i] + y[i] 
            SparseRCDComplexMatrix2D yy = (SparseRCDComplexMatrix2D) y;

            final int[] rowPointersY = yy.rowPointers;
            final int[] columnIndexesY = yy.columnIndexes;
            final double[] valuesY = yy.values;

            final int[] rowPointersC = new int[rows + 1];
            int cnz = Math.max(columnIndexes.length, (int) Math.min(Integer.MAX_VALUE, (long) rowPointers[rows]
                    + (long) rowPointersY[rows]));
            final int[] columnIndexesC = new int[cnz];
            final double[] valuesC = new double[2 * cnz];
            int nrow = rows;
            int ncol = columns;
            int nzmax = cnz;
            if (function == cern.jet.math.tdcomplex.DComplexFunctions.plus) { // x[i] = x[i] + y[i]
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

        if (function instanceof cern.jet.math.tdcomplex.DComplexPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final double[] alpha = ((cern.jet.math.tdcomplex.DComplexPlusMultSecond) function).multiplicator;
            if (alpha[0] == 0 && alpha[1] == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tdcomplex.IntIntDComplexFunction() {
                public double[] apply(int i, int j, double[] value) {
                    setQuick(i, j, DComplex.plus(getQuick(i, j), DComplex.mult(alpha, value)));
                    return value;
                }
            });
            return this;
        }

        if (function instanceof cern.jet.math.tdcomplex.DComplexPlusMultFirst) { // x[i] = alpha*x[i] + y[i]
            final double[] alpha = ((cern.jet.math.tdcomplex.DComplexPlusMultFirst) function).multiplicator;
            if (alpha[0] == 0 && alpha[1] == 0)
                return assign(y);
            y.forEachNonZero(new cern.colt.function.tdcomplex.IntIntDComplexFunction() {
                public double[] apply(int i, int j, double[] value) {
                    setQuick(i, j, DComplex.plus(DComplex.mult(alpha, getQuick(i, j)), value));
                    return value;
                }
            });
            return this;
        }

        if (function == cern.jet.math.tdcomplex.DComplexFunctions.mult) { // x[i] = x[i] * y[i]
            double[] elem = new double[2];
            for (int i = 0; i < rows; i++) {
                int high = rowPointers[i + 1];
                for (int k = rowPointers[i]; k < high; k++) {
                    int j = columnIndexes[k];
                    elem[0] = values[2 * k];
                    elem[1] = values[2 * k + 1];
                    elem = DComplex.mult(elem, y.getQuick(i, j));
                    values[2 * k] = elem[0];
                    values[2 * k + 1] = elem[1];
                    if (values[2 * k] == 0 && values[2 * k + 1] == 0)
                        remove(i, j);
                }
            }
            return this;
        }

        if (function == cern.jet.math.tdcomplex.DComplexFunctions.div) { // x[i] = x[i] / y[i]
            double[] elem = new double[2];
            for (int i = 0; i < rows; i++) {
                int high = rowPointers[i + 1];
                for (int k = rowPointers[i]; k < high; k++) {
                    int j = columnIndexes[k];
                    elem[0] = values[2 * k];
                    elem[1] = values[2 * k + 1];
                    elem = DComplex.div(elem, y.getQuick(i, j));
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

    public DComplexMatrix2D forEachNonZero(final cern.colt.function.tdcomplex.IntIntDComplexFunction function) {
        double[] value = new double[2];
        for (int i = 0; i < rows; i++) {
            int high = rowPointers[i + 1];
            for (int k = rowPointers[i]; k < high; k++) {
                int j = columnIndexes[k];
                value[0] = values[2 * k];
                value[1] = values[2 * k + 1];
                double[] r = function.apply(i, j, value);
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
    public SparseCCDComplexMatrix2D getColumnCompressed() {
        SparseRCDComplexMatrix2D tr = getConjugateTranspose();
        SparseCCDComplexMatrix2D cc = new SparseCCDComplexMatrix2D(rows, columns);
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
    public DenseDComplexMatrix2D getDense() {
        final DenseDComplexMatrix2D dense = new DenseDComplexMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tdcomplex.IntIntDComplexFunction() {
            public double[] apply(int i, int j, double[] value) {
                dense.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return dense;
    }

    public synchronized double[] getQuick(int row, int column) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        double[] v = new double[2];
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

    public SparseRCDComplexMatrix2D getConjugateTranspose() {
        int nnz = rowPointers[rows];
        int[] w = new int[columns];
        int[] rowPointersT = new int[columns + 1];
        int[] columnIndexesT = new int[nnz];
        double[] valuesT = new double[2 * nnz];

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
        SparseRCDComplexMatrix2D T = new SparseRCDComplexMatrix2D(columns, rows);
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
    public double[] getValues() {
        return values;
    }

    public DComplexMatrix2D like(int rows, int columns) {
        return new SparseRCDComplexMatrix2D(rows, columns);
    }

    public DComplexMatrix1D like1D(int size) {
        return new SparseDComplexMatrix1D(size);
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
        double eps = Math.pow(2, -52);
        double[] elem = new double[2];
        for (int j = 0; j < rows; j++) {
            int p = rowPointers[j]; /* get current location of row j */
            rowPointers[j] = nz; /* record new location of row j */
            for (; p < rowPointers[j + 1]; p++) {
                elem[0] = values[2 * p];
                elem[1] = values[2 * p + 1];
                if (DComplex.abs(elem) > eps) {
                    values[2 * nz] = values[2 * p]; /* keep A(i,j) */
                    values[2 * nz + 1] = values[2 * p + 1];
                    columnIndexes[nz++] = columnIndexes[p];
                }
            }
        }
        rowPointers[rows] = nz; /* finalize A */
    }

    public synchronized void setQuick(int row, int column, double[] value) {
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

    public synchronized void setQuick(int row, int column, double re, double im) {
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

    public DComplexMatrix1D zMult(DComplexMatrix1D y, DComplexMatrix1D z, final double[] alpha, final double[] beta,
            final boolean transposeA) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;

        boolean ignore = (z == null || !transposeA);
        if (z == null)
            z = new DenseDComplexMatrix1D(rowsA);

        if (!(y instanceof DenseDComplexMatrix1D && z instanceof DenseDComplexMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        DenseDComplexMatrix1D zz = (DenseDComplexMatrix1D) z;
        final double[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) z.index(0);

        DenseDComplexMatrix1D yy = (DenseDComplexMatrix1D) y;
        final double[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) y.index(0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();

        if (transposeA) {
            if ((!ignore) && !((beta[0] == 1) && (beta[1] == 0)))
                z.assign(cern.jet.math.tdcomplex.DComplexFunctions.mult(beta));

            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = 2;
                Future<?>[] futures = new Future[nthreads];
                final double[] result = new double[2 * rowsA];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    final int threadID = j;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            double[] yElem = new double[2];
                            double[] val = new double[2];
                            if (threadID == 0) {
                                for (int i = firstRow; i < lastRow; i++) {
                                    int high = rowPointers[i + 1];
                                    yElem[0] = elementsY[zeroY + strideY * i];
                                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                                    yElem = DComplex.mult(alpha, yElem);
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = columnIndexes[k];
                                        val[0] = values[2 * k];
                                        val[1] = -values[2 * k + 1];
                                        val = DComplex.mult(val, yElem);
                                        elementsZ[zeroZ + strideZ * j] += val[0];
                                        elementsZ[zeroZ + strideZ * j + 1] += val[1];
                                    }
                                }
                            } else {
                                for (int i = firstRow; i < lastRow; i++) {
                                    int high = rowPointers[i + 1];
                                    yElem[0] = elementsY[zeroY + strideY * i];
                                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                                    yElem = DComplex.mult(alpha, yElem);
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = columnIndexes[k];
                                        val[0] = values[2 * k];
                                        val[1] = -values[2 * k + 1];
                                        val = DComplex.mult(val, yElem);
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
                double[] yElem = new double[2];
                double[] val = new double[2];
                for (int i = 0; i < rows; i++) {
                    int high = rowPointers[i + 1];
                    yElem[0] = elementsY[zeroY + strideY * i];
                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                    yElem = DComplex.mult(alpha, yElem);
                    for (int k = rowPointers[i]; k < high; k++) {
                        int j = columnIndexes[k];
                        val[0] = values[2 * k];
                        val[1] = -values[2 * k + 1];
                        val = DComplex.mult(val, yElem);
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
                        double[] yElem = new double[2];
                        double[] val = new double[2];
                        if (beta[0] == 0.0 && beta[1] == 0) {
                            for (int i = firstRow; i < lastRow; i++) {
                                double[] sum = new double[2];
                                int high = rowPointers[i + 1];
                                for (int k = rowPointers[i]; k < high; k++) {
                                    yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                                    yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                                    val[0] = values[2 * k];
                                    val[1] = values[2 * k + 1];
                                    sum = DComplex.plus(sum, DComplex.mult(val, yElem));
                                }
                                sum = DComplex.mult(alpha, sum);
                                elementsZ[zidx] = sum[0];
                                elementsZ[zidx + 1] = sum[1];
                                zidx += strideZ;
                            }
                        } else {
                            double[] zElem = new double[2];
                            for (int i = firstRow; i < lastRow; i++) {
                                double[] sum = new double[2];
                                int high = rowPointers[i + 1];
                                for (int k = rowPointers[i]; k < high; k++) {
                                    yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                                    yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                                    val[0] = values[2 * k];
                                    val[1] = values[2 * k + 1];
                                    sum = DComplex.plus(sum, DComplex.mult(val, yElem));
                                }
                                sum = DComplex.mult(alpha, sum);
                                zElem[0] = elementsZ[zidx];
                                zElem[1] = elementsZ[zidx + 1];
                                zElem = DComplex.mult(beta, zElem);
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
            double[] yElem = new double[2];
            double[] val = new double[2];
            if (beta[0] == 0.0 && beta[1] == 0) {
                for (int i = 0; i < rows; i++) {
                    double[] sum = new double[2];
                    int high = rowPointers[i + 1];
                    for (int k = rowPointers[i]; k < high; k++) {
                        yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                        yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                        val[0] = values[2 * k];
                        val[1] = values[2 * k + 1];
                        sum = DComplex.plus(sum, DComplex.mult(val, yElem));
                    }
                    sum = DComplex.mult(alpha, sum);
                    elementsZ[zidx] = sum[0];
                    elementsZ[zidx + 1] = sum[1];
                    zidx += strideZ;
                }
            } else {
                double[] zElem = new double[2];
                for (int i = 0; i < rows; i++) {
                    double[] sum = new double[2];
                    int high = rowPointers[i + 1];
                    for (int k = rowPointers[i]; k < high; k++) {
                        yElem[0] = elementsY[zeroY + strideY * columnIndexes[k]];
                        yElem[1] = elementsY[zeroY + strideY * columnIndexes[k] + 1];
                        val[0] = values[2 * k];
                        val[1] = values[2 * k + 1];
                        sum = DComplex.plus(sum, DComplex.mult(val, yElem));
                    }
                    sum = DComplex.mult(alpha, sum);
                    zElem[0] = elementsZ[zidx];
                    zElem[1] = elementsZ[zidx + 1];
                    zElem = DComplex.mult(beta, zElem);
                    elementsZ[zidx] = sum[0] + zElem[0];
                    elementsZ[zidx + 1] = sum[1] + zElem[1];
                    zidx += strideZ;
                }
            }
        }
        return z;
    }

    public DComplexMatrix2D zMult(DComplexMatrix2D B, DComplexMatrix2D C, final double[] alpha, double[] beta,
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
            if (B instanceof SparseRCDComplexMatrix2D) {
                C = new SparseRCDComplexMatrix2D(rowsA, p, (rowsA * p));
            } else {
                C = new DenseDComplexMatrix2D(rowsA, p);
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
            C.assign(cern.jet.math.tdcomplex.DComplexFunctions.mult(beta));
        }

        if ((B instanceof DenseDComplexMatrix2D) && (C instanceof DenseDComplexMatrix2D)) {
            SparseRCDComplexMatrix2D AA;
            if (transposeA) {
                AA = getConjugateTranspose();
            } else {
                AA = this;
            }
            DenseDComplexMatrix2D BB;
            if (transposeB) {
                BB = (DenseDComplexMatrix2D) B.getConjugateTranspose();
            } else {
                BB = (DenseDComplexMatrix2D) B;
            }

            DenseDComplexMatrix2D CC = (DenseDComplexMatrix2D) C;
            int[] rowPointersA = AA.rowPointers;
            int[] columnIndexesA = AA.columnIndexes;
            double[] valuesA = AA.values;
            double[] valA = new double[2];
            for (int ii = 0; ii < rowsA; ii++) {
                int highA = rowPointersA[ii + 1];
                for (int ka = rowPointersA[ii]; ka < highA; ka++) {
                    valA[0] = valuesA[2 * ka];
                    valA[1] = valuesA[2 * ka + 1];
                    double[] scal = DComplex.mult(alpha, valA);
                    int jj = columnIndexesA[ka];
                    CC.viewRow(ii).assign(BB.viewRow(jj), DComplexFunctions.plusMultSecond(scal));
                }
            }
        } else if ((B instanceof SparseRCDComplexMatrix2D) && (C instanceof SparseRCDComplexMatrix2D)) {
            SparseRCDComplexMatrix2D AA;
            SparseRCDComplexMatrix2D BB;
            SparseRCDComplexMatrix2D CC = (SparseRCDComplexMatrix2D) C;
            if (transposeA) {
                AA = getConjugateTranspose();
            } else {
                AA = this;
            }
            if (transposeB) {
                BB = ((SparseRCDComplexMatrix2D) B).getConjugateTranspose();
            } else {
                BB = (SparseRCDComplexMatrix2D) B;
            }

            int[] rowPointersA = AA.rowPointers;
            int[] columnIndexesA = AA.columnIndexes;
            double[] valuesA = AA.values;

            int[] rowPointersB = BB.rowPointers;
            int[] columnIndexesB = BB.columnIndexes;
            double[] valuesB = BB.values;

            int[] rowPointersC = CC.rowPointers;
            int[] columnIndexesC = CC.columnIndexes;
            double[] valuesC = CC.values;
            int nzmax = columnIndexesC.length;

            int[] iw = new int[columnsB + 1];
            for (int i = 0; i < iw.length; i++) {
                iw[i] = -1;
            }
            int len = -1;
            double[] valA = new double[2];
            double[] valB = new double[2];
            double[] valC = new double[2];
            for (int ii = 0; ii < rowsA; ii++) {
                int highA = rowPointersA[ii + 1];
                for (int ka = rowPointersA[ii]; ka < highA; ka++) {
                    valA[0] = valuesA[2 * ka];
                    valA[1] = valuesA[2 * ka + 1];
                    double[] scal = DComplex.mult(alpha, valA);
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
                            valB = DComplex.mult(scal, valB);
                            valuesC[2 * len] = valB[0];
                            valuesC[2 * len + 1] = valB[1];
                        } else {
                            valB[0] = valuesB[2 * kb];
                            valB[1] = valuesB[2 * kb + 1];
                            valB = DComplex.mult(scal, valB);
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
                //                DComplexMatrix1D valuesCPart = valuesC.viewPart(rowPointersC[ii], length).viewSelection(indexes);
                //                valuesC.viewPart(rowPointersC[ii], length).assign(valuesCPart);
            }
            //            CC.columnIndexes.elements((int[]) columnIndexesC.elements());
            //            CC.columnIndexes.setSize(columnIndexesSize);
            //            CC.values.elements((double[]) valuesC.elements());
            //            CC.values.setSize(columnIndexesSize);
        } else {
            if (transposeB) {
                B = B.getConjugateTranspose();
            }
            // cache views
            final DComplexMatrix1D[] Brows = new DComplexMatrix1D[columnsA];
            for (int i = columnsA; --i >= 0;)
                Brows[i] = B.viewRow(i);
            final DComplexMatrix1D[] Crows = new DComplexMatrix1D[rowsA];
            for (int i = rowsA; --i >= 0;)
                Crows[i] = C.viewRow(i);

            final cern.jet.math.tdcomplex.DComplexPlusMultSecond fun = cern.jet.math.tdcomplex.DComplexPlusMultSecond
                    .plusMult(new double[2]);

            final int[] columnIndexesA = columnIndexes;
            final double[] valuesA = values;
            double[] valA = new double[2];
            for (int i = rows; --i >= 0;) {
                int low = rowPointers[i];
                for (int k = rowPointers[i + 1]; --k >= low;) {
                    int j = columnIndexesA[k];
                    valA[0] = valuesA[2 * k];
                    valA[1] = valuesA[2 * k + 1];
                    fun.multiplicator = DComplex.mult(valA, alpha);
                    if (!transposeA)
                        Crows[i].assign(Brows[j], fun);
                    else
                        Crows[j].assign(Brows[i], fun);
                }
            }
        }
        return C;
    }

    private double cumsum(int[] p, int[] c, int n) {
        int nz = 0;
        double nz2 = 0;
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
        double[] valuesNew = new double[2 * nzmax];
        length = Math.min(nzmax, values.length);
        System.arraycopy(values, 0, valuesNew, 0, length);
        values = valuesNew;
    }

    protected DComplexMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, double[] value) {
        IntArrayList columnIndexesList = new IntArrayList(columnIndexes);
        columnIndexesList.setSizeRaw(rowPointers[rows]);
        DoubleArrayList valuesList = new DoubleArrayList(values);
        valuesList.setSizeRaw(2 * rowPointers[rows]);
        columnIndexesList.beforeInsert(index, column);
        valuesList.beforeInsert(2 * index, value[0]);
        valuesList.beforeInsert(2 * index + 1, value[1]);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]++;
        columnIndexes = columnIndexesList.elements();
        values = valuesList.elements();
    }

    protected void insert(int row, int column, int index, double re, double im) {
        IntArrayList columnIndexesList = new IntArrayList(columnIndexes);
        columnIndexesList.setSizeRaw(rowPointers[rows]);
        DoubleArrayList valuesList = new DoubleArrayList(values);
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
        DoubleArrayList valuesList = new DoubleArrayList(values);
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
