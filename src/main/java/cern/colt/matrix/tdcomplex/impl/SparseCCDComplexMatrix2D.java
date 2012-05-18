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
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse column-compressed 2-d matrix holding <tt>complex</tt> elements. First
 * see the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally uses the standard sparse column-compressed format. <br>
 * Note that this implementation is not synchronized.
 * <p>
 * Cells that
 * <ul>
 * <li>are never set to non-zero values do not use any memory.
 * <li>switch from zero to non-zero state do use memory.
 * <li>switch back from non-zero to zero state also do use memory. Their memory
 * is <i>not</i> automatically reclaimed. Reclamation can be triggered via
 * {@link #trimToSize()}.
 * </ul>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseCCDComplexMatrix2D extends WrapperDComplexMatrix2D {
    private static final long serialVersionUID = 1L;
    /*
     * Internal storage.
     */
    protected int[] columnPointers;

    protected int[] rowIndexes;

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
    public SparseCCDComplexMatrix2D(double[][] values) {
        this(values.length, values[0].length);
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
     *             if <tt>rows<0 || columns<0</tt> .
     */
    public SparseCCDComplexMatrix2D(int rows, int columns) {
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
     *             if <tt>rows<0 || columns<0</tt> .
     */
    public SparseCCDComplexMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        rowIndexes = new int[nzmax];
        values = new double[2 * nzmax];
        columnPointers = new int[columns + 1];
    }

    /**
     * Constructs a matrix with indexes given in the coordinate format and a
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
     *            the real part of numerical value
     * @param im
     *            the imaginary part of numerical value
     * @param removeDuplicates
     *            if true, then duplicates (if any) are removed
     */
    public SparseCCDComplexMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double re, double im,
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
        this.rowIndexes = new int[nz];
        this.values = new double[2 * nz];
        this.columnPointers = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nz; k++) {
            w[columnIndexes[k]]++;
        }
        cumsum(this.columnPointers, w, columns);
        for (int k = 0; k < nz; k++) {
            this.rowIndexes[r = w[columnIndexes[k]]++] = rowIndexes[k];
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
    public SparseCCDComplexMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double[] values,
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
        this.rowIndexes = new int[nz];
        this.values = new double[2 * nz];
        this.columnPointers = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nz; k++) {
            w[columnIndexes[k]]++;
        }
        cumsum(this.columnPointers, w, columns);
        for (int k = 0; k < nz; k++) {
            this.rowIndexes[r = w[columnIndexes[k]]++] = rowIndexes[k];
            this.values[2 * r] = values[2 * k];
            this.values[2 * r + 1] = values[2 * k + 1];
        }
        if (removeDuplicates) {
            removeDuplicates();
        }
    }

    public DComplexMatrix2D assign(final cern.colt.function.tdcomplex.DComplexDComplexFunction function) {
        if (function instanceof cern.jet.math.tdcomplex.DComplexMult) { // x[i] = mult*x[i]
            final double[] alpha = ((cern.jet.math.tdcomplex.DComplexMult) function).multiplicator;
            if (alpha[0] == 1 && alpha[1] == 0)
                return this;
            if (alpha[0] == 0 && alpha[1] == 0)
                return assign(0, 0);
            if (alpha[0] != alpha[0] || alpha[1] != alpha[1])
                return assign(alpha); // the funny definition of isNaN(). This should better not happen.

            final double[] valuesE = values;
            int nz = cardinality();
            double[] valE = new double[2];
            for (int j = 0; j < nz; j++) {
                valE[0] = valuesE[2 * j];
                valE[1] = valuesE[2 * j + 1];
                valE = DComplex.mult(valE, alpha);
                valuesE[2 * j] = valE[0];
                valuesE[2 * j + 1] = valE[1];
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
            Arrays.fill(rowIndexes, 0);
            Arrays.fill(columnPointers, 0);
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

        if (source instanceof SparseCCDComplexMatrix2D) {
            SparseCCDComplexMatrix2D other = (SparseCCDComplexMatrix2D) source;
            System.arraycopy(other.getColumnPointers(), 0, columnPointers, 0, columns + 1);
            int nzmax = other.getRowIndexes().length;
            if (rowIndexes.length < nzmax) {
                rowIndexes = new int[nzmax];
                values = new double[2 * nzmax];
            }
            System.arraycopy(other.getRowIndexes(), 0, rowIndexes, 0, nzmax);
            System.arraycopy(other.getValues(), 0, values, 0, other.getValues().length);
        } else if (source instanceof SparseRCDComplexMatrix2D) {
            SparseRCDComplexMatrix2D other = ((SparseRCDComplexMatrix2D) source).getConjugateTranspose();
            columnPointers = other.getRowPointers();
            rowIndexes = other.getColumnIndexes();
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

        if ((y instanceof SparseCCDComplexMatrix2D) && (function == cern.jet.math.tdcomplex.DComplexFunctions.plus)) { // x[i] = x[i] + y[i] 
            SparseCCDComplexMatrix2D yy = (SparseCCDComplexMatrix2D) y;
            int p, j, nz = 0, anz;
            int Cp[], Ci[], Bp[], m, n, bnz, w[];
            double x[], Cx[];
            m = rows;
            anz = columnPointers[columns];
            n = yy.columns;
            Bp = yy.columnPointers;
            bnz = Bp[n];
            w = new int[m]; /* get workspace */
            x = new double[2 * m]; /* get workspace */
            SparseCCDComplexMatrix2D C = new SparseCCDComplexMatrix2D(m, n, anz + bnz); /* allocate result*/
            Cp = C.columnPointers;
            Ci = C.rowIndexes;
            Cx = C.values;
            double[] one = new double[] { 1, 0 };
            for (j = 0; j < n; j++) {
                Cp[j] = nz; /* column j of C starts here */
                nz = scatter(this, j, one, w, x, j + 1, C, nz); /* alpha*A(:,j)*/
                nz = scatter(yy, j, one, w, x, j + 1, C, nz); /* beta*B(:,j) */
                for (p = Cp[j]; p < nz; p++) {
                    Cx[2 * p] = x[2 * Ci[p]];
                    Cx[2 * p + 1] = x[2 * Ci[p] + 1];
                }
            }
            Cp[n] = nz; /* finalize the last column of C */
            rowIndexes = Ci;
            columnPointers = Cp;
            values = Cx;
            return this;
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
            final int[] rowIndexesA = rowIndexes;
            final int[] columnPointersA = columnPointers;
            final double[] valuesA = values;
            double[] valA = new double[2];
            for (int j = columns; --j >= 0;) {
                int low = columnPointersA[j];
                for (int k = columnPointersA[j + 1]; --k >= low;) {
                    int i = rowIndexesA[k];
                    valA[0] = valuesA[2 * k];
                    valA[1] = valuesA[2 * k + 1];
                    valA = DComplex.mult(valA, y.getQuick(i, j));
                    valuesA[2 * k] = valA[0];
                    valuesA[2 * k + 1] = valA[1];
                    if (valuesA[2 * k] == 0 && valuesA[2 * k + 1] == 0)
                        remove(i, j);
                }
            }
            return this;
        }

        if (function == cern.jet.math.tdcomplex.DComplexFunctions.div) { // x[i] = x[i] / y[i]
            final int[] rowIndexesA = rowIndexes;
            final int[] columnPointersA = columnPointers;
            final double[] valuesA = values;

            double[] valA = new double[2];
            for (int j = columns; --j >= 0;) {
                int low = columnPointersA[j];
                for (int k = columnPointersA[j + 1]; --k >= low;) {
                    int i = rowIndexesA[k];
                    valA[0] = valuesA[2 * k];
                    valA[1] = valuesA[2 * k + 1];
                    valA = DComplex.div(valA, y.getQuick(i, j));
                    valuesA[2 * k] = valA[0];
                    valuesA[2 * k + 1] = valA[1];
                    if (valuesA[2 * k] == 0 && valuesA[2 * k + 1] == 0)
                        remove(i, j);
                }
            }
            return this;
        }
        return super.assign(y, function);
    }

    public int cardinality() {
        return columnPointers[columns];
    }

    public DComplexMatrix2D forEachNonZero(final cern.colt.function.tdcomplex.IntIntDComplexFunction function) {
        final int[] rowIndexesA = rowIndexes;
        final int[] columnPointersA = columnPointers;
        final double[] valuesA = values;
        double[] valA = new double[2];
        for (int j = columns; --j >= 0;) {
            int low = columnPointersA[j];
            for (int k = columnPointersA[j + 1]; --k >= low;) {
                int i = rowIndexesA[k];
                valA[0] = valuesA[2 * k];
                valA[1] = valuesA[2 * k + 1];
                valA = function.apply(i, j, valA);
                valuesA[2 * k] = valA[0];
                valuesA[2 * k + 1] = valA[1];
            }
        }
        return this;
    }

    /**
     * Returns column pointers
     * 
     * @return column pointers
     */
    public int[] getColumnPointers() {
        return columnPointers;
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
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(rowIndexes, row, columnPointers[column], columnPointers[column + 1] - 1);
        double[] v = new double[2];
        if (k >= 0) {
            v[0] = values[2 * k];
            v[1] = values[2 * k + 1];
        }
        return v;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a row-compressed form. This method creates a new object (not a view), so
     * changes in the returned matrix are NOT reflected in this matrix.
     * 
     * @return this matrix in a row-compressed form
     */
    public SparseRCDComplexMatrix2D getRowCompressed() {
        SparseCCDComplexMatrix2D tr = getConjugateTranspose();
        SparseRCDComplexMatrix2D rc = new SparseRCDComplexMatrix2D(rows, columns);
        rc.columnIndexes = tr.rowIndexes;
        rc.rowPointers = tr.columnPointers;
        rc.values = tr.values;
        return rc;
    }

    /**
     * Returns row indexes;
     * 
     * @return row indexes
     */
    public int[] getRowIndexes() {
        return rowIndexes;
    }

    public SparseCCDComplexMatrix2D getConjugateTranspose() {
        int p, q, j, Cp[], Ci[], n, m, Ap[], Ai[], w[];
        double Cx[], Ax[];
        m = rows;
        n = columns;
        Ap = columnPointers;
        Ai = rowIndexes;
        Ax = values;
        SparseCCDComplexMatrix2D C = new SparseCCDComplexMatrix2D(columns, rows, Ai.length); /* allocate result */
        w = new int[m]; /* get workspace */
        Cp = C.columnPointers;
        Ci = C.rowIndexes;
        Cx = C.values;
        for (p = 0; p < Ap[n]; p++)
            w[Ai[p]]++; /* row counts */
        cumsum(Cp, w, m); /* row pointers */
        for (j = 0; j < n; j++) {
            for (p = Ap[j]; p < Ap[j + 1]; p++) {
                Ci[q = w[Ai[p]]++] = j; /* place A(i,j) as entry C(j,i) */
                Cx[2 * q] = Ax[2 * p];
                Cx[2 * q + 1] = -Ax[2 * p + 1];
            }
        }
        return C;
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
        return new SparseCCDComplexMatrix2D(rows, columns);
    }

    public DComplexMatrix1D like1D(int size) {
        return new SparseDComplexMatrix1D(size);
    }

    public synchronized void setQuick(int row, int column, double[] value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(rowIndexes, row, columnPointers[column], columnPointers[column + 1] - 1);

        if (k >= 0) { // found
            if (value[0] == 0 && value[1] == 0) {
                remove(column, k);
            } else {
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
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(rowIndexes, row, columnPointers[column], columnPointers[column + 1] - 1);

        if (k >= 0) { // found
            if (re == 0 && im == 0) {
                remove(column, k);
            } else {
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

    /**
     * Sorts row indexes
     */
    public void sortRowIndexes() {
        SparseCCDComplexMatrix2D tr = getConjugateTranspose();
        tr = tr.getConjugateTranspose();
        columnPointers = tr.columnPointers;
        rowIndexes = tr.rowIndexes;
        values = tr.values;
    }

    /**
     * Removes (sums) duplicate entries (if any}
     */
    public void removeDuplicates() {
        int i, j, p, q, nz = 0, n, m, Ap[], Ai[], w[];
        double Ax[];
        /* check inputs */
        m = rows;
        n = columns;
        Ap = columnPointers;
        Ai = rowIndexes;
        Ax = values;
        w = new int[m]; /* get workspace */
        for (i = 0; i < m; i++)
            w[i] = -1; /* row i not yet seen */
        for (j = 0; j < n; j++) {
            q = nz; /* column j will start at q */
            for (p = Ap[j]; p < Ap[j + 1]; p++) {
                i = Ai[p]; /* A(i,j) is nonzero */
                if (w[i] >= q) {
                    Ax[w[i]] += Ax[p]; /* A(i,j) is a duplicate */
                } else {
                    w[i] = nz; /* record where row i occurs */
                    Ai[nz] = i; /* keep A(i,j) */
                    Ax[2 * nz] = Ax[2 * p];
                    Ax[2 * nz + 1] = Ax[2 * p + 1];
                    nz++;
                }
            }
            Ap[j] = q; /* record start of column j */
        }
        Ap[n] = nz; /* finalize A */
    }

    /**
     * Removes zero entries (if any)
     */
    public void removeZeroes() {
        int j, p, nz = 0, n, Ap[], Ai[];
        double Ax[];
        n = columns;
        Ap = columnPointers;
        Ai = rowIndexes;
        Ax = values;
        for (j = 0; j < n; j++) {
            p = Ap[j]; /* get current location of col j */
            Ap[j] = nz; /* record new location of col j */
            for (; p < Ap[j + 1]; p++) {
                if (Ax[p] != 0) {
                    Ai[nz] = Ai[p];
                    Ax[2 * nz] = Ax[2 * p]; /* keep A(i,j) */
                    Ax[2 * nz + 1] = Ax[2 * p + 1];
                    nz++;
                }
            }
        }
        Ap[n] = nz; /* finalize A */
    }

    public void trimToSize() {
        realloc(0);
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(rows).append(" x ").append(columns).append(" sparse matrix, nnz = ").append(cardinality())
                .append('\n');
        for (int i = 0; i < columns; i++) {
            int high = columnPointers[i + 1];
            for (int j = columnPointers[i]; j < high; j++) {
                if (values[2 * j + 1] > 0) {
                    builder.append('(').append(rowIndexes[j]).append(',').append(i).append(')').append('\t').append(
                            values[2 * j]).append('+').append(values[2 * j + 1]).append('i').append('\n');
                } else if (values[2 * j + 1] == 0) {
                    builder.append('(').append(rowIndexes[j]).append(',').append(i).append(')').append('\t').append(
                            values[2 * j]).append('\n');
                } else {
                    builder.append('(').append(rowIndexes[j]).append(',').append(i).append(')').append('\t').append(
                            values[2 * j]).append('-').append(values[2 * j + 1]).append('i').append('\n');
                }
            }
        }
        return builder.toString();
    }

    public DComplexMatrix1D zMult(DComplexMatrix1D y, DComplexMatrix1D z, final double[] alpha, final double[] beta,
            final boolean transposeA) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;

        boolean ignore = (z == null || transposeA);
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
        final int zeroZ = (int) zz.index(0);

        DenseDComplexMatrix1D yy = (DenseDComplexMatrix1D) y;
        final double[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) yy.index(0);

        final int[] rowIndexesA = rowIndexes;
        final int[] columnPointersA = columnPointers;
        final double[] valuesA = values;

        int zidx = zeroZ;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (!transposeA) {
            if ((!ignore) && !(beta[0] == 1 && beta[1] == 0)) {
                z.assign(cern.jet.math.tdcomplex.DComplexFunctions.mult(beta));
            }

            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = 2;
                Future<?>[] futures = new Future[nthreads];
                final double[] result = new double[2 * rowsA];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = j * k;
                    final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                    final int threadID = j;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            double[] yElem = new double[2];
                            double[] valA = new double[2];
                            if (threadID == 0) {
                                for (int i = firstColumn; i < lastColumn; i++) {
                                    int high = columnPointersA[i + 1];
                                    yElem[0] = elementsY[zeroY + strideY * i];
                                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                                    for (int k = columnPointersA[i]; k < high; k++) {
                                        int j = rowIndexesA[k];
                                        valA[0] = valuesA[2 * k];
                                        valA[1] = valuesA[2 * k + 1];
                                        valA = DComplex.mult(valA, yElem);
                                        valA = DComplex.mult(valA, alpha);
                                        elementsZ[zeroZ + strideZ * j] += valA[0];
                                        elementsZ[zeroZ + strideZ * j + 1] += valA[1];
                                    }
                                }
                            } else {
                                for (int i = firstColumn; i < lastColumn; i++) {
                                    int high = columnPointersA[i + 1];
                                    yElem[0] = elementsY[zeroY + strideY * i];
                                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                                    for (int k = columnPointersA[i]; k < high; k++) {
                                        int j = rowIndexesA[k];
                                        valA[0] = valuesA[2 * k];
                                        valA[1] = valuesA[2 * k + 1];
                                        valA = DComplex.mult(valA, yElem);
                                        valA = DComplex.mult(valA, alpha);
                                        result[2 * j] += valA[0];
                                        result[2 * j + 1] += valA[1];
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
                double[] valA = new double[2];
                for (int i = 0; i < columns; i++) {
                    int high = columnPointersA[i + 1];
                    yElem[0] = elementsY[zeroY + strideY * i];
                    yElem[1] = elementsY[zeroY + strideY * i + 1];
                    for (int k = columnPointersA[i]; k < high; k++) {
                        int j = rowIndexesA[k];
                        valA[0] = valuesA[2 * k];
                        valA[1] = valuesA[2 * k + 1];
                        valA = DComplex.mult(valA, yElem);
                        valA = DComplex.mult(valA, alpha);
                        elementsZ[zeroZ + strideZ * j] += valA[0];
                        elementsZ[zeroZ + strideZ * j + 1] += valA[1];
                    }
                }
            }
        } else {
            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                Future<?>[] futures = new Future[nthreads];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = j * k;
                    final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            int zidx = zeroZ + firstColumn * strideZ;
                            double[] valA = new double[2];
                            double[] valY = new double[2];
                            double[] valZ = new double[2];
                            for (int i = firstColumn; i < lastColumn; i++) {
                                double[] sum = new double[2];
                                int high = columnPointers[i + 1];
                                for (int k = columnPointers[i]; k < high; k++) {
                                    valA[0] = valuesA[2 * k];
                                    valA[1] = -valuesA[2 * k + 1];
                                    valY[0] = elementsY[zeroY + strideY * rowIndexes[k]];
                                    valY[1] = elementsY[zeroY + strideY * rowIndexes[k] + 1];
                                    sum = DComplex.plus(sum, DComplex.mult(valA, valY));
                                }
                                sum = DComplex.mult(alpha, sum);
                                valZ[0] = elementsZ[zidx];
                                valZ[1] = elementsZ[zidx + 1];
                                valZ = DComplex.mult(valZ, beta);
                                elementsZ[zidx] = sum[0] + valZ[0];
                                elementsZ[zidx + 1] = sum[1] + valZ[1];
                                zidx += strideZ;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                double[] valA = new double[2];
                double[] valY = new double[2];
                double[] valZ = new double[2];
                for (int i = 0; i < columns; i++) {
                    double[] sum = new double[2];
                    int high = columnPointers[i + 1];
                    for (int k = columnPointers[i]; k < high; k++) {
                        valA[0] = valuesA[2 * k];
                        valA[1] = -valuesA[2 * k + 1];
                        valY[0] = elementsY[zeroY + strideY * rowIndexes[k]];
                        valY[1] = elementsY[zeroY + strideY * rowIndexes[k] + 1];
                        sum = DComplex.plus(sum, DComplex.mult(valA, valY));
                    }
                    sum = DComplex.mult(alpha, sum);
                    valZ[0] = elementsZ[zidx];
                    valZ[1] = elementsZ[zidx + 1];
                    valZ = DComplex.mult(valZ, beta);
                    elementsZ[zidx] = sum[0] + valZ[0];
                    elementsZ[zidx + 1] = sum[1] + valZ[1];
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
            if (B instanceof SparseCCDComplexMatrix2D) {
                C = new SparseCCDComplexMatrix2D(rowsA, p, (rowsA * p));
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

        if (!ignore && !(beta[0] == 1 && beta[1] == 0)) {
            C.assign(cern.jet.math.tdcomplex.DComplexFunctions.mult(beta));
        }

        if ((B instanceof DenseDComplexMatrix2D) && (C instanceof DenseDComplexMatrix2D)) {
            SparseCCDComplexMatrix2D AA;
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
            int[] columnPointersA = AA.columnPointers;
            int[] rowIndexesA = AA.rowIndexes;
            double[] valuesA = AA.values;

            int zeroB = (int) BB.index(0, 0);
            int rowStrideB = BB.rowStride();
            int columnStrideB = BB.columnStride();
            double[] elementsB = BB.elements;

            int zeroC = (int) CC.index(0, 0);
            int rowStrideC = CC.rowStride();
            int columnStrideC = CC.columnStride();
            double[] elementsC = CC.elements;
            double[] valA = new double[2];
            double[] valB = new double[2];
            for (int jj = 0; jj < columnsB; jj++) {
                for (int kk = 0; kk < columnsA; kk++) {
                    int high = columnPointersA[kk + 1];
                    valB[0] = elementsB[zeroB + kk * rowStrideB + jj * columnStrideB];
                    valB[1] = elementsB[zeroB + kk * rowStrideB + jj * columnStrideB + 1];
                    for (int ii = columnPointersA[kk]; ii < high; ii++) {
                        int j = rowIndexesA[ii];
                        valA[0] = valuesA[2 * ii];
                        valA[1] = valuesA[2 * ii + 1];
                        valA = DComplex.mult(valA, valB);
                        elementsC[zeroC + j * rowStrideC + jj * columnStrideC] += valA[0];
                        elementsC[zeroC + j * rowStrideC + jj * columnStrideC + 1] += valA[1];
                    }
                }
            }
            if (!(alpha[0] == 1.0 && alpha[1] == 0)) {
                C.assign(cern.jet.math.tdcomplex.DComplexFunctions.mult(alpha));
            }

        } else if ((B instanceof SparseCCDComplexMatrix2D) && (C instanceof SparseCCDComplexMatrix2D)) {
            SparseCCDComplexMatrix2D AA;
            if (transposeA) {
                AA = getConjugateTranspose();
            } else {
                AA = this;
            }
            SparseCCDComplexMatrix2D BB = (SparseCCDComplexMatrix2D) B;
            if (transposeB) {
                BB = BB.getConjugateTranspose();
            }
            SparseCCDComplexMatrix2D CC = (SparseCCDComplexMatrix2D) C;
            int j, nz = 0, Cp[], Ci[], Bp[], m, n, w[], Bi[];
            double x[], Bx[], Cx[];
            m = rowsA;
            n = columnsB;
            Bp = BB.columnPointers;
            Bi = BB.rowIndexes;
            Bx = BB.values;
            w = new int[m]; /* get workspace */
            x = new double[2 * m]; /* get workspace */
            Cp = CC.columnPointers;
            Ci = CC.rowIndexes;
            Cx = CC.values;
            for (j = 0; j < n; j++) {
                int nzmaxC = CC.rowIndexes.length;
                if (nz + m > nzmaxC) {
                    nzmaxC = 2 * nzmaxC + m;
                    int[] rowIndexesNew = new int[nzmaxC];
                    System.arraycopy(Ci, 0, rowIndexesNew, 0, Ci.length);
                    Ci = rowIndexesNew;
                    double[] valuesNew = new double[2 * nzmaxC];
                    System.arraycopy(Cx, 0, valuesNew, 0, Cx.length);
                    Cx = valuesNew;
                }
                Cp[j] = nz; /* column j of C starts here */
                double[] elemB = new double[2];
                for (p = Bp[j]; p < Bp[j + 1]; p++) {
                    elemB[0] = Bx[2 * p];
                    elemB[1] = Bx[2 * p + 1];
                    nz = scatter(AA, Bi[p], elemB, w, x, j + 1, CC, nz);
                }
                for (p = Cp[j]; p < nz; p++) {
                    Cx[2 * p] = x[2 * Ci[p]];
                    Cx[2 * p + 1] = x[2 * Ci[p] + 1];
                }
            }
            Cp[n] = nz; /* finalize the last column of C */
            if (!(alpha[0] == 1.0 && alpha[1] == 0)) {
                CC.assign(cern.jet.math.tdcomplex.DComplexFunctions.mult(alpha));
            }
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

            final int[] rowIndexesA = rowIndexes;
            final int[] columnPointersA = columnPointers;
            final double[] valuesA = values;
            double[] valA = new double[2];
            for (int i = columns; --i >= 0;) {
                int low = columnPointersA[i];
                for (int k = columnPointersA[i + 1]; --k >= low;) {
                    int j = rowIndexesA[k];
                    valA[0] = valuesA[2 * k];
                    valA[1] = valuesA[2 * k + 1];
                    fun.multiplicator = DComplex.mult(valA, alpha);
                    if (!transposeA)
                        Crows[j].assign(Brows[i], fun);
                    else
                        Crows[i].assign(Brows[j], fun);
                }
            }
        }
        return C;
    }

    protected DComplexMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, double[] value) {
        IntArrayList rowIndexesList = new IntArrayList(rowIndexes);
        rowIndexesList.setSizeRaw(columnPointers[columns]);
        DoubleArrayList valuesList = new DoubleArrayList(values);
        valuesList.setSizeRaw(2 * columnPointers[columns]);
        rowIndexesList.beforeInsert(index, row);
        valuesList.beforeInsert(2 * index, value[0]);
        valuesList.beforeInsert(2 * index + 1, value[1]);
        for (int i = columnPointers.length; --i > column;)
            columnPointers[i]++;
        rowIndexes = rowIndexesList.elements();
        values = valuesList.elements();
    }

    protected void insert(int row, int column, int index, double re, double im) {
        IntArrayList rowIndexesList = new IntArrayList(rowIndexes);
        rowIndexesList.setSizeRaw(columnPointers[columns]);
        DoubleArrayList valuesList = new DoubleArrayList(values);
        valuesList.setSizeRaw(2 * columnPointers[columns]);
        rowIndexesList.beforeInsert(index, row);
        valuesList.beforeInsert(2 * index, re);
        valuesList.beforeInsert(2 * index + 1, im);
        for (int i = columnPointers.length; --i > column;)
            columnPointers[i]++;
        rowIndexes = rowIndexesList.elements();
        values = valuesList.elements();
    }

    protected void remove(int column, int index) {
        IntArrayList rowIndexesList = new IntArrayList(rowIndexes);
        DoubleArrayList valuesList = new DoubleArrayList(values);
        rowIndexesList.remove(index);
        valuesList.remove(2 * index);
        valuesList.remove(2 * index + 1);
        for (int i = columnPointers.length; --i > column;)
            columnPointers[i]--;
        rowIndexes = rowIndexesList.elements();
        values = valuesList.elements();
    }

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

    private long cumsum(int[] p, int[] c, int n) {
        int nz = 0;
        long nz2 = 0;
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
            nzmax = columnPointers[columns];
        int[] rowIndexesNew = new int[nzmax];
        int length = Math.min(nzmax, rowIndexes.length);
        System.arraycopy(rowIndexes, 0, rowIndexesNew, 0, length);
        rowIndexes = rowIndexesNew;
        double[] valuesNew = new double[2 * nzmax];
        length = Math.min(nzmax, values.length);
        System.arraycopy(values, 0, valuesNew, 0, length);
        values = valuesNew;
    }

    private int scatter(SparseCCDComplexMatrix2D A, int j, double[] beta, int[] w, double[] x, int mark,
            SparseCCDComplexMatrix2D C, int nz) {
        int i, p;
        int Ap[], Ai[], Ci[];
        double[] Ax;
        Ap = A.columnPointers;
        Ai = A.rowIndexes;
        Ax = A.values;
        Ci = C.rowIndexes;
        double[] valX = new double[2];
        double[] valA = new double[2];
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p]; /* A(i,j) is nonzero */
            if (w[i] < mark) {
                w[i] = mark; /* i is new entry in column j */
                Ci[nz++] = i; /* add i to pattern of C(:,j) */
                if (x != null) {
                    valA[0] = Ax[2 * p];
                    valA[1] = Ax[2 * p + 1];
                    valA = DComplex.mult(beta, valA);
                    x[2 * i] = valA[0]; /* x(i) = beta*A(i,j) */
                    x[2 * i + 1] = valA[1];
                }
            } else if (x != null) {
                valA[0] = Ax[2 * p];
                valA[1] = Ax[2 * p + 1];
                valA = DComplex.mult(beta, valA);
                x[2 * i] += valA[0]; /* i exists in C(:,j) already */
                x[2 * i + 1] += valA[1];
            }
        }
        return nz;
    }
}
