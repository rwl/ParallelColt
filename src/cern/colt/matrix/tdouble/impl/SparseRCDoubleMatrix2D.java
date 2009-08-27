/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import java.util.Arrays;
import java.util.concurrent.Future;

import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse row-compressed 2-d matrix holding <tt>double</tt> elements. First see
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
 * <p>
 * <tt>memory [bytes] = 4*rows + 12 * nonZeros</tt>. <br>
 * Where <tt>nonZeros = cardinality()</tt> is the number of non-zero cells.
 * Thus, a 1000 x 1000 matrix with 1000000 non-zero cells consumes 11.5 MB. The
 * same 1000 x 1000 matrix with 1000 non-zero cells consumes 15 KB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * Getting a cell value takes time<tt> O(log nzr)</tt> where <tt>nzr</tt> is the
 * number of non-zeros of the touched row. This is usually quick, because
 * typically there are only few nonzeros per row. So, in practice, get has
 * <i>expected</i> constant time. Setting a cell value takes <i> </i>worst-case
 * time <tt>O(nz)</tt> where <tt>nzr</tt> is the total number of non-zeros in
 * the matrix. This can be extremely slow, but if you traverse coordinates
 * properly (i.e. upwards), each write is done much quicker:
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 * // rather quick
 * matrix.assign(0);
 * for (int row = 0; row &lt; rows; row++) {
 *     for (int column = 0; column &lt; columns; column++) {
 *         if (someCondition)
 *             matrix.setQuick(row, column, someValue);
 *     }
 * }
 * 
 * // poor
 * matrix.assign(0);
 * for (int row = rows; --row &gt;= 0;) {
 *     for (int column = columns; --column &gt;= 0;) {
 *         if (someCondition)
 *             matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * </td>
 * </table>
 * If for whatever reasons you can't iterate properly, consider to create an
 * empty dense matrix, store your non-zeros in it, then call
 * <tt>sparse.assign(dense)</tt>. Under the circumstances, this is still rather
 * quick.
 * <p>
 * Fast iteration over non-zeros can be done via {@link #forEachNonZero}, which
 * supplies your function with row, column and value of each nonzero. Although
 * the internally implemented version is a bit more sophisticated, here is how a
 * quite efficient user-level matrix-vector multiplication could look like:
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 * // Linear algebraic y = A * x
 * A.forEachNonZero(new cern.colt.function.IntIntDoubleFunction() {
 *     public double apply(int row, int column, double value) {
 *         y.setQuick(row, y.getQuick(row) + value * x.getQuick(column));
 *         return value;
 *     }
 * });
 * </pre>
 * 
 * </td>
 * </table>
 * <p>
 * Here is how a a quite efficient user-level combined scaling operation could
 * look like:
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 * // Elementwise A = A + alpha*B
 * B.forEachNonZero(new cern.colt.function.IntIntDoubleFunction() {
 *     public double apply(int row, int column, double value) {
 *         A.setQuick(row, column, A.getQuick(row, column) + alpha * value);
 *         return value;
 *     }
 * });
 * </pre>
 * 
 * </td>
 * </table>
 * Method
 * {@link #assign(DoubleMatrix2D,cern.colt.function.tdouble.DoubleDoubleFunction)}
 * does just that if you supply
 * {@link cern.jet.math.tdouble.DoubleFunctions#plusMultSecond} as argument.
 * 
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 0.9, 04/14/2000
 */
public class SparseRCDoubleMatrix2D extends WrapperDoubleMatrix2D {
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

    protected boolean columnIndexesSorted = false;

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
    public SparseRCDoubleMatrix2D(double[][] values) {
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
    public SparseRCDoubleMatrix2D(int rows, int columns) {
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
    public SparseRCDoubleMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        columnIndexes = new int[nzmax];
        values = new double[nzmax];
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
     * @param value
     *            numerical value, cannot be zero
     * @param removeDuplicates
     *            if true, then duplicates (if any) are removed
     * @param sortColumnIndexes
     *            if true, then column indexes are sorted
     */
    public SparseRCDoubleMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double value,
            boolean removeDuplicates, boolean sortColumnIndexes) {
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
        if (value == 0) {
            throw new IllegalArgumentException("value cannot be 0");
        }

        int nz = Math.max(rowIndexes.length, 1);
        this.columnIndexes = new int[nz];
        this.values = new double[nz];
        this.rowPointers = new int[rows + 1];
        int[] w = new int[rows];
        int r;
        for (int k = 0; k < nz; k++) {
            w[rowIndexes[k]]++;
        }
        cumsum(this.rowPointers, w, rows);
        for (int k = 0; k < nz; k++) {
            this.columnIndexes[r = w[rowIndexes[k]]++] = columnIndexes[k];
            this.values[r] = value;
        }
        if (removeDuplicates) {
            removeDuplicates();
        }
        if (sortColumnIndexes) {
            sortColumnIndexes();
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
     * @param sortColumnIndexes
     *            if true, then column indexes are sorted
     */
    public SparseRCDoubleMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double[] values,
            boolean removeDuplicates, boolean removeZeroes, boolean sortColumnIndexes) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        if (rowIndexes.length != columnIndexes.length) {
            throw new IllegalArgumentException("rowIndexes.length != columnIndexes.length");
        } else if (rowIndexes.length != values.length) {
            throw new IllegalArgumentException("rowIndexes.length != values.length");
        }
        int nz = Math.max(rowIndexes.length, 1);
        this.columnIndexes = new int[nz];
        this.values = new double[nz];
        this.rowPointers = new int[rows + 1];
        int[] w = new int[rows];
        int r;
        for (int k = 0; k < nz; k++) {
            w[rowIndexes[k]]++;
        }
        cumsum(this.rowPointers, w, rows);
        for (int k = 0; k < nz; k++) {
            this.columnIndexes[r = w[rowIndexes[k]]++] = columnIndexes[k];
            this.values[r] = values[k];
        }
        if (removeZeroes) {
            removeZeroes();
        }
        if (removeDuplicates) {
            removeDuplicates();
        }
        if (sortColumnIndexes) {
            sortColumnIndexes();
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
    public SparseRCDoubleMatrix2D(int rows, int columns, int[] rowPointers, int[] columnIndexes, double[] values) {
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
        this.rowPointers = rowPointers;
        this.columnIndexes = columnIndexes;
        this.values = values;
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

            int nz = cardinality();
            for (int j = 0; j < nz; j++) {
                values[j] *= alpha;
            }
        } else {
            forEachNonZero(new cern.colt.function.tdouble.IntIntDoubleFunction() {
                public double apply(int i, int j, double value) {
                    return function.apply(value);
                }
            });
        }
        return this;
    }

    public DoubleMatrix2D assign(double value) {
        if (value == 0) {
            Arrays.fill(rowPointers, 0);
            Arrays.fill(columnIndexes, 0);
            Arrays.fill(values, 0);
        } else {
            int nnz = cardinality();
            for (int i = 0; i < nnz; i++) {
                values[i] = value;
            }
        }
        return this;
    }

    public DoubleMatrix2D assign(DoubleMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof SparseRCDoubleMatrix2D) {
            SparseRCDoubleMatrix2D other = (SparseRCDoubleMatrix2D) source;
            System.arraycopy(other.rowPointers, 0, rowPointers, 0, rows + 1);
            int nzmax = other.columnIndexes.length;
            if (columnIndexes.length < nzmax) {
                columnIndexes = new int[nzmax];
                values = new double[nzmax];
            }
            System.arraycopy(other.columnIndexes, 0, columnIndexes, 0, nzmax);
            System.arraycopy(other.values, 0, values, 0, nzmax);
            columnIndexesSorted = other.columnIndexesSorted;
        } else if (source instanceof SparseCCDoubleMatrix2D) {
            SparseCCDoubleMatrix2D other = ((SparseCCDoubleMatrix2D) source).getTranspose();
            rowPointers = other.getColumnPointers();
            columnIndexes = other.getRowIndexes();
            values = other.getValues();
            columnIndexesSorted = true;
        } else {
            assign(0);
            source.forEachNonZero(new cern.colt.function.tdouble.IntIntDoubleFunction() {
                public double apply(int i, int j, double value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
        }
        return this;
    }

    public DoubleMatrix2D assign(final DoubleMatrix2D y, cern.colt.function.tdouble.DoubleDoubleFunction function) {
        checkShape(y);
        if ((y instanceof SparseRCDoubleMatrix2D) && (function == cern.jet.math.tdouble.DoubleFunctions.plus)) { // x[i] = x[i] + y[i] 
            SparseRCDoubleMatrix2D yy = (SparseRCDoubleMatrix2D) y;

            final int[] rowPointersY = yy.rowPointers;
            final int[] columnIndexesY = yy.columnIndexes;
            final double[] valuesY = yy.values;

            final int[] rowPointersC = new int[rows + 1];
            int cnz = Math.max(columnIndexes.length, (int) Math.min(Integer.MAX_VALUE, (long) rowPointers[rows]
                    + (long) rowPointersY[rows]));
            final int[] columnIndexesC = new int[cnz];
            final double[] valuesC = new double[cnz];
            int nrow = rows;
            int ncol = columns;
            int nzmax = valuesC.length;
            if (function == cern.jet.math.tdouble.DoubleFunctions.plus) { // x[i] = x[i] + y[i]
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
                            valuesC[kc] = values[ka] + valuesY[kb];
                            columnIndexesC[kc] = j1;
                            ka++;
                            kb++;
                            kc++;
                        } else if (j1 < j2) {
                            columnIndexesC[kc] = j1;
                            valuesC[kc] = values[ka];
                            ka++;
                            kc++;
                        } else if (j1 > j2) {
                            columnIndexesC[kc] = j2;
                            valuesC[kc] = valuesY[kb];
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

        if (function instanceof cern.jet.math.tdouble.DoublePlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final double alpha = ((cern.jet.math.tdouble.DoublePlusMultSecond) function).multiplicator;
            if (alpha == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tdouble.IntIntDoubleFunction() {
                public double apply(int i, int j, double value) {
                    setQuick(i, j, getQuick(i, j) + alpha * value);
                    return value;
                }
            });
            return this;
        }

        if (function instanceof cern.jet.math.tdouble.DoublePlusMultFirst) { // x[i] = alpha*x[i] + y[i]
            final double alpha = ((cern.jet.math.tdouble.DoublePlusMultFirst) function).multiplicator;
            if (alpha == 0)
                return assign(y);
            y.forEachNonZero(new cern.colt.function.tdouble.IntIntDoubleFunction() {
                public double apply(int i, int j, double value) {
                    setQuick(i, j, alpha * getQuick(i, j) + value);
                    return value;
                }
            });
            return this;
        }

        if (function == cern.jet.math.tdouble.DoubleFunctions.mult) { // x[i] = x[i] * y[i]
            for (int i = rows; --i >= 0;) {
                int low = rowPointers[i];
                for (int k = rowPointers[i + 1]; --k >= low;) {
                    int j = columnIndexes[k];
                    values[k] *= y.getQuick(i, j);
                    if (values[k] == 0)
                        remove(i, j);
                }
            }
            return this;
        }

        if (function == cern.jet.math.tdouble.DoubleFunctions.div) { // x[i] = x[i] / y[i]

            for (int i = rows; --i >= 0;) {
                int low = rowPointers[i];
                for (int k = rowPointers[i + 1]; --k >= low;) {
                    int j = columnIndexes[k];
                    values[k] /= y.getQuick(i, j);
                    if (values[k] == 0)
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

    public DoubleMatrix2D forEachNonZero(final cern.colt.function.tdouble.IntIntDoubleFunction function) {

        for (int i = rows; --i >= 0;) {
            int low = rowPointers[i];
            for (int k = rowPointers[i + 1]; --k >= low;) {
                int j = columnIndexes[k];
                double value = values[k];
                double r = function.apply(i, j, value);
                if (r != value)
                    values[k] = r;
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
    public SparseCCDoubleMatrix2D getColumnCompressed() {
        SparseRCDoubleMatrix2D tr = getTranspose();
        SparseCCDoubleMatrix2D cc = new SparseCCDoubleMatrix2D(rows, columns);
        cc.dcs.i = tr.columnIndexes;
        cc.dcs.p = tr.rowPointers;
        cc.dcs.x = tr.values;
        cc.dcs.nzmax = tr.values.length;
        cc.rowIndexesSorted = true;
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
    public DenseDoubleMatrix2D getDense() {
        final DenseDoubleMatrix2D dense = new DenseDoubleMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tdouble.IntIntDoubleFunction() {
            public double apply(int i, int j, double value) {
                dense.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return dense;
    }

    public synchronized double getQuick(int row, int column) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        double v = 0;
        if (k >= 0)
            v = values[k];
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

    /**
     * Returns a new matrix that is the transpose of this matrix. This method
     * creates a new object (not a view), so changes in the returned matrix are
     * NOT reflected in this matrix.
     * 
     * @return the transpose of this matrix
     */
    public SparseRCDoubleMatrix2D getTranspose() {
        int nnz = rowPointers[rows];
        int[] w = new int[columns];
        int[] rowPointersT = new int[columns + 1];
        int[] columnIndexesT = new int[nnz];
        double[] valuesT = new double[nnz];

        for (int p = 0; p < nnz; p++) {
            w[columnIndexes[p]]++;
        }
        cumsum(rowPointersT, w, columns);
        int q;
        for (int j = 0; j < rows; j++) {
            int high = rowPointers[j + 1];
            for (int p = rowPointers[j]; p < high; p++) {
                columnIndexesT[q = w[columnIndexes[p]]++] = j;
                valuesT[q] = values[p];
            }
        }
        SparseRCDoubleMatrix2D T = new SparseRCDoubleMatrix2D(columns, rows);
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

    /**
     * Returns true if column indexes are sorted, false otherwise
     * 
     * @return true if column indexes are sorted, false otherwise
     */
    public boolean hasColumnIndexesSorted() {
        return columnIndexesSorted;
    }

    public DoubleMatrix2D like(int rows, int columns) {
        return new SparseRCDoubleMatrix2D(rows, columns);
    }

    public DoubleMatrix1D like1D(int size) {
        return new SparseDoubleMatrix1D(size);
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
                    values[w[i]] += values[p]; /* A(i,j) is a duplicate */
                } else {
                    w[i] = nz; /* record where column i occurs */
                    columnIndexes[nz] = i; /* keep A(i,j) */
                    values[nz++] = values[p];
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
        for (int j = 0; j < rows; j++) {
            int p = rowPointers[j]; /* get current location of row j */
            rowPointers[j] = nz; /* record new location of row j */
            for (; p < rowPointers[j + 1]; p++) {
                if (Math.abs(values[p]) > eps) {
                    values[nz] = values[p]; /* keep A(i,j) */
                    columnIndexes[nz++] = columnIndexes[p];
                }
            }
        }
        rowPointers[rows] = nz; /* finalize A */
    }

    public synchronized void setQuick(int row, int column, double value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        if (k >= 0) { // found
            if (value == 0)
                remove(row, k);
            else
                values[k] = value;
            return;
        }

        if (value != 0) {
            k = -k - 1;
            insert(row, column, k, value);
        }
    }

    /**
     * Sorts column indexes
     */
    public void sortColumnIndexes() {
        SparseRCDoubleMatrix2D T = getTranspose();
        this.rows = T.rows;
        this.columns = T.columns;
        this.columnIndexes = T.columnIndexes;
        this.rowPointers = T.rowPointers;
        this.values = T.values;
        //        System.arraycopy(T.columnIndexes, 0, this.columnIndexes, 0, T.columnIndexes.length);
        //        System.arraycopy(T.rowPointers, 0, this.rowPointers, 0, T.rowPointers.length);
        //        System.arraycopy(T.values, 0, this.values, 0, T.values.length);
        T = getTranspose();
        this.rows = T.rows;
        this.columns = T.columns;
        this.columnIndexes = T.columnIndexes;
        this.rowPointers = T.rowPointers;
        this.values = T.values;
        columnIndexesSorted = true;
        //        System.arraycopy(T.columnIndexes, 0, this.columnIndexes, 0, T.columnIndexes.length);
        //        System.arraycopy(T.rowPointers, 0, this.rowPointers, 0, T.rowPointers.length);
        //        System.arraycopy(T.values, 0, this.values, 0, T.values.length);
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(rows).append(" x ").append(columns).append(" sparse matrix, nnz = ").append(cardinality())
                .append('\n');
        for (int i = 0; i < rows; i++) {
            int high = rowPointers[i + 1];
            for (int j = rowPointers[i]; j < high; j++) {
                builder.append('(').append(i).append(',').append(columnIndexes[j]).append(')').append('\t').append(
                        values[j]).append('\n');
            }
        }
        return builder.toString();
    }

    public void trimToSize() {
        realloc(0);
    }

    public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, final double alpha, final double beta,
            final boolean transposeA) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;

        boolean ignore = (z == null || !transposeA);
        if (z == null)
            z = new DenseDoubleMatrix1D(rowsA);

        if (!(y instanceof DenseDoubleMatrix1D && z instanceof DenseDoubleMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        DenseDoubleMatrix1D zz = (DenseDoubleMatrix1D) z;
        final double[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) z.index(0);

        DenseDoubleMatrix1D yy = (DenseDoubleMatrix1D) y;
        final double[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) y.index(0);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();

        if (transposeA) {
            if ((!ignore) && (beta != 1.0))
                z.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta));

            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = 2;
                Future<?>[] futures = new Future[nthreads];
                final double[] result = new double[rowsA];
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    final int threadID = j;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            if (threadID == 0) {
                                for (int i = firstRow; i < lastRow; i++) {
                                    int high = rowPointers[i + 1];
                                    double yElem = alpha * elementsY[zeroY + strideY * i];
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = columnIndexes[k];
                                        elementsZ[zeroZ + strideZ * j] += values[k] * yElem;
                                    }
                                }
                            } else {
                                for (int i = firstRow; i < lastRow; i++) {
                                    int high = rowPointers[i + 1];
                                    double yElem = alpha * elementsY[zeroY + strideY * i];
                                    for (int k = rowPointers[i]; k < high; k++) {
                                        int j = columnIndexes[k];
                                        result[j] += values[k] * yElem;
                                    }
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
                int rem = rowsA % 10;
                for (int j = rem; j < rowsA; j += 10) {
                    elementsZ[zeroZ + j * strideZ] += result[j];
                    elementsZ[zeroZ + (j + 1) * strideZ] += result[j + 1];
                    elementsZ[zeroZ + (j + 2) * strideZ] += result[j + 2];
                    elementsZ[zeroZ + (j + 3) * strideZ] += result[j + 3];
                    elementsZ[zeroZ + (j + 4) * strideZ] += result[j + 4];
                    elementsZ[zeroZ + (j + 5) * strideZ] += result[j + 5];
                    elementsZ[zeroZ + (j + 6) * strideZ] += result[j + 6];
                    elementsZ[zeroZ + (j + 7) * strideZ] += result[j + 7];
                    elementsZ[zeroZ + (j + 8) * strideZ] += result[j + 8];
                    elementsZ[zeroZ + (j + 9) * strideZ] += result[j + 9];
                }
                for (int j = 0; j < rem; j++) {
                    elementsZ[zeroZ + j * strideZ] += result[j];
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    int high = rowPointers[i + 1];
                    double yElem = alpha * elementsY[zeroY + strideY * i];
                    for (int k = rowPointers[i]; k < high; k++) {
                        int j = columnIndexes[k];
                        elementsZ[zeroZ + strideZ * j] += values[k] * yElem;
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
                        int k = rowPointers[firstRow];
                        if (beta == 0.0) {
                            for (int i = firstRow; i < lastRow; i++) {
                                double sum = 0;
                                int high = rowPointers[i + 1];
                                for (; k + 10 < high; k += 10) {
                                    int ind = k + 9;
                                    sum += values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]];
                                }
                                for (; k < high; k++) {
                                    sum += values[k] * elementsY[columnIndexes[k]];
                                }
                                elementsZ[zidx] = alpha * sum;
                                zidx += strideZ;
                            }
                        } else {
                            for (int i = firstRow; i < lastRow; i++) {
                                double sum = 0;
                                int high = rowPointers[i + 1];
                                for (; k + 10 < high; k += 10) {
                                    int ind = k + 9;
                                    sum += values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]]
                                            + values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]];
                                }
                                for (; k < high; k++) {
                                    sum += values[k] * elementsY[columnIndexes[k]];
                                }
                                elementsZ[zidx] = alpha * sum + beta * elementsZ[zidx];
                                zidx += strideZ;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int zidx = zeroZ;
            int k = rowPointers[0];
            if (beta == 0.0) {
                for (int i = 0; i < rows; i++) {
                    double sum = 0;
                    int high = rowPointers[i + 1];
                    for (; k + 10 < high; k += 10) {
                        int ind = k + 9;
                        sum += values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]];
                    }
                    for (; k < high; k++) {
                        sum += values[k] * elementsY[columnIndexes[k]];
                    }
                    elementsZ[zidx] = alpha * sum;
                    zidx += strideZ;
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    double sum = 0;
                    int high = rowPointers[i + 1];
                    for (; k + 10 < high; k += 10) {
                        int ind = k + 9;
                        sum += values[ind] * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]] + values[ind]
                                * elementsY[zeroY + strideY * columnIndexes[ind--]];
                    }
                    for (; k < high; k++) {
                        sum += values[k] * elementsY[columnIndexes[k]];
                    }
                    elementsZ[zidx] = alpha * sum + beta * elementsZ[zidx];
                    zidx += strideZ;
                }
            }
        }
        return z;
    }

    public DoubleMatrix2D zMult(DoubleMatrix2D B, DoubleMatrix2D C, final double alpha, double beta,
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
            if (B instanceof SparseRCDoubleMatrix2D) {
                C = new SparseRCDoubleMatrix2D(rowsA, p, (rowsA * p));
            } else {
                C = new DenseDoubleMatrix2D(rowsA, p);
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

        if (!ignore && beta != 1.0) {
            C.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta));
        }

        if ((B instanceof DenseDoubleMatrix2D) && (C instanceof DenseDoubleMatrix2D)) {
            SparseRCDoubleMatrix2D AA;
            if (transposeA) {
                AA = getTranspose();
            } else {
                AA = this;
            }
            DenseDoubleMatrix2D BB;
            if (transposeB) {
                BB = (DenseDoubleMatrix2D) B.viewDice();
            } else {
                BB = (DenseDoubleMatrix2D) B;
            }

            DenseDoubleMatrix2D CC = (DenseDoubleMatrix2D) C;
            int[] rowPointersA = AA.rowPointers;
            int[] columnIndexesA = AA.columnIndexes;
            double[] valuesA = AA.values;

            for (int ii = 0; ii < rowsA; ii++) {
                int highA = rowPointersA[ii + 1];
                for (int ka = rowPointersA[ii]; ka < highA; ka++) {
                    double scal = valuesA[ka] * alpha;
                    int jj = columnIndexesA[ka];
                    CC.viewRow(ii).assign(BB.viewRow(jj), DoubleFunctions.plusMultSecond(scal));
                }
            }
        } else if ((B instanceof SparseRCDoubleMatrix2D) && (C instanceof SparseRCDoubleMatrix2D)) {
            SparseRCDoubleMatrix2D AA;
            SparseRCDoubleMatrix2D BB;
            SparseRCDoubleMatrix2D CC = (SparseRCDoubleMatrix2D) C;
            if (transposeA) {
                AA = getTranspose();
            } else {
                AA = this;
            }
            if (transposeB) {
                BB = ((SparseRCDoubleMatrix2D) B).getTranspose();
            } else {
                BB = (SparseRCDoubleMatrix2D) B;
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
            int nzmax = valuesC.length;

            int[] iw = new int[columnsB + 1];
            for (int i = 0; i < iw.length; i++) {
                iw[i] = -1;
            }
            int len = -1;
            for (int ii = 0; ii < rowsA; ii++) {
                int highA = rowPointersA[ii + 1];
                for (int ka = rowPointersA[ii]; ka < highA; ka++) {
                    double scal = valuesA[ka] * alpha;
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
                            valuesC[len] = scal * valuesB[kb];
                        } else {
                            valuesC[jpos] += scal * valuesB[kb];
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
                //                DoubleMatrix1D valuesCPart = valuesC.viewPart(rowPointersC[ii], length).viewSelection(indexes);
                //                valuesC.viewPart(rowPointersC[ii], length).assign(valuesCPart);
            }
            //            CC.columnIndexes.elements((int[]) columnIndexesC.elements());
            //            CC.columnIndexes.setSize(columnIndexesSize);
            //            CC.values.elements((double[]) valuesC.elements());
            //            CC.values.setSize(columnIndexesSize);
        } else {
            if (transposeB) {
                B = B.viewDice();
            }
            // cache views
            final DoubleMatrix1D[] Brows = new DoubleMatrix1D[columnsA];
            for (int i = columnsA; --i >= 0;)
                Brows[i] = B.viewRow(i);
            final DoubleMatrix1D[] Crows = new DoubleMatrix1D[rowsA];
            for (int i = rowsA; --i >= 0;)
                Crows[i] = C.viewRow(i);

            final cern.jet.math.tdouble.DoublePlusMultSecond fun = cern.jet.math.tdouble.DoublePlusMultSecond
                    .plusMult(0);

            final int[] columnIndexesA = columnIndexes;
            final double[] valuesA = values;
            for (int i = rows; --i >= 0;) {
                int low = rowPointers[i];
                for (int k = rowPointers[i + 1]; --k >= low;) {
                    int j = columnIndexesA[k];
                    fun.multiplicator = valuesA[k] * alpha;
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
        double[] valuesNew = new double[nzmax];
        length = Math.min(nzmax, values.length);
        System.arraycopy(values, 0, valuesNew, 0, length);
        values = valuesNew;
    }

    protected DoubleMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, double value) {
        IntArrayList columnIndexesList = new IntArrayList(columnIndexes);
        columnIndexesList.setSizeRaw(rowPointers[rows]);
        DoubleArrayList valuesList = new DoubleArrayList(values);
        valuesList.setSizeRaw(rowPointers[rows]);
        columnIndexesList.beforeInsert(index, column);
        valuesList.beforeInsert(index, value);
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
        valuesList.remove(index);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]--;
        columnIndexes = columnIndexesList.elements();
        values = valuesList.elements();
    }

}
