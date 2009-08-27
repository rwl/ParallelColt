/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import java.util.Arrays;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse column-compressed 2-d matrix holding <tt>int</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
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
 * for (int column = 0; column &lt; columns; column++) {
 *     for (int row = 0; row &lt; rows; row++) {
 *         if (someCondition)
 *             matrix.setQuick(row, column, someValue);
 *     }
 * }
 * 
 * // poor
 * matrix.assign(0);
 * for (int column = columns; --column &gt;= 0;) {
 *     for (int row = rows; --row &gt;= 0;) {
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
 * A.forEachNonZero(new cern.colt.function.IntIntIntFunction() {
 *     public int apply(int row, int column, int value) {
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
 * B.forEachNonZero(new cern.colt.function.IntIntIntFunction() {
 *     public int apply(int row, int column, int value) {
 *         A.setQuick(row, column, A.getQuick(row, column) + alpha * value);
 *         return value;
 *     }
 * });
 * </pre>
 * 
 * </td>
 * </table>
 * Method {@link #assign(IntMatrix2D,cern.colt.function.tint.IntIntFunction)}
 * does just that if you supply
 * {@link cern.jet.math.tint.IntFunctions#plusMultSecond} as argument.
 * 
 * 
 * @author Piotr Wendykier
 * 
 */
public class SparseCCIntMatrix2D extends WrapperIntMatrix2D {
    private static final long serialVersionUID = 1L;
    /*
     * Internal storage.
     */
    protected int[] columnPointers;

    protected int[] rowIndexes;

    protected int[] values;

    protected boolean rowIndexesSorted = false;

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
    public SparseCCIntMatrix2D(int[][] values) {
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
    public SparseCCIntMatrix2D(int rows, int columns) {
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
    public SparseCCIntMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        rowIndexes = new int[nzmax];
        values = new int[nzmax];
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
     * @param value
     *            numerical value
     * @param removeDuplicates
     *            if true, then duplicates (if any) are removed
     * @param sortRowIndexes
     *            if true, then row indexes are sorted
     */
    public SparseCCIntMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, int value,
            boolean removeDuplicates, boolean sortRowIndexes) {
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
        this.rowIndexes = new int[nz];
        this.values = new int[nz];
        this.columnPointers = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nz; k++) {
            w[columnIndexes[k]]++;
        }
        cumsum(this.columnPointers, w, columns);
        for (int k = 0; k < nz; k++) {
            this.rowIndexes[r = w[columnIndexes[k]]++] = rowIndexes[k];
            this.values[r] = value;
        }
        if (removeDuplicates) {
            removeDuplicates();
        }
        if (sortRowIndexes) {
            sortRowIndexes();
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
     * @param sortRowIndexes
     *            if true, then row indexes are sorted
     */
    public SparseCCIntMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, int[] values,
            boolean removeDuplicates, boolean removeZeroes, boolean sortRowIndexes) {
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
        this.rowIndexes = new int[nz];
        this.values = new int[nz];
        this.columnPointers = new int[columns + 1];
        int[] w = new int[columns];
        int r;
        for (int k = 0; k < nz; k++) {
            w[columnIndexes[k]]++;
        }
        cumsum(this.columnPointers, w, columns);
        for (int k = 0; k < nz; k++) {
            this.rowIndexes[r = w[columnIndexes[k]]++] = rowIndexes[k];
            this.values[r] = values[k];
        }
        if (removeDuplicates) {
            removeDuplicates();
        }
        if (sortRowIndexes) {
            sortRowIndexes();
        }
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

            final int[] valuesE = values;
            int nz = cardinality();
            for (int j = 0; j < nz; j++) {
                valuesE[j] *= alpha;
            }
        } else {
            forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    return function.apply(value);
                }
            });
        }
        return this;
    }

    public IntMatrix2D assign(int value) {
        if (value == 0) {
            Arrays.fill(rowIndexes, 0);
            Arrays.fill(columnPointers, 0);
            Arrays.fill(values, 0);
        } else {
            int nnz = cardinality();
            for (int i = 0; i < nnz; i++) {
                values[i] = value;
            }
        }
        return this;
    }

    public IntMatrix2D assign(IntMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof SparseCCIntMatrix2D) {
            SparseCCIntMatrix2D other = (SparseCCIntMatrix2D) source;
            System.arraycopy(other.getColumnPointers(), 0, columnPointers, 0, columns + 1);
            int nzmax = other.getRowIndexes().length;
            if (rowIndexes.length < nzmax) {
                rowIndexes = new int[nzmax];
                values = new int[nzmax];
            }
            System.arraycopy(other.getRowIndexes(), 0, rowIndexes, 0, nzmax);
            System.arraycopy(other.getValues(), 0, values, 0, nzmax);
            rowIndexesSorted = other.rowIndexesSorted;
        } else if (source instanceof SparseRCIntMatrix2D) {
            SparseRCIntMatrix2D other = ((SparseRCIntMatrix2D) source).getTranspose();
            columnPointers = other.getRowPointers();
            rowIndexes = other.getColumnIndexes();
            values = other.getValues();
            rowIndexesSorted = true;
        } else {
            assign(0);
            source.forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
        }
        return this;
    }

    public IntMatrix2D assign(final IntMatrix2D y, cern.colt.function.tint.IntIntFunction function) {
        checkShape(y);

        if ((y instanceof SparseCCIntMatrix2D) && (function == cern.jet.math.tint.IntFunctions.plus)) { // x[i] = x[i] + y[i] 
            SparseCCIntMatrix2D yy = (SparseCCIntMatrix2D) y;
            int p, j, nz = 0, anz;
            int Cp[], Ci[], Bp[], m, n, bnz, w[];
            int x[], Cx[];
            m = rows;
            anz = columnPointers[columns];
            n = yy.columns;
            Bp = yy.columnPointers;
            bnz = Bp[n];
            w = new int[m]; /* get workspace */
            x = new int[m]; /* get workspace */
            SparseCCIntMatrix2D C = new SparseCCIntMatrix2D(m, n, anz + bnz); /* allocate result*/
            Cp = C.columnPointers;
            Ci = C.rowIndexes;
            Cx = C.values;
            for (j = 0; j < n; j++) {
                Cp[j] = nz; /* column j of C starts here */
                nz = scatter(this, j, 1, w, x, j + 1, C, nz); /* alpha*A(:,j)*/
                nz = scatter(yy, j, 1, w, x, j + 1, C, nz); /* beta*B(:,j) */
                for (p = Cp[j]; p < nz; p++)
                    Cx[p] = x[Ci[p]];
            }
            Cp[n] = nz; /* finalize the last column of C */
            rowIndexes = Ci;
            columnPointers = Cp;
            values = Cx;
            return this;
        }

        if (function instanceof cern.jet.math.tint.IntPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final int alpha = ((cern.jet.math.tint.IntPlusMultSecond) function).multiplicator;
            if (alpha == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    setQuick(i, j, getQuick(i, j) + alpha * value);
                    return value;
                }
            });
            return this;
        }

        if (function instanceof cern.jet.math.tint.IntPlusMultFirst) { // x[i] = alpha*x[i] + y[i]
            final int alpha = ((cern.jet.math.tint.IntPlusMultFirst) function).multiplicator;
            if (alpha == 0)
                return assign(y);
            y.forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
                public int apply(int i, int j, int value) {
                    setQuick(i, j, alpha * getQuick(i, j) + value);
                    return value;
                }
            });
            return this;
        }

        if (function == cern.jet.math.tint.IntFunctions.mult) { // x[i] = x[i] * y[i]
            final int[] rowIndexesA = rowIndexes;
            final int[] columnPointersA = columnPointers;
            final int[] valuesA = values;
            for (int j = columns; --j >= 0;) {
                int low = columnPointersA[j];
                for (int k = columnPointersA[j + 1]; --k >= low;) {
                    int i = rowIndexesA[k];
                    valuesA[k] *= y.getQuick(i, j);
                    if (valuesA[k] == 0)
                        remove(i, j);
                }
            }
            return this;
        }

        if (function == cern.jet.math.tint.IntFunctions.div) { // x[i] = x[i] / y[i]
            final int[] rowIndexesA = rowIndexes;
            final int[] columnPointersA = columnPointers;
            final int[] valuesA = values;

            for (int j = columns; --j >= 0;) {
                int low = columnPointersA[j];
                for (int k = columnPointersA[j + 1]; --k >= low;) {
                    int i = rowIndexesA[k];
                    valuesA[k] /= y.getQuick(i, j);
                    if (valuesA[k] == 0)
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

    public IntMatrix2D forEachNonZero(final cern.colt.function.tint.IntIntIntFunction function) {
        final int[] rowIndexesA = rowIndexes;
        final int[] columnPointersA = columnPointers;
        final int[] valuesA = values;

        for (int j = columns; --j >= 0;) {
            int low = columnPointersA[j];
            for (int k = columnPointersA[j + 1]; --k >= low;) {
                int i = rowIndexesA[k];
                int value = valuesA[k];
                int r = function.apply(i, j, value);
                valuesA[k] = r;
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
    public DenseIntMatrix2D getDense() {
        final DenseIntMatrix2D dense = new DenseIntMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tint.IntIntIntFunction() {
            public int apply(int i, int j, int value) {
                dense.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return dense;
    }

    public synchronized int getQuick(int row, int column) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(rowIndexes, row, columnPointers[column], columnPointers[column + 1] - 1);
        int v = 0;
        if (k >= 0)
            v = values[k];
        return v;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a row-compressed form. This method creates a new object (not a view), so
     * changes in the returned matrix are NOT reflected in this matrix.
     * 
     * @return this matrix in a row-compressed form
     */
    public SparseRCIntMatrix2D getRowCompressed() {
        SparseCCIntMatrix2D tr = getTranspose();
        SparseRCIntMatrix2D rc = new SparseRCIntMatrix2D(rows, columns);
        rc.columnIndexes = tr.rowIndexes;
        rc.rowPointers = tr.columnPointers;
        rc.values = tr.values;
        rc.columnIndexesSorted = true;
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

    /**
     * Returns a new matrix that is the transpose of this matrix. This method
     * creates a new object (not a view), so changes in the returned matrix are
     * NOT reflected in this matrix.
     * 
     * @return the transpose of this matrix
     */
    public SparseCCIntMatrix2D getTranspose() {
        int p, q, j, Cp[], Ci[], n, m, Ap[], Ai[], w[];
        int Cx[], Ax[];
        m = rows;
        n = columns;
        Ap = columnPointers;
        Ai = rowIndexes;
        Ax = values;
        SparseCCIntMatrix2D C = new SparseCCIntMatrix2D(columns, rows, Ai.length); /* allocate result */
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
                Cx[q] = Ax[p];
            }
        }
        return C;
    }

    /**
     * Returns numerical values
     * 
     * @return numerical values
     */
    public int[] getValues() {
        return values;
    }

    /**
     * Returns true if row indexes are sorted, false otherwise
     * 
     * @return true if row indexes are sorted, false otherwise
     */
    public boolean hasRowIndexesSorted() {
        return rowIndexesSorted;
    }

    public IntMatrix2D like(int rows, int columns) {
        return new SparseCCIntMatrix2D(rows, columns);
    }

    public IntMatrix1D like1D(int size) {
        return new SparseIntMatrix1D(size);
    }

    public synchronized void setQuick(int row, int column, int value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(rowIndexes, row, columnPointers[column], columnPointers[column + 1] - 1);

        if (k >= 0) { // found
            if (value == 0)
                remove(column, k);
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
     * Sorts row indexes
     */
    public void sortRowIndexes() {
        SparseCCIntMatrix2D tr = getTranspose();
        tr = tr.getTranspose();
        columnPointers = tr.columnPointers;
        rowIndexes = tr.rowIndexes;
        values = tr.values;
        rowIndexesSorted = true;
    }

    /**
     * Removes (sums) duplicate entries (if any}
     */
    public void removeDuplicates() {
        int i, j, p, q, nz = 0, n, m, Ap[], Ai[], w[];
        int Ax[];
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
                    Ax[nz++] = Ax[p];
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
        int Ax[];
        n = columns;
        Ap = columnPointers;
        Ai = rowIndexes;
        Ax = values;
        for (j = 0; j < n; j++) {
            p = Ap[j]; /* get current location of col j */
            Ap[j] = nz; /* record new location of col j */
            for (; p < Ap[j + 1]; p++) {
                if (Ax[p] != 0) {
                    Ax[nz] = Ax[p]; /* keep A(i,j) */
                    Ai[nz++] = Ai[p];
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
                builder.append('(').append(rowIndexes[j]).append(',').append(i).append(')').append('\t').append(
                        values[j]).append('\n');
            }
        }
        return builder.toString();
    }

    public IntMatrix1D zMult(IntMatrix1D y, IntMatrix1D z, final int alpha, final int beta, final boolean transposeA) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;

        boolean ignore = (z == null || transposeA);
        if (z == null)
            z = new DenseIntMatrix1D(rowsA);

        if (!(y instanceof DenseIntMatrix1D && z instanceof DenseIntMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        DenseIntMatrix1D zz = (DenseIntMatrix1D) z;
        final int[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) zz.index(0);

        DenseIntMatrix1D yy = (DenseIntMatrix1D) y;
        final int[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) yy.index(0);

        final int[] rowIndexesA = rowIndexes;
        final int[] columnPointersA = columnPointers;
        final int[] valuesA = values;

        int zidx = zeroZ;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (!transposeA) {
            if ((!ignore) && (beta != 1)) {
                z.assign(cern.jet.math.tint.IntFunctions.mult(beta));
            }

            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = 2;
                Future<?>[] futures = new Future[nthreads];
                final int[] result = new int[rowsA];
                int k = columns / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstColumn = j * k;
                    final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                    final int threadID = j;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            if (threadID == 0) {
                                for (int i = firstColumn; i < lastColumn; i++) {
                                    int high = columnPointersA[i + 1];
                                    int yElem = elementsY[zeroY + strideY * i];
                                    for (int k = columnPointersA[i]; k < high; k++) {
                                        int j = rowIndexesA[k];
                                        elementsZ[zeroZ + strideZ * j] += alpha * valuesA[k] * yElem;
                                    }
                                }
                            } else {
                                for (int i = firstColumn; i < lastColumn; i++) {
                                    int high = columnPointersA[i + 1];
                                    int yElem = elementsY[zeroY + strideY * i];
                                    for (int k = columnPointersA[i]; k < high; k++) {
                                        int j = rowIndexesA[k];
                                        result[j] += alpha * valuesA[k] * yElem;
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
                for (int i = 0; i < columns; i++) {
                    int high = columnPointersA[i + 1];
                    int yElem = elementsY[zeroY + strideY * i];
                    for (int k = columnPointersA[i]; k < high; k++) {
                        int j = rowIndexesA[k];
                        elementsZ[zeroZ + strideZ * j] += alpha * valuesA[k] * yElem;
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
                            int k = columnPointers[firstColumn];
                            for (int i = firstColumn; i < lastColumn; i++) {
                                int sum = 0;
                                int high = columnPointers[i + 1];
                                for (; k + 10 < high; k += 10) {
                                    int ind = k + 9;
                                    sum += valuesA[ind] * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * rowIndexes[ind--]];
                                }
                                for (; k < high; k++) {
                                    sum += valuesA[k] * elementsY[rowIndexes[k]];
                                }
                                elementsZ[zidx] = alpha * sum + beta * elementsZ[zidx];
                                zidx += strideZ;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int k = columnPointers[0];
                for (int i = 0; i < columns; i++) {
                    int sum = 0;
                    int high = columnPointers[i + 1];
                    for (; k + 10 < high; k += 10) {
                        int ind = k + 9;
                        sum += valuesA[ind] * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * rowIndexes[ind--]];
                    }
                    for (; k < high; k++) {
                        sum += valuesA[k] * elementsY[rowIndexes[k]];
                    }
                    elementsZ[zidx] = alpha * sum + beta * elementsZ[zidx];
                    zidx += strideZ;
                }
            }
        }
        return z;
    }

    public IntMatrix2D zMult(IntMatrix2D B, IntMatrix2D C, final int alpha, int beta, final boolean transposeA,
            boolean transposeB) {
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
            if (B instanceof SparseCCIntMatrix2D) {
                C = new SparseCCIntMatrix2D(rowsA, p, (rowsA * p));
            } else {
                C = new DenseIntMatrix2D(rowsA, p);
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
            C.assign(cern.jet.math.tint.IntFunctions.mult(beta));
        }

        if ((B instanceof DenseIntMatrix2D) && (C instanceof DenseIntMatrix2D)) {
            SparseCCIntMatrix2D AA;
            if (transposeA) {
                AA = getTranspose();
            } else {
                AA = this;
            }
            DenseIntMatrix2D BB;
            if (transposeB) {
                BB = (DenseIntMatrix2D) B.viewDice();
            } else {
                BB = (DenseIntMatrix2D) B;
            }
            DenseIntMatrix2D CC = (DenseIntMatrix2D) C;
            int[] columnPointersA = AA.columnPointers;
            int[] rowIndexesA = AA.rowIndexes;
            int[] valuesA = AA.values;

            int zeroB = (int) BB.index(0, 0);
            int rowStrideB = BB.rowStride();
            int columnStrideB = BB.columnStride();
            int[] elementsB = BB.elements;

            int zeroC = (int) CC.index(0, 0);
            int rowStrideC = CC.rowStride();
            int columnStrideC = CC.columnStride();
            int[] elementsC = CC.elements;

            for (int jj = 0; jj < columnsB; jj++) {
                for (int kk = 0; kk < columnsA; kk++) {
                    int high = columnPointersA[kk + 1];
                    int yElem = elementsB[zeroB + kk * rowStrideB + jj * columnStrideB];
                    for (int ii = columnPointersA[kk]; ii < high; ii++) {
                        int j = rowIndexesA[ii];
                        elementsC[zeroC + j * rowStrideC + jj * columnStrideC] += valuesA[ii] * yElem;
                    }
                }
            }
            if (alpha != 1.0) {
                C.assign(cern.jet.math.tint.IntFunctions.mult(alpha));
            }

        } else if ((B instanceof SparseCCIntMatrix2D) && (C instanceof SparseCCIntMatrix2D)) {
            SparseCCIntMatrix2D AA;
            if (transposeA) {
                AA = getTranspose();
            } else {
                AA = this;
            }
            SparseCCIntMatrix2D BB = (SparseCCIntMatrix2D) B;
            if (transposeB) {
                BB = BB.getTranspose();
            }
            SparseCCIntMatrix2D CC = (SparseCCIntMatrix2D) C;
            int j, nz = 0, Cp[], Ci[], Bp[], m, n, w[], Bi[];
            int x[], Bx[], Cx[];
            m = rowsA;
            n = columnsB;
            Bp = BB.columnPointers;
            Bi = BB.rowIndexes;
            Bx = BB.values;
            w = new int[m]; /* get workspace */
            x = new int[m]; /* get workspace */
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
                    int[] valuesNew = new int[nzmaxC];
                    System.arraycopy(Cx, 0, valuesNew, 0, Cx.length);
                    Cx = valuesNew;
                }
                Cp[j] = nz; /* column j of C starts here */
                for (p = Bp[j]; p < Bp[j + 1]; p++) {
                    nz = scatter(AA, Bi[p], Bx[p], w, x, j + 1, CC, nz);
                }
                for (p = Cp[j]; p < nz; p++)
                    Cx[p] = x[Ci[p]];
            }
            Cp[n] = nz; /* finalize the last column of C */
            if (alpha != 1.0) {
                CC.assign(cern.jet.math.tint.IntFunctions.mult(alpha));
            }
        } else {
            if (transposeB) {
                B = B.viewDice();
            }
            // cache views
            final IntMatrix1D[] Brows = new IntMatrix1D[columnsA];
            for (int i = columnsA; --i >= 0;)
                Brows[i] = B.viewRow(i);
            final IntMatrix1D[] Crows = new IntMatrix1D[rowsA];
            for (int i = rowsA; --i >= 0;)
                Crows[i] = C.viewRow(i);

            final cern.jet.math.tint.IntPlusMultSecond fun = cern.jet.math.tint.IntPlusMultSecond.plusMult(0);

            final int[] rowIndexesA = rowIndexes;
            final int[] columnPointersA = columnPointers;
            final int[] valuesA = values;
            for (int i = columns; --i >= 0;) {
                int low = columnPointersA[i];
                for (int k = columnPointersA[i + 1]; --k >= low;) {
                    int j = rowIndexesA[k];
                    fun.multiplicator = valuesA[k] * alpha;
                    if (!transposeA)
                        Crows[j].assign(Brows[i], fun);
                    else
                        Crows[i].assign(Brows[j], fun);
                }
            }
        }
        return C;
    }

    protected IntMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, int value) {
        IntArrayList rowIndexesList = new IntArrayList(rowIndexes);
        rowIndexesList.setSizeRaw(columnPointers[columns]);
        IntArrayList valuesList = new IntArrayList(values);
        valuesList.setSizeRaw(columnPointers[columns]);
        rowIndexesList.beforeInsert(index, row);
        valuesList.beforeInsert(index, value);
        for (int i = columnPointers.length; --i > column;)
            columnPointers[i]++;
        rowIndexes = rowIndexesList.elements();
        values = valuesList.elements();
    }

    protected void remove(int column, int index) {
        IntArrayList rowIndexesList = new IntArrayList(rowIndexes);
        IntArrayList valuesList = new IntArrayList(values);
        rowIndexesList.remove(index);
        valuesList.remove(index);
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

    private int cumsum(int[] p, int[] c, int n) {
        int nz = 0;
        int nz2 = 0;
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
        int[] valuesNew = new int[nzmax];
        length = Math.min(nzmax, values.length);
        System.arraycopy(values, 0, valuesNew, 0, length);
        values = valuesNew;
    }

    private int scatter(SparseCCIntMatrix2D A, int j, int beta, int[] w, int[] x, int mark, SparseCCIntMatrix2D C,
            int nz) {
        int i, p;
        int Ap[], Ai[], Ci[];
        int[] Ax;
        Ap = A.columnPointers;
        Ai = A.rowIndexes;
        Ax = A.values;
        Ci = C.rowIndexes;
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p]; /* A(i,j) is nonzero */
            if (w[i] < mark) {
                w[i] = mark; /* i is new entry in column j */
                Ci[nz++] = i; /* add i to pattern of C(:,j) */
                if (x != null)
                    x[i] = beta * Ax[p]; /* x(i) = beta*A(i,j) */
            } else if (x != null)
                x[i] += beta * Ax[p]; /* i exists in C(:,j) already */
        }
        return nz;
    }
}
