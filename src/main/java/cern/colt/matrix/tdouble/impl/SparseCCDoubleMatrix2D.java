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
import edu.emory.mathcs.csparsej.tdouble.Dcs_add;
import edu.emory.mathcs.csparsej.tdouble.Dcs_cumsum;
import edu.emory.mathcs.csparsej.tdouble.Dcs_dropzeros;
import edu.emory.mathcs.csparsej.tdouble.Dcs_dupl;
import edu.emory.mathcs.csparsej.tdouble.Dcs_multiply;
import edu.emory.mathcs.csparsej.tdouble.Dcs_transpose;
import edu.emory.mathcs.csparsej.tdouble.Dcs_util;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse column-compressed 2-d matrix holding <tt>double</tt> elements. First
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
 * @author Piotr Wendykier
 * 
 */
public class SparseCCDoubleMatrix2D extends WrapperDoubleMatrix2D {
    private static final long serialVersionUID = 1L;
    /*
     * Internal storage.
     */
    protected Dcs dcs;

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
    public SparseCCDoubleMatrix2D(double[][] values) {
        this(values.length, values[0].length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given internal storage.
     * 
     * @param dcs
     *            internal storage.
     */
    public SparseCCDoubleMatrix2D(Dcs dcs) {
        super(null);
        try {
            setUp(dcs.m, dcs.n);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.dcs = dcs;
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
    public SparseCCDoubleMatrix2D(int rows, int columns) {
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
    public SparseCCDoubleMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        dcs = Dcs_util.cs_spalloc(rows, columns, nzmax, true, false);
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
    public SparseCCDoubleMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double value,
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
        dcs = Dcs_util.cs_spalloc(rows, columns, nz, true, false);
        int[] w = new int[columns];
        int[] Cp = dcs.p;
        int[] Ci = dcs.i;
        double[] Cx = dcs.x;
        for (int k = 0; k < nz; k++)
            w[columnIndexes[k]]++;
        Dcs_cumsum.cs_cumsum(Cp, w, columns);
        int p;
        for (int k = 0; k < nz; k++) {
            Ci[p = w[columnIndexes[k]]++] = rowIndexes[k];
            if (Cx != null)
                Cx[p] = value;
        }
        if (removeDuplicates) {
            if (!Dcs_dupl.cs_dupl(dcs)) { //remove duplicates
                throw new IllegalArgumentException("Exception occured in cs_dupl()!");
            }
        }
        if (sortRowIndexes) {
            //sort row indexes
            dcs = Dcs_transpose.cs_transpose(dcs, true);
            dcs = Dcs_transpose.cs_transpose(dcs, true);
            if (dcs == null) {
                throw new IllegalArgumentException("Exception occured in cs_transpose()!");
            }
            rowIndexesSorted = true;
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
    public SparseCCDoubleMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, double[] values,
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
        dcs = Dcs_util.cs_spalloc(rows, columns, nz, true, false);
        int[] w = new int[columns];
        int[] Cp = dcs.p;
        int[] Ci = dcs.i;
        double[] Cx = dcs.x;
        for (int k = 0; k < nz; k++)
            w[columnIndexes[k]]++;
        Dcs_cumsum.cs_cumsum(Cp, w, columns);
        int p;
        for (int k = 0; k < nz; k++) {
            Ci[p = w[columnIndexes[k]]++] = rowIndexes[k];
            if (Cx != null)
                Cx[p] = values[k];
        }
        if (removeZeroes) {
            Dcs_dropzeros.cs_dropzeros(dcs); //remove zeroes
        }
        if (removeDuplicates) {
            if (!Dcs_dupl.cs_dupl(dcs)) { //remove duplicates
                throw new IllegalArgumentException("Exception occured in cs_dupl()!");
            }
        }
        //sort row indexes
        if (sortRowIndexes) {
            dcs = Dcs_transpose.cs_transpose(dcs, true);
            dcs = Dcs_transpose.cs_transpose(dcs, true);
            if (dcs == null) {
                throw new IllegalArgumentException("Exception occured in cs_transpose()!");
            }
            rowIndexesSorted = true;
        }
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

            final double[] valuesE = dcs.x;
            int nz = cardinality();
            for (int j = 0; j < nz; j++) {
                valuesE[j] *= alpha;
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
            Arrays.fill(dcs.i, 0);
            Arrays.fill(dcs.p, 0);
            Arrays.fill(dcs.x, 0);
        } else {
            int nnz = cardinality();
            for (int i = 0; i < nnz; i++) {
                dcs.x[i] = value;
            }
        }
        return this;
    }

    public DoubleMatrix2D assign(DoubleMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof SparseCCDoubleMatrix2D) {
            SparseCCDoubleMatrix2D other = (SparseCCDoubleMatrix2D) source;
            System.arraycopy(other.getColumnPointers(), 0, this.dcs.p, 0, columns + 1);
            int nzmax = other.getRowIndexes().length;
            if (dcs.nzmax < nzmax) {
                dcs.i = new int[nzmax];
                dcs.x = new double[nzmax];
            }
            System.arraycopy(other.getRowIndexes(), 0, this.dcs.i, 0, nzmax);
            System.arraycopy(other.getValues(), 0, this.dcs.x, 0, nzmax);
            rowIndexesSorted = other.rowIndexesSorted;
        } else if (source instanceof SparseRCDoubleMatrix2D) {
            SparseRCDoubleMatrix2D other = ((SparseRCDoubleMatrix2D) source).getTranspose();
            this.dcs.p = other.getRowPointers();
            this.dcs.i = other.getColumnIndexes();
            this.dcs.x = other.getValues();
            this.dcs.nzmax = this.dcs.x.length;
            rowIndexesSorted = true;
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

        if ((y instanceof SparseCCDoubleMatrix2D) && (function == cern.jet.math.tdouble.DoubleFunctions.plus)) { // x[i] = x[i] + y[i] 
            SparseCCDoubleMatrix2D yy = (SparseCCDoubleMatrix2D) y;
            dcs = Dcs_add.cs_add(dcs, yy.dcs, 1, 1);
            return this;
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
            final int[] rowIndexesA = dcs.i;
            final int[] columnPointersA = dcs.p;
            final double[] valuesA = dcs.x;
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

        if (function == cern.jet.math.tdouble.DoubleFunctions.div) { // x[i] = x[i] / y[i]
            final int[] rowIndexesA = dcs.i;
            final int[] columnPointersA = dcs.p;
            final double[] valuesA = dcs.x;

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
        return dcs.p[columns];
    }

    public Dcs elements() {
        return dcs;
    }

    public DoubleMatrix2D forEachNonZero(final cern.colt.function.tdouble.IntIntDoubleFunction function) {
        final int[] rowIndexesA = dcs.i;
        final int[] columnPointersA = dcs.p;
        final double[] valuesA = dcs.x;

        for (int j = columns; --j >= 0;) {
            int low = columnPointersA[j];
            for (int k = columnPointersA[j + 1]; --k >= low;) {
                int i = rowIndexesA[k];
                double value = valuesA[k];
                double r = function.apply(i, j, value);
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
        return dcs.p;
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
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        double v = 0;
        if (k >= 0)
            v = dcs.x[k];
        return v;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a row-compressed form. This method creates a new object (not a view), so
     * changes in the returned matrix are NOT reflected in this matrix.
     * 
     * @return this matrix in a row-compressed form
     */
    public SparseRCDoubleMatrix2D getRowCompressed() {
        Dcs dcst = Dcs_transpose.cs_transpose(dcs, true);
        SparseRCDoubleMatrix2D rc = new SparseRCDoubleMatrix2D(rows, columns);
        rc.columnIndexes = dcst.i;
        rc.rowPointers = dcst.p;
        rc.values = dcst.x;
        rc.columnIndexesSorted = true;
        return rc;
    }

    /**
     * Returns row indexes;
     * 
     * @return row indexes
     */
    public int[] getRowIndexes() {
        return dcs.i;
    }

    /**
     * Returns a new matrix that is the transpose of this matrix. This method
     * creates a new object (not a view), so changes in the returned matrix are
     * NOT reflected in this matrix.
     * 
     * @return the transpose of this matrix
     */
    public SparseCCDoubleMatrix2D getTranspose() {
        Dcs dcst = Dcs_transpose.cs_transpose(dcs, true);
        SparseCCDoubleMatrix2D tr = new SparseCCDoubleMatrix2D(columns, rows);
        tr.dcs = dcst;
        return tr;
    }

    /**
     * Returns numerical values
     * 
     * @return numerical values
     */
    public double[] getValues() {
        return dcs.x;
    }

    /**
     * Returns true if row indexes are sorted, false otherwise
     * 
     * @return true if row indexes are sorted, false otherwise
     */
    public boolean hasRowIndexesSorted() {
        return rowIndexesSorted;
    }

    public DoubleMatrix2D like(int rows, int columns) {
        return new SparseCCDoubleMatrix2D(rows, columns);
    }

    public DoubleMatrix1D like1D(int size) {
        return new SparseDoubleMatrix1D(size);
    }

    public synchronized void setQuick(int row, int column, double value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);

        if (k >= 0) { // found
            if (value == 0)
                remove(column, k);
            else
                dcs.x[k] = value;
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
        dcs = Dcs_transpose.cs_transpose(dcs, true);
        dcs = Dcs_transpose.cs_transpose(dcs, true);
        if (dcs == null) {
            throw new IllegalArgumentException("Exception occured in cs_transpose()!");
        }
        rowIndexesSorted = true;
    }

    /**
     * Removes (sums) duplicate entries (if any}
     */
    public void removeDuplicates() {
        if (!Dcs_dupl.cs_dupl(dcs)) { //remove duplicates
            throw new IllegalArgumentException("Exception occured in cs_dupl()!");
        }
    }

    /**
     * Removes zero entries (if any)
     */
    public void removeZeroes() {
        Dcs_dropzeros.cs_dropzeros(dcs); //remove zeroes
    }

    public void trimToSize() {
        Dcs_util.cs_sprealloc(dcs, 0);
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(rows).append(" x ").append(columns).append(" sparse matrix, nnz = ").append(cardinality())
                .append('\n');
        for (int i = 0; i < columns; i++) {
            int high = dcs.p[i + 1];
            for (int j = dcs.p[i]; j < high; j++) {
                builder.append('(').append(dcs.i[j]).append(',').append(i).append(')').append('\t').append(dcs.x[j])
                        .append('\n');
            }
        }
        return builder.toString();
    }

    public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, final double alpha, final double beta,
            final boolean transposeA) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;

        boolean ignore = (z == null || transposeA);
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
        final int zeroZ = (int) zz.index(0);

        DenseDoubleMatrix1D yy = (DenseDoubleMatrix1D) y;
        final double[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) yy.index(0);

        final int[] rowIndexesA = dcs.i;
        final int[] columnPointersA = dcs.p;
        final double[] valuesA = dcs.x;

        int zidx = zeroZ;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (!transposeA) {
            if ((!ignore) && (beta / alpha != 1.0)) {
                z.assign(cern.jet.math.tdouble.DoubleFunctions.mult(beta / alpha));
            }

            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = 2;
                Future<?>[] futures = new Future[nthreads];
                final double[] result = new double[rowsA];
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
                                    double yElem = elementsY[zeroY + strideY * i];
                                    for (int k = columnPointersA[i]; k < high; k++) {
                                        int j = rowIndexesA[k];
                                        elementsZ[zeroZ + strideZ * j] += valuesA[k] * yElem;
                                    }
                                }
                            } else {
                                for (int i = firstColumn; i < lastColumn; i++) {
                                    int high = columnPointersA[i + 1];
                                    double yElem = elementsY[zeroY + strideY * i];
                                    for (int k = columnPointersA[i]; k < high; k++) {
                                        int j = rowIndexesA[k];
                                        result[j] += valuesA[k] * yElem;
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
                    double yElem = elementsY[zeroY + strideY * i];
                    for (int k = columnPointersA[i]; k < high; k++) {
                        int j = rowIndexesA[k];
                        elementsZ[zeroZ + strideZ * j] += valuesA[k] * yElem;
                    }
                }
            }
            if (alpha != 1.0) {
                z.assign(cern.jet.math.tdouble.DoubleFunctions.mult(alpha));
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
                            int k = dcs.p[firstColumn];
                            for (int i = firstColumn; i < lastColumn; i++) {
                                double sum = 0;
                                int high = dcs.p[i + 1];
                                for (; k + 10 < high; k += 10) {
                                    int ind = k + 9;
                                    sum += valuesA[ind] * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * dcs.i[ind--]];
                                }
                                for (; k < high; k++) {
                                    sum += valuesA[k] * elementsY[dcs.i[k]];
                                }
                                elementsZ[zidx] = alpha * sum + beta * elementsZ[zidx];
                                zidx += strideZ;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int k = dcs.p[0];
                for (int i = 0; i < columns; i++) {
                    double sum = 0;
                    int high = dcs.p[i + 1];
                    for (; k + 10 < high; k += 10) {
                        int ind = k + 9;
                        sum += valuesA[ind] * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * dcs.i[ind--]];
                    }
                    for (; k < high; k++) {
                        sum += valuesA[k] * elementsY[dcs.i[k]];
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
            if (B instanceof SparseCCDoubleMatrix2D) {
                C = new SparseCCDoubleMatrix2D(rowsA, p, (rowsA * p));
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
            SparseCCDoubleMatrix2D AA;
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
            int[] columnPointersA = AA.dcs.p;
            int[] rowIndexesA = AA.dcs.i;
            double[] valuesA = AA.dcs.x;

            int zeroB = (int) BB.index(0, 0);
            int rowStrideB = BB.rowStride();
            int columnStrideB = BB.columnStride();
            double[] elementsB = BB.elements;

            int zeroC = (int) CC.index(0, 0);
            int rowStrideC = CC.rowStride();
            int columnStrideC = CC.columnStride();
            double[] elementsC = CC.elements;

            for (int jj = 0; jj < columnsB; jj++) {
                for (int kk = 0; kk < columnsA; kk++) {
                    int high = columnPointersA[kk + 1];
                    double yElem = elementsB[zeroB + kk * rowStrideB + jj * columnStrideB];
                    for (int ii = columnPointersA[kk]; ii < high; ii++) {
                        int j = rowIndexesA[ii];
                        elementsC[zeroC + j * rowStrideC + jj * columnStrideC] += valuesA[ii] * yElem;
                    }
                }
            }
            if (alpha != 1.0) {
                C.assign(cern.jet.math.tdouble.DoubleFunctions.mult(alpha));
            }

        } else if ((B instanceof SparseCCDoubleMatrix2D) && (C instanceof SparseCCDoubleMatrix2D)) {
            SparseCCDoubleMatrix2D AA;
            if (transposeA) {
                AA = getTranspose();
            } else {
                AA = this;
            }
            SparseCCDoubleMatrix2D BB = (SparseCCDoubleMatrix2D) B;
            if (transposeB) {
                BB = BB.getTranspose();
            }
            SparseCCDoubleMatrix2D CC = (SparseCCDoubleMatrix2D) C;
            CC.dcs = Dcs_multiply.cs_multiply(AA.dcs, BB.dcs);
            if (CC.dcs == null) {
                throw new IllegalArgumentException("Exception occured in cs_multiply()");
            }
            if (alpha != 1.0) {
                CC.assign(cern.jet.math.tdouble.DoubleFunctions.mult(alpha));
            }
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

            final int[] rowIndexesA = dcs.i;
            final int[] columnPointersA = dcs.p;
            final double[] valuesA = dcs.x;
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

    protected DoubleMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, double value) {
        IntArrayList rowIndexes = new IntArrayList(dcs.i);
        rowIndexes.setSizeRaw(dcs.p[columns]);
        DoubleArrayList values = new DoubleArrayList(dcs.x);
        values.setSizeRaw(dcs.p[columns]);
        rowIndexes.beforeInsert(index, row);
        values.beforeInsert(index, value);
        for (int i = dcs.p.length; --i > column;)
            dcs.p[i]++;
        dcs.i = rowIndexes.elements();
        dcs.x = values.elements();
        dcs.nzmax = rowIndexes.elements().length;
    }

    protected void remove(int column, int index) {
        IntArrayList rowIndexes = new IntArrayList(dcs.i);
        DoubleArrayList values = new DoubleArrayList(dcs.x);
        rowIndexes.remove(index);
        values.remove(index);
        for (int i = dcs.p.length; --i > column;)
            dcs.p[i]--;
        dcs.i = rowIndexes.elements();
        dcs.x = values.elements();
        dcs.nzmax = rowIndexes.elements().length;
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
}
