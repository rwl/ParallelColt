/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import java.util.Arrays;
import java.util.concurrent.Future;

import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import edu.emory.mathcs.csparsej.tfloat.Scs_add;
import edu.emory.mathcs.csparsej.tfloat.Scs_cumsum;
import edu.emory.mathcs.csparsej.tfloat.Scs_dropzeros;
import edu.emory.mathcs.csparsej.tfloat.Scs_dupl;
import edu.emory.mathcs.csparsej.tfloat.Scs_multiply;
import edu.emory.mathcs.csparsej.tfloat.Scs_transpose;
import edu.emory.mathcs.csparsej.tfloat.Scs_util;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scs;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Sparse column-compressed 2-d matrix holding <tt>float</tt> elements. First
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
 * A.forEachNonZero(new cern.colt.function.IntIntFloatFunction() {
 *     public float apply(int row, int column, float value) {
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
 * B.forEachNonZero(new cern.colt.function.IntIntFloatFunction() {
 *     public float apply(int row, int column, float value) {
 *         A.setQuick(row, column, A.getQuick(row, column) + alpha * value);
 *         return value;
 *     }
 * });
 * </pre>
 * 
 * </td>
 * </table>
 * Method
 * {@link #assign(FloatMatrix2D,cern.colt.function.tfloat.FloatFloatFunction)}
 * does just that if you supply
 * {@link cern.jet.math.tfloat.FloatFunctions#plusMultSecond} as argument.
 * 
 * 
 * @author Piotr Wendykier
 * 
 */
public class SparseCCFloatMatrix2D extends WrapperFloatMatrix2D {
    private static final long serialVersionUID = 1L;
    /*
     * Internal storage.
     */
    protected Scs scs;

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
    public SparseCCFloatMatrix2D(float[][] values) {
        this(values.length, values[0].length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given internal storage.
     * 
     * @param dcs
     *            internal storage.
     */
    public SparseCCFloatMatrix2D(Scs dcs) {
        super(null);
        try {
            setUp(dcs.m, dcs.n);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        this.scs = dcs;
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
    public SparseCCFloatMatrix2D(int rows, int columns) {
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
    public SparseCCFloatMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        scs = Scs_util.cs_spalloc(rows, columns, nzmax, true, false);
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
    public SparseCCFloatMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, float value,
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
        scs = Scs_util.cs_spalloc(rows, columns, nz, true, false);
        int[] w = new int[columns];
        int[] Cp = scs.p;
        int[] Ci = scs.i;
        float[] Cx = scs.x;
        for (int k = 0; k < nz; k++)
            w[columnIndexes[k]]++;
        Scs_cumsum.cs_cumsum(Cp, w, columns);
        int p;
        for (int k = 0; k < nz; k++) {
            Ci[p = w[columnIndexes[k]]++] = rowIndexes[k];
            if (Cx != null)
                Cx[p] = value;
        }
        if (removeDuplicates) {
            if (!Scs_dupl.cs_dupl(scs)) { //remove duplicates
                throw new IllegalArgumentException("Exception occured in cs_dupl()!");
            }
        }
        if (sortRowIndexes) {
            //sort row indexes
            scs = Scs_transpose.cs_transpose(scs, true);
            scs = Scs_transpose.cs_transpose(scs, true);
            if (scs == null) {
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
    public SparseCCFloatMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, float[] values,
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
        scs = Scs_util.cs_spalloc(rows, columns, nz, true, false);
        int[] w = new int[columns];
        int[] Cp = scs.p;
        int[] Ci = scs.i;
        float[] Cx = scs.x;
        for (int k = 0; k < nz; k++)
            w[columnIndexes[k]]++;
        Scs_cumsum.cs_cumsum(Cp, w, columns);
        int p;
        for (int k = 0; k < nz; k++) {
            Ci[p = w[columnIndexes[k]]++] = rowIndexes[k];
            if (Cx != null)
                Cx[p] = values[k];
        }
        if (removeZeroes) {
            Scs_dropzeros.cs_dropzeros(scs); //remove zeroes
        }
        if (removeDuplicates) {
            if (!Scs_dupl.cs_dupl(scs)) { //remove duplicates
                throw new IllegalArgumentException("Exception occured in cs_dupl()!");
            }
        }
        //sort row indexes
        if (sortRowIndexes) {
            scs = Scs_transpose.cs_transpose(scs, true);
            scs = Scs_transpose.cs_transpose(scs, true);
            if (scs == null) {
                throw new IllegalArgumentException("Exception occured in cs_transpose()!");
            }
            rowIndexesSorted = true;
        }
    }

    public FloatMatrix2D assign(final cern.colt.function.tfloat.FloatFunction function) {
        if (function instanceof cern.jet.math.tfloat.FloatMult) { // x[i] = mult*x[i]
            final float alpha = ((cern.jet.math.tfloat.FloatMult) function).multiplicator;
            if (alpha == 1)
                return this;
            if (alpha == 0)
                return assign(0);
            if (alpha != alpha)
                return assign(alpha); // the funny definition of isNaN(). This should better not happen.

            final float[] valuesE = scs.x;
            int nz = cardinality();
            for (int j = 0; j < nz; j++) {
                valuesE[j] *= alpha;
            }
        } else {
            forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    return function.apply(value);
                }
            });
        }
        return this;
    }

    public FloatMatrix2D assign(float value) {
        if (value == 0) {
            Arrays.fill(scs.i, 0);
            Arrays.fill(scs.p, 0);
            Arrays.fill(scs.x, 0);
        } else {
            int nnz = cardinality();
            for (int i = 0; i < nnz; i++) {
                scs.x[i] = value;
            }
        }
        return this;
    }

    public FloatMatrix2D assign(FloatMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof SparseCCFloatMatrix2D) {
            SparseCCFloatMatrix2D other = (SparseCCFloatMatrix2D) source;
            System.arraycopy(other.getColumnPointers(), 0, this.scs.p, 0, columns + 1);
            int nzmax = other.getRowIndexes().length;
            if (scs.nzmax < nzmax) {
                scs.i = new int[nzmax];
                scs.x = new float[nzmax];
            }
            System.arraycopy(other.getRowIndexes(), 0, this.scs.i, 0, nzmax);
            System.arraycopy(other.getValues(), 0, this.scs.x, 0, nzmax);
            rowIndexesSorted = other.rowIndexesSorted;
        } else if (source instanceof SparseRCFloatMatrix2D) {
            SparseRCFloatMatrix2D other = ((SparseRCFloatMatrix2D) source).getTranspose();
            this.scs.p = other.getRowPointers();
            this.scs.i = other.getColumnIndexes();
            this.scs.x = other.getValues();
            this.scs.nzmax = this.scs.x.length;
            rowIndexesSorted = true;
        } else {
            assign(0);
            source.forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
        }
        return this;
    }

    public FloatMatrix2D assign(final FloatMatrix2D y, cern.colt.function.tfloat.FloatFloatFunction function) {
        checkShape(y);

        if ((y instanceof SparseCCFloatMatrix2D) && (function == cern.jet.math.tfloat.FloatFunctions.plus)) { // x[i] = x[i] + y[i] 
            SparseCCFloatMatrix2D yy = (SparseCCFloatMatrix2D) y;
            scs = Scs_add.cs_add(scs, yy.scs, 1, 1);
            return this;
        }

        if (function instanceof cern.jet.math.tfloat.FloatPlusMultSecond) { // x[i] = x[i] + alpha*y[i]
            final float alpha = ((cern.jet.math.tfloat.FloatPlusMultSecond) function).multiplicator;
            if (alpha == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    setQuick(i, j, getQuick(i, j) + alpha * value);
                    return value;
                }
            });
            return this;
        }

        if (function instanceof cern.jet.math.tfloat.FloatPlusMultFirst) { // x[i] = alpha*x[i] + y[i]
            final float alpha = ((cern.jet.math.tfloat.FloatPlusMultFirst) function).multiplicator;
            if (alpha == 0)
                return assign(y);
            y.forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    setQuick(i, j, alpha * getQuick(i, j) + value);
                    return value;
                }
            });
            return this;
        }

        if (function == cern.jet.math.tfloat.FloatFunctions.mult) { // x[i] = x[i] * y[i]
            final int[] rowIndexesA = scs.i;
            final int[] columnPointersA = scs.p;
            final float[] valuesA = scs.x;
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

        if (function == cern.jet.math.tfloat.FloatFunctions.div) { // x[i] = x[i] / y[i]
            final int[] rowIndexesA = scs.i;
            final int[] columnPointersA = scs.p;
            final float[] valuesA = scs.x;

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
        return scs.p[columns];
    }

    public Scs elements() {
        return scs;
    }

    public FloatMatrix2D forEachNonZero(final cern.colt.function.tfloat.IntIntFloatFunction function) {
        final int[] rowIndexesA = scs.i;
        final int[] columnPointersA = scs.p;
        final float[] valuesA = scs.x;

        for (int j = columns; --j >= 0;) {
            int low = columnPointersA[j];
            for (int k = columnPointersA[j + 1]; --k >= low;) {
                int i = rowIndexesA[k];
                float value = valuesA[k];
                float r = function.apply(i, j, value);
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
        return scs.p;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a dense form. This method creates a new object (not a view), so changes
     * in the returned matrix are NOT reflected in this matrix.
     * 
     * @return this matrix in a dense form
     */
    public DenseFloatMatrix2D getDense() {
        final DenseFloatMatrix2D dense = new DenseFloatMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tfloat.IntIntFloatFunction() {
            public float apply(int i, int j, float value) {
                dense.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return dense;
    }

    public synchronized float getQuick(int row, int column) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(scs.i, row, scs.p[column], scs.p[column + 1] - 1);
        float v = 0;
        if (k >= 0)
            v = scs.x[k];
        return v;
    }

    /**
     * Returns a new matrix that has the same elements as this matrix, but is in
     * a row-compressed form. This method creates a new object (not a view), so
     * changes in the returned matrix are NOT reflected in this matrix.
     * 
     * @return this matrix in a row-compressed form
     */
    public SparseRCFloatMatrix2D getRowCompressed() {
        Scs dcst = Scs_transpose.cs_transpose(scs, true);
        SparseRCFloatMatrix2D rc = new SparseRCFloatMatrix2D(rows, columns);
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
        return scs.i;
    }

    /**
     * Returns a new matrix that is the transpose of this matrix. This method
     * creates a new object (not a view), so changes in the returned matrix are
     * NOT reflected in this matrix.
     * 
     * @return the transpose of this matrix
     */
    public SparseCCFloatMatrix2D getTranspose() {
        Scs dcst = Scs_transpose.cs_transpose(scs, true);
        SparseCCFloatMatrix2D tr = new SparseCCFloatMatrix2D(columns, rows);
        tr.scs = dcst;
        return tr;
    }

    /**
     * Returns numerical values
     * 
     * @return numerical values
     */
    public float[] getValues() {
        return scs.x;
    }

    /**
     * Returns true if row indexes are sorted, false otherwise
     * 
     * @return true if row indexes are sorted, false otherwise
     */
    public boolean hasRowIndexesSorted() {
        return rowIndexesSorted;
    }

    public FloatMatrix2D like(int rows, int columns) {
        return new SparseCCFloatMatrix2D(rows, columns);
    }

    public FloatMatrix1D like1D(int size) {
        return new SparseFloatMatrix1D(size);
    }

    public synchronized void setQuick(int row, int column, float value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(scs.i, row, scs.p[column], scs.p[column + 1] - 1);

        if (k >= 0) { // found
            if (value == 0)
                remove(column, k);
            else
                scs.x[k] = value;
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
        scs = Scs_transpose.cs_transpose(scs, true);
        scs = Scs_transpose.cs_transpose(scs, true);
        if (scs == null) {
            throw new IllegalArgumentException("Exception occured in cs_transpose()!");
        }
        rowIndexesSorted = true;
    }

    /**
     * Removes (sums) duplicate entries (if any}
     */
    public void removeDuplicates() {
        if (!Scs_dupl.cs_dupl(scs)) { //remove duplicates
            throw new IllegalArgumentException("Exception occured in cs_dupl()!");
        }
    }

    /**
     * Removes zero entries (if any)
     */
    public void removeZeroes() {
        Scs_dropzeros.cs_dropzeros(scs); //remove zeroes
    }

    public void trimToSize() {
        Scs_util.cs_sprealloc(scs, 0);
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(rows).append(" x ").append(columns).append(" sparse matrix, nnz = ").append(cardinality())
                .append('\n');
        for (int i = 0; i < columns; i++) {
            int high = scs.p[i + 1];
            for (int j = scs.p[i]; j < high; j++) {
                builder.append('(').append(scs.i[j]).append(',').append(i).append(')').append('\t').append(scs.x[j])
                        .append('\n');
            }
        }
        return builder.toString();
    }

    public FloatMatrix1D zMult(FloatMatrix1D y, FloatMatrix1D z, final float alpha, final float beta,
            final boolean transposeA) {
        final int rowsA = transposeA ? columns : rows;
        final int columnsA = transposeA ? rows : columns;

        boolean ignore = (z == null || transposeA);
        if (z == null)
            z = new DenseFloatMatrix1D(rowsA);

        if (!(y instanceof DenseFloatMatrix1D && z instanceof DenseFloatMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (columnsA != y.size() || rowsA > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        DenseFloatMatrix1D zz = (DenseFloatMatrix1D) z;
        final float[] elementsZ = zz.elements;
        final int strideZ = zz.stride();
        final int zeroZ = (int) zz.index(0);

        DenseFloatMatrix1D yy = (DenseFloatMatrix1D) y;
        final float[] elementsY = yy.elements;
        final int strideY = yy.stride();
        final int zeroY = (int) yy.index(0);

        final int[] rowIndexesA = scs.i;
        final int[] columnPointersA = scs.p;
        final float[] valuesA = scs.x;

        int zidx = zeroZ;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (!transposeA) {
            if ((!ignore) && (beta / alpha != 1.0)) {
                z.assign(cern.jet.math.tfloat.FloatFunctions.mult(beta / alpha));
            }

            if ((nthreads > 1) && (cardinality() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = 2;
                Future<?>[] futures = new Future[nthreads];
                final float[] result = new float[rowsA];
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
                                    float yElem = elementsY[zeroY + strideY * i];
                                    for (int k = columnPointersA[i]; k < high; k++) {
                                        int j = rowIndexesA[k];
                                        elementsZ[zeroZ + strideZ * j] += valuesA[k] * yElem;
                                    }
                                }
                            } else {
                                for (int i = firstColumn; i < lastColumn; i++) {
                                    int high = columnPointersA[i + 1];
                                    float yElem = elementsY[zeroY + strideY * i];
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
                    float yElem = elementsY[zeroY + strideY * i];
                    for (int k = columnPointersA[i]; k < high; k++) {
                        int j = rowIndexesA[k];
                        elementsZ[zeroZ + strideZ * j] += valuesA[k] * yElem;
                    }
                }
            }
            if (alpha != 1.0) {
                z.assign(cern.jet.math.tfloat.FloatFunctions.mult(alpha));
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
                            int k = scs.p[firstColumn];
                            for (int i = firstColumn; i < lastColumn; i++) {
                                float sum = 0;
                                int high = scs.p[i + 1];
                                for (; k + 10 < high; k += 10) {
                                    int ind = k + 9;
                                    sum += valuesA[ind] * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                            * elementsY[zeroY + strideY * scs.i[ind--]];
                                }
                                for (; k < high; k++) {
                                    sum += valuesA[k] * elementsY[scs.i[k]];
                                }
                                elementsZ[zidx] = alpha * sum + beta * elementsZ[zidx];
                                zidx += strideZ;
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                int k = scs.p[0];
                for (int i = 0; i < columns; i++) {
                    float sum = 0;
                    int high = scs.p[i + 1];
                    for (; k + 10 < high; k += 10) {
                        int ind = k + 9;
                        sum += valuesA[ind] * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]] + valuesA[ind]
                                * elementsY[zeroY + strideY * scs.i[ind--]];
                    }
                    for (; k < high; k++) {
                        sum += valuesA[k] * elementsY[scs.i[k]];
                    }
                    elementsZ[zidx] = alpha * sum + beta * elementsZ[zidx];
                    zidx += strideZ;
                }
            }
        }
        return z;
    }

    public FloatMatrix2D zMult(FloatMatrix2D B, FloatMatrix2D C, final float alpha, float beta,
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
            if (B instanceof SparseCCFloatMatrix2D) {
                C = new SparseCCFloatMatrix2D(rowsA, p, (rowsA * p));
            } else {
                C = new DenseFloatMatrix2D(rowsA, p);
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
            C.assign(cern.jet.math.tfloat.FloatFunctions.mult(beta));
        }

        if ((B instanceof DenseFloatMatrix2D) && (C instanceof DenseFloatMatrix2D)) {
            SparseCCFloatMatrix2D AA;
            if (transposeA) {
                AA = getTranspose();
            } else {
                AA = this;
            }
            DenseFloatMatrix2D BB;
            if (transposeB) {
                BB = (DenseFloatMatrix2D) B.viewDice();
            } else {
                BB = (DenseFloatMatrix2D) B;
            }
            DenseFloatMatrix2D CC = (DenseFloatMatrix2D) C;
            int[] columnPointersA = AA.scs.p;
            int[] rowIndexesA = AA.scs.i;
            float[] valuesA = AA.scs.x;

            int zeroB = (int) BB.index(0, 0);
            int rowStrideB = BB.rowStride();
            int columnStrideB = BB.columnStride();
            float[] elementsB = BB.elements;

            int zeroC = (int) CC.index(0, 0);
            int rowStrideC = CC.rowStride();
            int columnStrideC = CC.columnStride();
            float[] elementsC = CC.elements;

            for (int jj = 0; jj < columnsB; jj++) {
                for (int kk = 0; kk < columnsA; kk++) {
                    int high = columnPointersA[kk + 1];
                    float yElem = elementsB[zeroB + kk * rowStrideB + jj * columnStrideB];
                    for (int ii = columnPointersA[kk]; ii < high; ii++) {
                        int j = rowIndexesA[ii];
                        elementsC[zeroC + j * rowStrideC + jj * columnStrideC] += valuesA[ii] * yElem;
                    }
                }
            }
            if (alpha != 1.0) {
                C.assign(cern.jet.math.tfloat.FloatFunctions.mult(alpha));
            }

        } else if ((B instanceof SparseCCFloatMatrix2D) && (C instanceof SparseCCFloatMatrix2D)) {
            SparseCCFloatMatrix2D AA;
            if (transposeA) {
                AA = getTranspose();
            } else {
                AA = this;
            }
            SparseCCFloatMatrix2D BB = (SparseCCFloatMatrix2D) B;
            if (transposeB) {
                BB = BB.getTranspose();
            }
            SparseCCFloatMatrix2D CC = (SparseCCFloatMatrix2D) C;
            CC.scs = Scs_multiply.cs_multiply(AA.scs, BB.scs);
            if (CC.scs == null) {
                throw new IllegalArgumentException("Exception occured in cs_multiply()");
            }
            if (alpha != 1.0) {
                CC.assign(cern.jet.math.tfloat.FloatFunctions.mult(alpha));
            }
        } else {
            if (transposeB) {
                B = B.viewDice();
            }
            // cache views
            final FloatMatrix1D[] Brows = new FloatMatrix1D[columnsA];
            for (int i = columnsA; --i >= 0;)
                Brows[i] = B.viewRow(i);
            final FloatMatrix1D[] Crows = new FloatMatrix1D[rowsA];
            for (int i = rowsA; --i >= 0;)
                Crows[i] = C.viewRow(i);

            final cern.jet.math.tfloat.FloatPlusMultSecond fun = cern.jet.math.tfloat.FloatPlusMultSecond.plusMult(0);

            final int[] rowIndexesA = scs.i;
            final int[] columnPointersA = scs.p;
            final float[] valuesA = scs.x;
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

    protected FloatMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, float value) {
        IntArrayList rowIndexes = new IntArrayList(scs.i);
        rowIndexes.setSizeRaw(scs.p[columns]);
        FloatArrayList values = new FloatArrayList(scs.x);
        values.setSizeRaw(scs.p[columns]);
        rowIndexes.beforeInsert(index, row);
        values.beforeInsert(index, value);
        for (int i = scs.p.length; --i > column;)
            scs.p[i]++;
        scs.i = rowIndexes.elements();
        scs.x = values.elements();
        scs.nzmax = rowIndexes.elements().length;
    }

    protected void remove(int column, int index) {
        IntArrayList rowIndexes = new IntArrayList(scs.i);
        FloatArrayList values = new FloatArrayList(scs.x);
        rowIndexes.remove(index);
        values.remove(index);
        for (int i = scs.p.length; --i > column;)
            scs.p[i]--;
        scs.i = rowIndexes.elements();
        scs.x = values.elements();
        scs.nzmax = rowIndexes.elements().length;
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
