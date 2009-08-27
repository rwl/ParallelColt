/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import java.util.Arrays;

import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tobject.ObjectArrayList;
import cern.colt.matrix.tobject.ObjectMatrix1D;
import cern.colt.matrix.tobject.ObjectMatrix2D;

/**
 * Sparse column-compressed 2-d matrix holding <tt>Object</tt> elements. First
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
 * 
 * @author Piotr Wendykier
 * 
 */
public class SparseCCObjectMatrix2D extends WrapperObjectMatrix2D {
    private static final long serialVersionUID = 1L;
    /*
     * Internal storage.
     */
    protected int[] columnPointers;

    protected int[] rowIndexes;

    protected Object[] values;

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
    public SparseCCObjectMatrix2D(Object[][] values) {
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
    public SparseCCObjectMatrix2D(int rows, int columns) {
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
    public SparseCCObjectMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        rowIndexes = new int[nzmax];
        values = new Object[nzmax];
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
     * @param sortRowIndexes
     *            if true, then row indexes are sorted
     */
    public SparseCCObjectMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, Object value,
            boolean sortRowIndexes) {
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
        if (value == null) {
            throw new IllegalArgumentException("value cannot be null");
        }

        int nz = Math.max(rowIndexes.length, 1);
        this.rowIndexes = new int[nz];
        this.values = new Object[nz];
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
     * @param removeZeroes
     *            if true, then zeroes (if any) are removed
     * @param sortRowIndexes
     *            if true, then row indexes are sorted
     */
    public SparseCCObjectMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, Object[] values,
            boolean removeZeroes, boolean sortRowIndexes) {
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
        this.values = new Object[nz];
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
        if (sortRowIndexes) {
            sortRowIndexes();
        }
    }

    public ObjectMatrix2D assign(final cern.colt.function.tobject.ObjectFunction function) {
        forEachNonZero(new cern.colt.function.tobject.IntIntObjectFunction() {
            public Object apply(int i, int j, Object value) {
                return function.apply(value);
            }
        });
        return this;
    }

    public ObjectMatrix2D assign(Object value) {
        if (value == null) {
            Arrays.fill(rowIndexes, 0);
            Arrays.fill(columnPointers, 0);
            Arrays.fill(values, null);
        } else {
            int nnz = cardinality();
            for (int i = 0; i < nnz; i++) {
                values[i] = value;
            }
        }
        return this;
    }

    public ObjectMatrix2D assign(ObjectMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof SparseCCObjectMatrix2D) {
            SparseCCObjectMatrix2D other = (SparseCCObjectMatrix2D) source;
            System.arraycopy(other.getColumnPointers(), 0, columnPointers, 0, columns + 1);
            int nzmax = other.getRowIndexes().length;
            if (rowIndexes.length < nzmax) {
                rowIndexes = new int[nzmax];
                values = new Object[nzmax];
            }
            System.arraycopy(other.getRowIndexes(), 0, rowIndexes, 0, nzmax);
            System.arraycopy(other.getValues(), 0, values, 0, nzmax);
            rowIndexesSorted = other.rowIndexesSorted;
        } else if (source instanceof SparseRCObjectMatrix2D) {
            SparseRCObjectMatrix2D other = ((SparseRCObjectMatrix2D) source).getTranspose();
            columnPointers = other.getRowPointers();
            rowIndexes = other.getColumnIndexes();
            values = other.getValues();
            rowIndexesSorted = true;
        } else {
            assign((Object) null);
            source.forEachNonZero(new cern.colt.function.tobject.IntIntObjectFunction() {
                public Object apply(int i, int j, Object value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
        }
        return this;
    }

    public int cardinality() {
        return columnPointers[columns];
    }

    public ObjectMatrix2D forEachNonZero(final cern.colt.function.tobject.IntIntObjectFunction function) {
        final int[] rowIndexesA = rowIndexes;
        final int[] columnPointersA = columnPointers;
        final Object[] valuesA = values;

        for (int j = columns; --j >= 0;) {
            int low = columnPointersA[j];
            for (int k = columnPointersA[j + 1]; --k >= low;) {
                int i = rowIndexesA[k];
                Object value = valuesA[k];
                Object r = function.apply(i, j, value);
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
    public DenseObjectMatrix2D getDense() {
        final DenseObjectMatrix2D dense = new DenseObjectMatrix2D(rows, columns);
        forEachNonZero(new cern.colt.function.tobject.IntIntObjectFunction() {
            public Object apply(int i, int j, Object value) {
                dense.setQuick(i, j, getQuick(i, j));
                return value;
            }
        });
        return dense;
    }

    public synchronized Object getQuick(int row, int column) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(rowIndexes, row, columnPointers[column], columnPointers[column + 1] - 1);
        Object v = 0;
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
    public SparseRCObjectMatrix2D getRowCompressed() {
        SparseCCObjectMatrix2D tr = getTranspose();
        SparseRCObjectMatrix2D rc = new SparseRCObjectMatrix2D(rows, columns);
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
    public SparseCCObjectMatrix2D getTranspose() {
        int p, q, j, Cp[], Ci[], n, m, Ap[], Ai[], w[];
        Object Cx[], Ax[];
        m = rows;
        n = columns;
        Ap = columnPointers;
        Ai = rowIndexes;
        Ax = values;
        SparseCCObjectMatrix2D C = new SparseCCObjectMatrix2D(columns, rows, Ai.length); /* allocate result */
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
    public Object[] getValues() {
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

    public ObjectMatrix2D like(int rows, int columns) {
        return new SparseCCObjectMatrix2D(rows, columns);
    }

    public ObjectMatrix1D like1D(int size) {
        return new SparseObjectMatrix1D(size);
    }

    public synchronized void setQuick(int row, int column, Object value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
        int k = searchFromTo(rowIndexes, row, columnPointers[column], columnPointers[column + 1] - 1);

        if (k >= 0) { // found
            if (value == null)
                remove(column, k);
            else
                values[k] = value;
            return;
        }

        if (value != null) {
            k = -k - 1;
            insert(row, column, k, value);
        }
    }

    /**
     * Sorts row indexes
     */
    public void sortRowIndexes() {
        SparseCCObjectMatrix2D tr = getTranspose();
        tr = tr.getTranspose();
        columnPointers = tr.columnPointers;
        rowIndexes = tr.rowIndexes;
        values = tr.values;
        rowIndexesSorted = true;
    }

    /**
     * Removes zero entries (if any)
     */
    public void removeZeroes() {
        int j, p, nz = 0, n, Ap[], Ai[];
        Object Ax[];
        n = columns;
        Ap = columnPointers;
        Ai = rowIndexes;
        Ax = values;
        for (j = 0; j < n; j++) {
            p = Ap[j]; /* get current location of col j */
            Ap[j] = nz; /* record new location of col j */
            for (; p < Ap[j + 1]; p++) {
                if (Ax[p] != null) {
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

    protected ObjectMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, Object value) {
        IntArrayList rowIndexesList = new IntArrayList(rowIndexes);
        rowIndexesList.setSizeRaw(columnPointers[columns]);
        ObjectArrayList valuesList = new ObjectArrayList(values);
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
        ObjectArrayList valuesList = new ObjectArrayList(values);
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

    private Object cumsum(int[] p, int[] c, int n) {
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
            nzmax = columnPointers[columns];
        int[] rowIndexesNew = new int[nzmax];
        int length = Math.min(nzmax, rowIndexes.length);
        System.arraycopy(rowIndexes, 0, rowIndexesNew, 0, length);
        rowIndexes = rowIndexesNew;
        Object[] valuesNew = new Object[nzmax];
        length = Math.min(nzmax, values.length);
        System.arraycopy(values, 0, valuesNew, 0, length);
        values = valuesNew;
    }
}
