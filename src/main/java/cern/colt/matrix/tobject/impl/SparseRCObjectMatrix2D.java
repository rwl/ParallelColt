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
 * Sparse row-compressed 2-d matrix holding <tt>Object</tt> elements. First see
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
public class SparseRCObjectMatrix2D extends WrapperObjectMatrix2D {
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

    protected Object[] values;

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
    public SparseRCObjectMatrix2D(Object[][] values) {
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
    public SparseRCObjectMatrix2D(int rows, int columns) {
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
    public SparseRCObjectMatrix2D(int rows, int columns, int nzmax) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        columnIndexes = new int[nzmax];
        values = new Object[nzmax];
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
     * @param sortColumnIndexes
     *            if true, then column indexes are sorted
     */
    public SparseRCObjectMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, Object value,
            boolean sortColumnIndexes) {
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
        this.columnIndexes = new int[nz];
        this.values = new Object[nz];
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
     * @param removeZeroes
     *            if true, then zeroes (if any) are removed
     * @param sortColumnIndexes
     *            if true, then column indexes are sorted
     */
    public SparseRCObjectMatrix2D(int rows, int columns, int[] rowIndexes, int[] columnIndexes, Object[] values,
            boolean removeZeroes, boolean sortColumnIndexes) {
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
        this.values = new Object[nz];
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
    public SparseRCObjectMatrix2D(int rows, int columns, int[] rowPointers, int[] columnIndexes, Object[] values) {
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
            Arrays.fill(rowPointers, 0);
            Arrays.fill(columnIndexes, 0);
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

        if (source instanceof SparseRCObjectMatrix2D) {
            SparseRCObjectMatrix2D other = (SparseRCObjectMatrix2D) source;
            System.arraycopy(other.rowPointers, 0, rowPointers, 0, rows + 1);
            int nzmax = other.columnIndexes.length;
            if (columnIndexes.length < nzmax) {
                columnIndexes = new int[nzmax];
                values = new Object[nzmax];
            }
            System.arraycopy(other.columnIndexes, 0, columnIndexes, 0, nzmax);
            System.arraycopy(other.values, 0, values, 0, nzmax);
            columnIndexesSorted = other.columnIndexesSorted;
        } else if (source instanceof SparseCCObjectMatrix2D) {
            SparseCCObjectMatrix2D other = ((SparseCCObjectMatrix2D) source).getTranspose();
            rowPointers = other.getColumnPointers();
            columnIndexes = other.getRowIndexes();
            values = other.getValues();
            columnIndexesSorted = true;
        } else {
            assign(0);
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
        return rowPointers[rows];
    }

    public ObjectMatrix2D forEachNonZero(final cern.colt.function.tobject.IntIntObjectFunction function) {

        for (int i = rows; --i >= 0;) {
            int low = rowPointers[i];
            for (int k = rowPointers[i + 1]; --k >= low;) {
                int j = columnIndexes[k];
                Object value = values[k];
                Object r = function.apply(i, j, value);
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
    public SparseCCObjectMatrix2D getColumnCompressed() {
        SparseRCObjectMatrix2D tr = getTranspose();
        SparseCCObjectMatrix2D cc = new SparseCCObjectMatrix2D(rows, columns);
        cc.rowIndexes = tr.columnIndexes;
        cc.columnPointers = tr.rowPointers;
        cc.values = tr.values;
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
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        Object v = 0;
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
    public SparseRCObjectMatrix2D getTranspose() {
        int nnz = rowPointers[rows];
        int[] w = new int[columns];
        int[] rowPointersT = new int[columns + 1];
        int[] columnIndexesT = new int[nnz];
        Object[] valuesT = new Object[nnz];

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
        SparseRCObjectMatrix2D T = new SparseRCObjectMatrix2D(columns, rows);
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
    public Object[] getValues() {
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

    public ObjectMatrix2D like(int rows, int columns) {
        return new SparseRCObjectMatrix2D(rows, columns);
    }

    public ObjectMatrix1D like1D(int size) {
        return new SparseObjectMatrix1D(size);
    }

    /**
     * Removes zero entries (if any)
     */
    public void removeZeroes() {
        int nz = 0;
        for (int j = 0; j < rows; j++) {
            int p = rowPointers[j]; /* get current location of row j */
            rowPointers[j] = nz; /* record new location of row j */
            for (; p < rowPointers[j + 1]; p++) {
                if (values[p] != null) {
                    values[nz] = values[p]; /* keep A(i,j) */
                    columnIndexes[nz++] = columnIndexes[p];
                }
            }
        }
        rowPointers[rows] = nz; /* finalize A */
    }

    public synchronized void setQuick(int row, int column, Object value) {
        //        int k = cern.colt.Sorting.binarySearchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);
        int k = searchFromTo(columnIndexes, column, rowPointers[row], rowPointers[row + 1] - 1);

        if (k >= 0) { // found
            if (value == null)
                remove(row, k);
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
     * Sorts column indexes
     */
    public void sortColumnIndexes() {
        SparseRCObjectMatrix2D T = getTranspose();
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
            nzmax = rowPointers[rows];
        int[] columnIndexesNew = new int[nzmax];
        int length = Math.min(nzmax, columnIndexes.length);
        System.arraycopy(columnIndexes, 0, columnIndexesNew, 0, length);
        columnIndexes = columnIndexesNew;
        Object[] valuesNew = new Object[nzmax];
        length = Math.min(nzmax, values.length);
        System.arraycopy(values, 0, valuesNew, 0, length);
        values = valuesNew;
    }

    protected ObjectMatrix2D getContent() {
        return this;
    }

    protected void insert(int row, int column, int index, Object value) {
        IntArrayList columnIndexesList = new IntArrayList(columnIndexes);
        columnIndexesList.setSizeRaw(rowPointers[rows]);
        ObjectArrayList valuesList = new ObjectArrayList(values);
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
        ObjectArrayList valuesList = new ObjectArrayList(values);
        valuesList.setSizeRaw(rowPointers[rows]);
        columnIndexesList.remove(index);
        valuesList.remove(index);
        for (int i = rowPointers.length; --i > row;)
            rowPointers[i]--;
        columnIndexes = columnIndexesList.elements();
        values = valuesList.elements();
    }

}
