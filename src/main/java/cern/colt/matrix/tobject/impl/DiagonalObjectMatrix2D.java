/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.matrix.tobject.ObjectMatrix1D;
import cern.colt.matrix.tobject.ObjectMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Diagonal 2-d matrix holding <tt>Object</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DiagonalObjectMatrix2D extends WrapperObjectMatrix2D {
    private static final long serialVersionUID = 1L;

    /*
     * The non zero elements of the matrix.
     */
    protected Object[] elements;

    /*
     * Length of the diagonal
     */
    protected int dlength;

    /*
     * An m-by-n matrix A has m+n-1 diagonals. Since the DiagonalObjectMatrix2D can have only one
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
    public DiagonalObjectMatrix2D(Object[][] values, int dindex) {
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
     *             if <tt>size<0 (Object)size > Integer.MAX_VALUE</tt>.
     */
    public DiagonalObjectMatrix2D(int rows, int columns, int dindex) {
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
        elements = new Object[dlength];
    }

    public ObjectMatrix2D assign(final cern.colt.function.tobject.ObjectFunction function) {
        for (int j = dlength; --j >= 0;) {
            elements[j] = function.apply(elements[j]);
        }
        return this;
    }

    public ObjectMatrix2D assign(Object value) {
        for (int i = dlength; --i >= 0;)
            elements[i] = value;
        return this;
    }

    public ObjectMatrix2D assign(final Object[] values) {
        if (values.length != dlength)
            throw new IllegalArgumentException("Must have same length: length=" + values.length + " dlength=" + dlength);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, dlength);
            Future<?>[] futures = new Future[nthreads];
            int k = dlength / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            elements[r] = values[r];
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int r = dlength; --r >= 0;) {
                elements[r] = values[r];
            }
        }
        return this;
    }

    public ObjectMatrix2D assign(final Object[][] values) {
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
            if (values[i].length != columns) {
                throw new IllegalArgumentException("Must have same number of columns in every row: columns="
                        + values[r].length + "columns()=" + columns());
            }
            elements[i] = values[r++][c++];
        }
        return this;
    }

    public ObjectMatrix2D assign(ObjectMatrix2D source) {
        // overriden for performance only
        if (source == this)
            return this; // nothing to do
        checkShape(source);

        if (source instanceof DiagonalObjectMatrix2D) {
            DiagonalObjectMatrix2D other = (DiagonalObjectMatrix2D) source;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("source is DiagonalObjectMatrix2D with different diagonal stored.");
            }
            // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        } else {
            return super.assign(source);
        }
    }

    public ObjectMatrix2D assign(final ObjectMatrix2D y, final cern.colt.function.tobject.ObjectObjectFunction function) {
        checkShape(y);
        if (y instanceof DiagonalObjectMatrix2D) {
            DiagonalObjectMatrix2D other = (DiagonalObjectMatrix2D) y;
            if ((dindex != other.dindex) || (dlength != other.dlength)) {
                throw new IllegalArgumentException("y is DiagonalObjectMatrix2D with different diagonal stored.");
            }
            final Object[] otherElements = other.elements;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, dlength);
                Future<?>[] futures = new Future[nthreads];
                int k = dlength / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int startrow = j * k;
                    final int stoprow;
                    if (j == nthreads - 1) {
                        stoprow = dlength;
                    } else {
                        stoprow = startrow + k;
                    }
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            for (int j = startrow; j < stoprow; j++) {
                                elements[j] = function.apply(elements[j], otherElements[j]);
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int j = dlength; --j >= 0;) {
                    elements[j] = function.apply(elements[j], otherElements[j]);
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
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? dlength : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Integer>() {
                    public Integer call() throws Exception {
                        int cardinality = 0;
                        for (int r = firstRow; r < lastRow; r++) {
                            if (elements[r] != null)
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
            for (int r = 0; r < dlength; r++) {
                if (elements[r] != null)
                    cardinality++;
            }
        }
        return cardinality;
    }

    public Object[] elements() {
        return elements;
    }

    public boolean equals(Object obj) {
        if (obj instanceof DiagonalObjectMatrix2D) {
            DiagonalObjectMatrix2D other = (DiagonalObjectMatrix2D) obj;
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
            Object[] otherElements = other.elements;
            for (int r = 0; r < dlength; r++) {
                Object x = elements[r];
                Object value = otherElements[r];
                if (!x.equals(value)) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(obj);
        }
    }

    public ObjectMatrix2D forEachNonZero(final cern.colt.function.tobject.IntIntObjectFunction function) {
        for (int j = dlength; --j >= 0;) {
            Object value = elements[j];
            if (value != null) {
                elements[j] = function.apply(j, j, value);
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

    public Object getQuick(int row, int column) {
        if (dindex >= 0) {
            if (column < dindex) {
                return 0;
            } else {
                if ((row < dlength) && (row + dindex == column)) {
                    return elements[row];
                } else {
                    return 0;
                }
            }
        } else {
            if (row < -dindex) {
                return 0;
            } else {
                if ((column < dlength) && (row + dindex == column)) {
                    return elements[column];
                } else {
                    return 0;
                }
            }
        }
    }

    public ObjectMatrix2D like(int rows, int columns) {
        return new SparseObjectMatrix2D(rows, columns);
    }

    public ObjectMatrix1D like1D(int size) {
        return new SparseObjectMatrix1D(size);
    }

    public void setQuick(int row, int column, Object value) {
        if (dindex >= 0) {
            if (column < dindex) {
                //do nothing
            } else {
                if ((row < dlength) && (row + dindex == column)) {
                    elements[row] = value;
                } else {
                    // do nothing
                }
            }
        } else {
            if (row < -dindex) {
                //do nothing
            } else {
                if ((column < dlength) && (row + dindex == column)) {
                    elements[column] = value;
                } else {
                    //do nothing;
                }
            }
        }
    }

    protected ObjectMatrix2D getContent() {
        return this;
    }
}
