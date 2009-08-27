/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import cern.colt.map.tdouble.AbstractLongDoubleMap;
import cern.colt.map.tdouble.OpenLongDoubleHashMap;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;

/**
 * Sparse hashed 1-d matrix (aka <i>vector</i>) holding <tt>double</tt>
 * elements. First see the <a href="package-summary.html">package summary</a>
 * and javadoc <a href="package-tree.html">tree view</a> to get the broad
 * picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Note that this implementation is not synchronized. Uses a
 * {@link cern.colt.map.tdouble.OpenIntDoubleHashMap}, which is a compact and
 * performant hashing technique.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * Cells that
 * <ul>
 * <li>are never set to non-zero values do not use any memory.
 * <li>switch from zero to non-zero state do use memory.
 * <li>switch back from non-zero to zero state also do use memory. However,
 * their memory is automatically reclaimed from time to time. It can also
 * manually be reclaimed by calling {@link #trimToSize()}.
 * </ul>
 * <p>
 * worst case: <tt>memory [bytes] = (1/minLoadFactor) * nonZeros * 13</tt>. <br>
 * best case: <tt>memory [bytes] = (1/maxLoadFactor) * nonZeros * 13</tt>. <br>
 * Where <tt>nonZeros = cardinality()</tt> is the number of non-zero cells.
 * Thus, a 1000000 matrix with minLoadFactor=0.25 and maxLoadFactor=0.5 and
 * 1000000 non-zero cells consumes between 25 MB and 50 MB. The same 1000000
 * matrix with 1000 non-zero cells consumes between 25 and 50 KB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * This class offers <i>expected</i> time complexity <tt>O(1)</tt> (i.e.
 * constant time) for the basic operations <tt>get</tt>, <tt>getQuick</tt>,
 * <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt> assuming the hash function
 * disperses the elements properly among the buckets. Otherwise, pathological
 * cases, although highly improbable, can occur, degrading performance to
 * <tt>O(N)</tt> in the worst case. As such this sparse class is expected to
 * have no worse time complexity than its dense counterpart
 * {@link DenseDoubleMatrix1D}. However, constant factors are considerably
 * larger.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.1, 08/22/2007
 */
public class SparseDoubleMatrix1D extends DoubleMatrix1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected AbstractLongDoubleMap elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public SparseDoubleMatrix1D(double[] values) {
        this(values.length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of cells. All entries are
     * initially <tt>0</tt>.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public SparseDoubleMatrix1D(int size) {
        this(size, size / 1000, 0.2, 0.5);
    }

    /**
     * Constructs a matrix with a given number of parameters. All entries are
     * initially <tt>0</tt>. For details related to memory usage see
     * {@link cern.colt.map.tdouble.OpenIntDoubleHashMap}.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param initialCapacity
     *            the initial capacity of the hash map. If not known, set
     *            <tt>initialCapacity=0</tt> or small.
     * @param minLoadFactor
     *            the minimum load factor of the hash map.
     * @param maxLoadFactor
     *            the maximum load factor of the hash map.
     * @throws IllegalArgumentException
     *             if
     * 
     *             <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) || (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >= maxLoadFactor)</tt>
     *             .
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public SparseDoubleMatrix1D(int size, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
        setUp(size);
        this.elements = new OpenLongDoubleHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
    }

    /**
     * Constructs a matrix view with a given number of parameters.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param elements
     *            the cells.
     * @param offset
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    protected SparseDoubleMatrix1D(int size, AbstractLongDoubleMap elements, int offset, int stride) {
        setUp(size, offset, stride);
        this.elements = elements;
        this.isNoView = false;
    }

    /**
     * Sets all cells to the state specified by <tt>value</tt>.
     * 
     * @param value
     *            the value to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     */

    public DoubleMatrix1D assign(double value) {
        // overriden for performance only
        if (this.isNoView && value == 0)
            this.elements.clear();
        else
            super.assign(value);
        return this;
    }

    /**
     * Returns the number of cells having non-zero values.
     */

    public int cardinality() {
        if (this.isNoView)
            return this.elements.size();
        else
            return super.cardinality();
    }

    /**
     * Returns the elements of this matrix.
     * 
     * @return the elements
     */

    public AbstractLongDoubleMap elements() {
        return elements;
    }

    /**
     * Ensures that the receiver can hold at least the specified number of
     * non-zero cells without needing to allocate new internal memory. If
     * necessary, allocates new internal memory and increases the capacity of
     * the receiver.
     * <p>
     * This method never need be called; it is for performance tuning only.
     * Calling this method before tt>set()</tt>ing a large number of non-zero
     * values boosts performance, because the receiver will grow only once
     * instead of potentially many times and hash collisions get less probable.
     * 
     * @param minCapacity
     *            the desired minimum number of non-zero cells.
     */

    public void ensureCapacity(int minCapacity) {
        this.elements.ensureCapacity(minCapacity);
    }

    /**
     * Returns the matrix cell value at coordinate <tt>index</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @return the value of the specified cell.
     */

    public synchronized double getQuick(int index) {
        // if (debug) if (index<0 || index>=size) checkIndex(index);
        // return this.elements.get(index(index));
        // manually inlined:
        return elements.get((long) zero + (long) index * (long) stride);
    }

    /**
     * Returns the position of the element with the given relative rank within
     * the (virtual or non-virtual) internal 1-dimensional array. You may want
     * to override this method for performance.
     * 
     * @param rank
     *            the rank of the element.
     */

    public long index(int rank) {
        // overriden for manual inlining only
        // return _offset(_rank(rank));
        return (long) zero + (long) rank * (long) stride;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified size. For example, if the receiver
     * is an instance of type <tt>DenseDoubleMatrix1D</tt> the new matrix must
     * also be of type <tt>DenseDoubleMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseDoubleMatrix1D</tt> the new matrix must also
     * be of type <tt>SparseDoubleMatrix1D</tt>, etc. In general, the new matrix
     * should have internal parametrization as similar as possible.
     * 
     * @param size
     *            the number of cell the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */

    public DoubleMatrix1D like(int size) {
        return new SparseDoubleMatrix1D(size);
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseDoubleMatrix1D</tt> the new
     * matrix must be of type <tt>DenseDoubleMatrix2D</tt>, if the receiver is
     * an instance of type <tt>SparseDoubleMatrix1D</tt> the new matrix must be
     * of type <tt>SparseDoubleMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */

    public DoubleMatrix2D like2D(int rows, int columns) {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    public DoubleMatrix2D reshape(int rows, int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        DoubleMatrix2D M = new SparseDoubleMatrix2D(rows, columns);
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                double elem = getQuick(idx++);
                if (elem != 0) {
                    M.setQuick(r, c, elem);
                }
            }
        }
        return M;
    }

    public DoubleMatrix3D reshape(int slices, int rows, int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        DoubleMatrix3D M = new SparseDoubleMatrix3D(slices, rows, columns);
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    double elem = getQuick(idx++);
                    if (elem != 0) {
                        M.setQuick(s, r, c, elem);
                    }
                }
            }
        }
        return M;
    }

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @param value
     *            the value to be filled into the specified cell.
     */

    public synchronized void setQuick(int index, double value) {
        // if (debug) if (index<0 || index>=size) checkIndex(index);
        // int i = index(index);
        // manually inlined:
        long i = (long) zero + (long) index * (long) stride;
        if (value == 0)
            this.elements.removeKey(i);
        else
            this.elements.put(i, value);
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("1 x ").append(size).append(" sparse matrix, nnz = ").append(cardinality()).append('\n');
        for (int i = 0; i < size; i++) {
            double elem = getQuick(i);
            if (elem != 0) {
                builder.append('(').append(i).append(')').append('\t').append(elem).append('\n');
            }
        }
        return builder.toString();
    }

    public void trimToSize() {
        this.elements.trimToSize();
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */

    protected boolean haveSharedCellsRaw(DoubleMatrix1D other) {
        if (other instanceof SelectedSparseDoubleMatrix1D) {
            SelectedSparseDoubleMatrix1D otherMatrix = (SelectedSparseDoubleMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseDoubleMatrix1D) {
            SparseDoubleMatrix1D otherMatrix = (SparseDoubleMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param offsets
     *            the offsets of the visible elements.
     * @return a new view.
     */

    protected DoubleMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedSparseDoubleMatrix1D(this.elements, offsets);
    }
}
