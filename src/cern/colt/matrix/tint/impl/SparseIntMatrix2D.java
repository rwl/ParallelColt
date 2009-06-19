/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import cern.colt.map.tint.AbstractIntIntMap;
import cern.colt.map.tint.OpenIntIntHashMap;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;

/**
 * Sparse hashed 2-d matrix holding <tt>int</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Note that this implementation is not synchronized. Uses a
 * {@link cern.colt.map.tint.OpenIntIntHashMap}, which is a compact and
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
 * Thus, a 1000 x 1000 matrix with minLoadFactor=0.25 and maxLoadFactor=0.5 and
 * 1000000 non-zero cells consumes between 25 MB and 50 MB. The same 1000 x 1000
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
 * {@link DenseIntMatrix2D}. However, constant factors are considerably larger.
 * <p>
 * Cells are internally addressed in row-major. Performance sensitive
 * applications can exploit this fact. Setting values in a loop row-by-row is
 * quicker than column-by-column, because fewer hash collisions occur. Thus
 * 
 * <pre>
 * for (int row = 0; row &lt; rows; row++) {
 *     for (int column = 0; column &lt; columns; column++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * is quicker than
 * 
 * <pre>
 * for (int column = 0; column &lt; columns; column++) {
 *     for (int row = 0; row &lt; rows; row++) {
 *         matrix.setQuick(row, column, someValue);
 *     }
 * }
 * </pre>
 * 
 * @see cern.colt.map
 * @see cern.colt.map.tint.OpenIntIntHashMap
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.1, 08/22/2007
 */
public class SparseIntMatrix2D extends IntMatrix2D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected AbstractIntIntMap elements;

    protected int dummy;

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
    public SparseIntMatrix2D(int[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of rows and columns and default
     * memory usage. All entries are initially <tt>0</tt>.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (int)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseIntMatrix2D(int rows, int columns) {
        this(rows, columns, rows * (columns / 1000), 0.2, 0.5);
    }

    /**
     * Constructs a matrix with a given number of rows and columns using memory
     * as specified. All entries are initially <tt>0</tt>. For details related
     * to memory usage see {@link cern.colt.map.tint.OpenIntIntHashMap}.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
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
     *             if
     *             <tt>rows<0 || columns<0 || (int)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseIntMatrix2D(int rows, int columns, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
        setUp(rows, columns);
        this.elements = new OpenIntIntHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
    }

    /**
     * Constructs a view with the given parameters.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @param elements
     *            the cells.
     * @param rowZero
     *            the position of the first element.
     * @param columnZero
     *            the position of the first element.
     * @param rowStride
     *            the number of elements between two rows, i.e.
     *            <tt>index(i+1,j)-index(i,j)</tt>.
     * @param columnStride
     *            the number of elements between two columns, i.e.
     *            <tt>index(i,j+1)-index(i,j)</tt>.
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (int)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    protected SparseIntMatrix2D(int rows, int columns, AbstractIntIntMap elements, int rowZero, int columnZero,
            int rowStride, int columnStride) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
        this.elements = elements;
        this.isNoView = false;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[row,col] = function(x[row,col])</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * 	 matrix = 2 x 2 matrix
     * 	 0.5 1.5      
     * 	 2.5 3.5
     * 
     * 	 // change each cell to its sine
     * 	 matrix.assign(cern.jet.math.Functions.sin);
     * 	 --&gt;
     * 	 2 x 2 matrix
     * 	 0.479426  0.997495 
     * 	 0.598472 -0.350783
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param function
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tint.IntFunctions
     */
    @Override
    public IntMatrix2D assign(cern.colt.function.tint.IntFunction function) {
        if (this.isNoView && function instanceof cern.jet.math.tint.IntMult) { // x[i]
            // =
            // mult*x[i]
            this.elements.assign(function);
        } else {
            super.assign(function);
        }
        return this;
    }

    /**
     * Sets all cells to the state specified by <tt>value</tt>.
     * 
     * @param value
     *            the value to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     */
    @Override
    public IntMatrix2D assign(int value) {
        // overriden for performance only
        if (this.isNoView && value == 0)
            this.elements.clear();
        else
            super.assign(value);
        return this;
    }

    /**
     * Replaces all cell values of the receiver with the values of another
     * matrix. Both matrices must have the same number of rows and columns. If
     * both matrices share the same cells (as is the case if they are views
     * derived from the same matrix) and intersect in an ambiguous way, then
     * replaces <i>as if</i> using an intermediate auxiliary deep copy of
     * <tt>other</tt>.
     * 
     * @param source
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>columns() != source.columns() || rows() != source.rows()</tt>
     */
    @Override
    public IntMatrix2D assign(IntMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof SparseIntMatrix2D)) {
            return super.assign(source);
        }
        SparseIntMatrix2D other = (SparseIntMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);

        if (this.isNoView && other.isNoView) { // quickest
            this.elements.assign(other.elements);
            return this;
        }
        return super.assign(source);
    }

    @Override
    public IntMatrix2D assign(final IntMatrix2D y, cern.colt.function.tint.IntIntFunction function) {
        if (!this.isNoView)
            return super.assign(y, function);

        checkShape(y);

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

        if (function == cern.jet.math.tint.IntFunctions.mult) { // x[i] = x[i] * y[i]
            this.elements.forEachPair(new cern.colt.function.tint.IntIntProcedure() {
                public boolean apply(int key, int value) {
                    int i = key / columns;
                    int j = key % columns;
                    int r = value * y.getQuick(i, j);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        }

        if (function == cern.jet.math.tint.IntFunctions.div) { // x[i] = x[i] /
            // y[i]
            this.elements.forEachPair(new cern.colt.function.tint.IntIntProcedure() {
                public boolean apply(int key, int value) {
                    int i = key / columns;
                    int j = key % columns;
                    int r = value / y.getQuick(i, j);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        }

        return super.assign(y, function);
    }

    /**
     * Returns the number of cells having non-zero values.
     */
    @Override
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
    @Override
    public AbstractIntIntMap elements() {
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
    @Override
    public void ensureCapacity(int minCapacity) {
        this.elements.ensureCapacity(minCapacity);
    }

    @Override
    public IntMatrix2D forEachNonZero(final cern.colt.function.tint.IntIntIntFunction function) {
        if (this.isNoView) {
            this.elements.forEachPair(new cern.colt.function.tint.IntIntProcedure() {
                public boolean apply(int key, int value) {
                    int i = key / columns;
                    int j = key % columns;
                    int r = function.apply(i, j, value);
                    if (r != value)
                        elements.put(key, r);
                    return true;
                }
            });
        } else {
            super.forEachNonZero(function);
        }
        return this;
    }

    /**
     * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @return the value at the specified coordinate.
     */
    @Override
    public int getQuick(int row, int column) {
        // if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
        // throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
        // return this.elements.get(index(row,column));
        // manually inlined:
        return this.elements.get(rowZero + row * rowStride + columnZero + column * columnStride);
    }

    /**
     * Returns the position of the given coordinate within the (virtual or
     * non-virtual) internal 1-dimensional array.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     */
    @Override
    public long index(int row, int column) {
        // return super.index(row,column);
        // manually inlined for speed:
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type <tt>DenseIntMatrix2D</tt>
     * the new matrix must also be of type <tt>DenseIntMatrix2D</tt>, if the
     * receiver is an instance of type <tt>SparseIntMatrix2D</tt> the new matrix
     * must also be of type <tt>SparseIntMatrix2D</tt>, etc. In general, the new
     * matrix should have internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    @Override
    public IntMatrix2D like(int rows, int columns) {
        return new SparseIntMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseIntMatrix2D</tt> the new matrix
     * must be of type <tt>DenseIntMatrix1D</tt>, if the receiver is an instance
     * of type <tt>SparseIntMatrix2D</tt> the new matrix must be of type
     * <tt>SparseIntMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    @Override
    public IntMatrix1D like1D(int size) {
        return new SparseIntMatrix1D(size);
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified
     * value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked):
     * <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
     * 
     * @param row
     *            the index of the row-coordinate.
     * @param column
     *            the index of the column-coordinate.
     * @param value
     *            the value to be filled into the specified cell.
     */
    @Override
    public synchronized void setQuick(int row, int column, int value) {
        // if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
        // throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
        // int index = index(row,column);
        // manually inlined:
        int index = rowZero + row * rowStride + columnZero + column * columnStride;

        // if (value == 0 || Math.abs(value) < TOLERANCE)
        if (value == 0)
            this.elements.removeKey(index);
        else
            this.elements.put(index, value);
    }

    /**
     * Releases any superfluous memory created by explicitly putting zero values
     * into cells formerly having non-zero values; An application can use this
     * operation to minimize the storage of the receiver.
     * <p>
     * <b>Background:</b>
     * <p>
     * Cells that
     * <ul>
     * <li>are never set to non-zero values do not use any memory.
     * <li>switch from zero to non-zero state do use memory.
     * <li>switch back from non-zero to zero state also do use memory. However,
     * their memory can be reclaimed by calling <tt>trimToSize()</tt>.
     * </ul>
     * A sequence like <tt>set(r,c,5); set(r,c,0);</tt> sets a cell to non-zero
     * state and later back to zero state. Such as sequence generates obsolete
     * memory that is automatically reclaimed from time to time or can manually
     * be reclaimed by calling <tt>trimToSize()</tt>. Putting zeros into cells
     * already containing zeros does not generate obsolete memory since no
     * memory was allocated to them in the first place.
     */
    @Override
    public void trimToSize() {
        this.elements.trimToSize();
    }

    /**
     * Returns a vector obtained by stacking the columns of the matrix on top of
     * one another.
     * 
     * @return a vector obtained by stacking the columns of the matrix on top of
     *         one another
     */
    @Override
    public IntMatrix1D vectorize() {
        SparseIntMatrix1D v = new SparseIntMatrix1D((int) size());
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                v.setQuick(idx++, getQuick(c, r));
            }
        }
        return v;
    }

    @Override
    public IntMatrix1D zMult(IntMatrix1D y, IntMatrix1D z, int alpha, int beta, final boolean transposeA) {
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }

        boolean ignore = (z == null);
        if (z == null)
            z = new DenseIntMatrix1D(m);

        if (!(this.isNoView && y instanceof DenseIntMatrix1D && z instanceof DenseIntMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (n != y.size() || m > z.size())
            throw new IllegalArgumentException("Incompatible args: "
                    + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", "
                    + z.toStringShort());

        if (!ignore)
            z.assign(cern.jet.math.tint.IntFunctions.mult(beta / alpha));

        DenseIntMatrix1D zz = (DenseIntMatrix1D) z;
        final int[] zElements = zz.elements;
        final int zStride = zz.stride();
        final int zi = (int) z.index(0);

        DenseIntMatrix1D yy = (DenseIntMatrix1D) y;
        final int[] yElements = yy.elements;
        final int yStride = yy.stride();
        final int yi = (int) y.index(0);

        if (yElements == null || zElements == null)
            throw new InternalError();

        this.elements.forEachPair(new cern.colt.function.tint.IntIntProcedure() {
            public boolean apply(int key, int value) {
                int i = key / columns;
                int j = key % columns;
                if (transposeA) {
                    int tmp = i;
                    i = j;
                    j = tmp;
                }
                zElements[zi + zStride * i] += value * yElements[yi + yStride * j];
                // System.out.println("["+i+","+j+"]-->"+value);
                return true;
            }
        });

        /*
         * forEachNonZero( new cern.colt.function.IntIntIntFunction() {
         * public int apply(int i, int j, int value) { if (transposeA) {
         * int tmp=i; i=j; j=tmp; } zElements[zi + zStride*i] += value *
         * yElements[yi + yStride*j]; //z.setQuick(row,z.getQuick(row) + value *
         * y.getQuick(column)); //System.out.println("["+i+","+j+"]-->"+value);
         * return value; } } );
         */

        if (alpha != 1)
            z.assign(cern.jet.math.tint.IntFunctions.mult(alpha));
        return z;
    }

    @Override
    public IntMatrix2D zMult(IntMatrix2D B, IntMatrix2D C, final int alpha, int beta, final boolean transposeA,
            boolean transposeB) {
        if (!(this.isNoView)) {
            return super.zMult(B, C, alpha, beta, transposeA, transposeB);
        }
        if (transposeB)
            B = B.viewDice();
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }
        int p = B.columns();
        boolean ignore = (C == null);
        if (C == null)
            C = new DenseIntMatrix2D(m, p);

        if (B.rows() != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", "
                    + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows() != m || C.columns() != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", "
                    + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!ignore)
            C.assign(cern.jet.math.tint.IntFunctions.mult(beta));

        // cache views
        final IntMatrix1D[] Brows = new IntMatrix1D[n];
        for (int i = n; --i >= 0;)
            Brows[i] = B.viewRow(i);
        final IntMatrix1D[] Crows = new IntMatrix1D[m];
        for (int i = m; --i >= 0;)
            Crows[i] = C.viewRow(i);

        final cern.jet.math.tint.IntPlusMultSecond fun = cern.jet.math.tint.IntPlusMultSecond.plusMult(0);

        this.elements.forEachPair(new cern.colt.function.tint.IntIntProcedure() {
            public boolean apply(int key, int value) {
                int i = key / columns;
                int j = key % columns;
                fun.multiplicator = value * alpha;
                if (!transposeA)
                    Crows[i].assign(Brows[j], fun);
                else
                    Crows[j].assign(Brows[i], fun);
                return true;
            }
        });

        return C;
    }

    /**
     * Returns <tt>true</tt> if both matrices share common cells. More formally,
     * returns <tt>true</tt> if at least one of the following conditions is met
     * <ul>
     * <li>the receiver is a view of the other matrix
     * <li>the other matrix is a view of the receiver
     * <li><tt>this == other</tt>
     * </ul>
     */
    @Override
    protected boolean haveSharedCellsRaw(IntMatrix2D other) {
        if (other instanceof SelectedSparseIntMatrix2D) {
            SelectedSparseIntMatrix2D otherMatrix = (SelectedSparseIntMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseIntMatrix2D) {
            SparseIntMatrix2D otherMatrix = (SparseIntMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseIntMatrix2D</tt> the new matrix must be of type
     * <tt>DenseIntMatrix1D</tt>, if the receiver is an instance of type
     * <tt>SparseIntMatrix2D</tt> the new matrix must be of type
     * <tt>SparseIntMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param offset
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @return a new matrix of the corresponding dynamic type.
     */
    @Override
    protected IntMatrix1D like1D(int size, int offset, int stride) {
        return new SparseIntMatrix1D(size, this.elements, offset, stride);
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param rowOffsets
     *            the offsets of the visible elements.
     * @param columnOffsets
     *            the offsets of the visible elements.
     * @return a new view.
     */
    @Override
    protected IntMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseIntMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }

}
