/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.impl;

import cern.colt.list.FloatArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.FloatMatrix1D;
import cern.colt.matrix.FloatMatrix2D;

/**
 * Sparse row-compressed 2-d matrix holding <tt>float</tt> elements. First see
 * the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally uses the standard sparse row-compressed format, with two important
 * differences that broaden the applicability of this storage format:
 * <ul>
 * <li>We use a {@link cern.colt.list.IntArrayList} and
 * {@link cern.colt.list.FloatArrayList} to hold the column indexes and nonzero
 * values, respectively. This improves set(...) performance, because the
 * standard way of using non-resizable primitive arrays causes excessive memory
 * allocation, garbage collection and array copying. The small downside of this
 * is that set(...,0) does not free memory (The capacity of an arraylist does
 * not shrink upon element removal).
 * <li>Column indexes are kept sorted within a row. This both improves get and
 * set performance on rows with many non-zeros, because we can use a binary
 * search. (Experiments show that this hurts < 10% on rows with < 4 nonZeros.)
 * </ul>
 * <br>
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
 * Getting a cell value takes time<tt> O(log nzr)</tt> where <tt>nzr</tt> is
 * the number of non-zeros of the touched row. This is usually quick, because
 * typically there are only few nonzeros per row. So, in practice, get has
 * <i>expected</i> constant time. Setting a cell value takes <i> </i>worst-case
 * time <tt>O(nz)</tt> where <tt>nzr</tt> is the total number of non-zeros
 * in the matrix. This can be extremely slow, but if you traverse coordinates
 * properly (i.e. upwards), each write is done much quicker: <table>
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
 * </table> If for whatever reasons you can't iterate properly, consider to
 * create an empty dense matrix, store your non-zeros in it, then call
 * <tt>sparse.assign(dense)</tt>. Under the circumstances, this is still
 * rather quick.
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
 * look like: <table>
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
 * </table> Method
 * {@link #assign(FloatMatrix2D,cern.colt.function.FloatFloatFunction)} does
 * just that if you supply {@link cern.jet.math.FloatFunctions#plusMult} as
 * argument.
 * 
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 0.9, 04/14/2000
 */
public class RCFloatMatrix2D extends WrapperFloatMatrix2D {
    /*
     * The elements of the matrix.
     */
    protected IntArrayList indexes;

    protected FloatArrayList values;

    protected int[] starts;

    // protected int N;
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
     *             <tt>for any 1 &lt;= row &lt; values.length: values[row].length != values[row-1].length</tt>.
     */
    public RCFloatMatrix2D(float[][] values) {
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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>.
     */
    public RCFloatMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold
            // rows*columns>Integer.MAX_VALUE
            // cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        indexes = new IntArrayList();
        values = new FloatArrayList();
        starts = new int[rows + 1];
    }

    /**
     * Sets all cells to the state specified by <tt>value</tt>.
     * 
     * @param value
     *            the value to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     */
    public FloatMatrix2D assign(float value) {
        // overriden for performance only
        if (value == 0) {
            indexes.clear();
            values.clear();
            for (int i = starts.length; --i >= 0;)
                starts[i] = 0;
        } else
            super.assign(value);
        return this;
    }

    public FloatMatrix2D assign(final cern.colt.function.FloatFunction function) {
        if (function instanceof cern.jet.math.FloatMult) { // x[i] = mult*x[i]
            final float alpha = ((cern.jet.math.FloatMult) function).multiplicator;
            if (alpha == 1)
                return this;
            if (alpha == 0)
                return assign(0);
            if (alpha != alpha)
                return assign(alpha); // the funny definition of isNaN(). This
            // should better not happen.

            float[] vals = values.elements();
            for (int j = values.size(); --j >= 0;) {
                vals[j] *= alpha;
            }

            /*
             * forEachNonZero( new cern.colt.function.IntIntFloatFunction() {
             * public float apply(int i, int j, float value) { return
             * function.apply(value); } } );
             */
        } else {
            super.assign(function);
        }
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
    public FloatMatrix2D assign(FloatMatrix2D source) {
        if (source == this)
            return this; // nothing to do
        checkShape(source);
        // overriden for performance only
        if (!(source instanceof RCFloatMatrix2D)) {
            // return super.assign(source);

            assign(0);
            source.forEachNonZero(new cern.colt.function.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    setQuick(i, j, value);
                    return value;
                }
            });
            /*
             * indexes.clear(); values.clear(); int nonZeros=0; for (int row=0;
             * row<rows; row++) { starts[row]=nonZeros; for (int column=0;
             * column<columns; column++) { float v =
             * source.getQuick(row,column); if (v!=0) { values.add(v);
             * indexes.add(column); nonZeros++; } } } starts[rows]=nonZeros;
             */
            return this;
        }

        // even quicker
        RCFloatMatrix2D other = (RCFloatMatrix2D) source;

        System.arraycopy(other.starts, 0, this.starts, 0, this.starts.length);
        int s = other.indexes.size();
        this.indexes.setSize(s);
        this.values.setSize(s);
        this.indexes.replaceFromToWithFrom(0, s - 1, other.indexes, 0);
        this.values.replaceFromToWithFrom(0, s - 1, other.values, 0);

        return this;
    }

    public FloatMatrix2D assign(FloatMatrix2D y, cern.colt.function.FloatFloatFunction function) {
        checkShape(y);

        if (function instanceof cern.jet.math.FloatPlusMult) { // x[i] = x[i] +
            // alpha*y[i]
            final float alpha = ((cern.jet.math.FloatPlusMult) function).multiplicator;
            if (alpha == 0)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.IntIntFloatFunction() {
                public float apply(int i, int j, float value) {
                    setQuick(i, j, getQuick(i, j) + alpha * value);
                    return value;
                }
            });
            return this;
        }

        if (function == cern.jet.math.FloatFunctions.mult) { // x[i] = x[i] *
            // y[i]
            int[] idx = indexes.elements();
            float[] vals = values.elements();

            for (int i = starts.length - 1; --i >= 0;) {
                int low = starts[i];
                for (int k = starts[i + 1]; --k >= low;) {
                    int j = idx[k];
                    vals[k] *= y.getQuick(i, j);
                    if (vals[k] == 0)
                        remove(i, j);
                }
            }
            return this;
        }

        if (function == cern.jet.math.FloatFunctions.div) { // x[i] = x[i] /
            // y[i]
            int[] idx = indexes.elements();
            float[] vals = values.elements();

            for (int i = starts.length - 1; --i >= 0;) {
                int low = starts[i];
                for (int k = starts[i + 1]; --k >= low;) {
                    int j = idx[k];
                    vals[k] /= y.getQuick(i, j);
                    if (vals[k] == 0)
                        remove(i, j);
                }
            }
            return this;
        }

        return super.assign(y, function);
    }

    public FloatMatrix2D forEachNonZero(final cern.colt.function.IntIntFloatFunction function) {
        int[] idx = indexes.elements();
        float[] vals = values.elements();

        for (int i = starts.length - 1; --i >= 0;) {
            int low = starts[i];
            for (int k = starts[i + 1]; --k >= low;) {
                int j = idx[k];
                float value = vals[k];
                float r = function.apply(i, j, value);
                if (r != value)
                    vals[k] = r;
            }
        }
        return this;
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     */
    protected FloatMatrix2D getContent() {
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
    public float getQuick(int row, int column) {
        int k = indexes.binarySearchFromTo(column, starts[row], starts[row + 1] - 1);
        float v = 0;
        if (k >= 0)
            v = values.getQuick(k);
        return v;
    }

    protected void insert(int row, int column, int index, float value) {
        indexes.beforeInsert(index, column);
        values.beforeInsert(index, value);
        for (int i = starts.length; --i > row;)
            starts[i]++;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseFloatMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseFloatMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseFloatMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseFloatMatrix2D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public FloatMatrix2D like(int rows, int columns) {
        return new RCFloatMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseFloatMatrix2D</tt> the new
     * matrix must be of type <tt>DenseFloatMatrix1D</tt>, if the receiver is
     * an instance of type <tt>SparseFloatMatrix2D</tt> the new matrix must be
     * of type <tt>SparseFloatMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public FloatMatrix1D like1D(int size) {
        return new SparseFloatMatrix1D(size);
    }

    protected void remove(int row, int index) {
        indexes.remove(index);
        values.remove(index);
        for (int i = starts.length; --i > row;)
            starts[i]--;
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the
     * specified value.
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
    public void setQuick(int row, int column, float value) {
        int k = indexes.binarySearchFromTo(column, starts[row], starts[row + 1] - 1);
        if (k >= 0) { // found
            if (value == 0)
                remove(row, k);
            else
                values.setQuick(k, value);
            return;
        }

        if (value != 0) {
            k = -k - 1;
            insert(row, column, k, value);
        }
    }

    public void trimToSize() {
        indexes.trimToSize();
        values.trimToSize();
    }

    public FloatMatrix1D zMult(FloatMatrix1D y, FloatMatrix1D z, float alpha, float beta, boolean transposeA) {
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }

        boolean ignore = (z == null || !transposeA);
        if (z == null)
            z = new DenseFloatMatrix1D(m);

        if (!(y instanceof DenseFloatMatrix1D && z instanceof DenseFloatMatrix1D)) {
            return super.zMult(y, z, alpha, beta, transposeA);
        }

        if (n != y.size() || m > z.size())
            throw new IllegalArgumentException("Incompatible args: " + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", " + z.toStringShort());

        DenseFloatMatrix1D zz = (DenseFloatMatrix1D) z;
        final float[] zElements = zz.elements;
        final int zStride = zz.stride;
        int zi = z.index(0);

        DenseFloatMatrix1D yy = (DenseFloatMatrix1D) y;
        final float[] yElements = yy.elements;
        final int yStride = yy.stride;
        final int yi = y.index(0);

        if (yElements == null || zElements == null)
            throw new InternalError();

        /*
         * forEachNonZero( new cern.colt.function.IntIntFloatFunction() { public
         * float apply(int i, int j, float value) { zElements[zi + zStride*i] +=
         * value * yElements[yi + yStride*j]; //z.setQuick(row,z.getQuick(row) +
         * value * y.getQuick(column));
         * //System.out.println("["+i+","+j+"]-->"+value); return value; } } );
         */

        int[] idx = indexes.elements();
        float[] vals = values.elements();
        int s = starts.length - 1;
        if (!transposeA) {
            for (int i = 0; i < s; i++) {
                int high = starts[i + 1];
                float sum = 0;
                for (int k = starts[i]; k < high; k++) {
                    int j = idx[k];
                    sum += vals[k] * yElements[yi + yStride * j];
                }
                zElements[zi] = alpha * sum + beta * zElements[zi];
                zi += zStride;
            }
        } else {
            if (!ignore)
                z.assign(cern.jet.math.FloatFunctions.mult(beta));
            for (int i = 0; i < s; i++) {
                int high = starts[i + 1];
                float yElem = alpha * yElements[yi + yStride * i];
                for (int k = starts[i]; k < high; k++) {
                    int j = idx[k];
                    zElements[zi + zStride * j] += vals[k] * yElem;
                }
            }
        }

        return z;
    }

    public FloatMatrix2D zMult(FloatMatrix2D B, FloatMatrix2D C, final float alpha, float beta, boolean transposeA, boolean transposeB) {
        if (transposeB)
            B = B.viewDice();
        int m = rows;
        int n = columns;
        if (transposeA) {
            m = columns;
            n = rows;
        }
        int p = B.columns;
        boolean ignore = (C == null);
        if (C == null)
            C = new DenseFloatMatrix2D(m, p);

        if (B.rows != n)
            throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort());
        if (C.rows != m || C.columns != p)
            throw new IllegalArgumentException("Incompatibel result matrix: " + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort() + ", " + C.toStringShort());
        if (this == C || B == C)
            throw new IllegalArgumentException("Matrices must not be identical");

        if (!ignore)
            C.assign(cern.jet.math.FloatFunctions.mult(beta));

        // cache views
        final FloatMatrix1D[] Brows = new FloatMatrix1D[n];
        for (int i = n; --i >= 0;)
            Brows[i] = B.viewRow(i);
        final FloatMatrix1D[] Crows = new FloatMatrix1D[m];
        for (int i = m; --i >= 0;)
            Crows[i] = C.viewRow(i);

        final cern.jet.math.FloatPlusMult fun = cern.jet.math.FloatPlusMult.plusMult(0);

        int[] idx = indexes.elements();
        float[] vals = values.elements();
        for (int i = starts.length - 1; --i >= 0;) {
            int low = starts[i];
            for (int k = starts[i + 1]; --k >= low;) {
                int j = idx[k];
                fun.multiplicator = vals[k] * alpha;
                if (!transposeA)
                    Crows[i].assign(Brows[j], fun);
                else
                    Crows[j].assign(Brows[i], fun);
            }
        }

        return C;
    }
}
