/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import java.util.concurrent.ConcurrentHashMap;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplex;

/**
 * Sparse hashed 2-d matrix holding <tt>complex</tt> elements.
 * 
 * Note that this implementation uses ConcurrentHashMap
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @version 1.0, 12/10/2007
 * 
 */
public class SparseDComplexMatrix2D extends DComplexMatrix2D {
    private static final long serialVersionUID = 4055279694434233679L;

    /*
     * The elements of the matrix.
     */
    protected ConcurrentHashMap<Integer, double[]> elements;

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
     *             <tt>for any 1 &lt;= row &lt; values.length: values[row].length != values[row-1].length</tt>.
     */
    public SparseDComplexMatrix2D(double[][] values) {
        this(values.length, values.length == 0 ? 0 : values[0].length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of rows and columns and default
     * memory usage.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @throws IllegalArgumentException
     *             if
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
     */
    public SparseDComplexMatrix2D(int rows, int columns) {
        setUp(rows, columns);
        this.elements = new ConcurrentHashMap<Integer, double[]>(rows * (columns / 1000));
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
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             or flip's are illegal.
     */
    protected SparseDComplexMatrix2D(int rows, int columns, ConcurrentHashMap<Integer, double[]> elements, int rowZero, int columnZero, int rowStride, int columnStride) {
        setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
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
    public DComplexMatrix2D assign(double[] value) {
        // overriden for performance only
        if (this.isNoView && value[0] == 0 && value[1] == 0)
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
    public DComplexMatrix2D assign(DComplexMatrix2D source) {
        // overriden for performance only
        if (!(source instanceof SparseDComplexMatrix2D)) {
            return super.assign(source);
        }
        SparseDComplexMatrix2D other = (SparseDComplexMatrix2D) source;
        if (other == this)
            return this; // nothing to do
        checkShape(other);

        if (this.isNoView && other.isNoView) { // quickest
            this.elements.clear();
            this.elements.putAll(other.elements);
            return this;
        }
        return super.assign(source);
    }

    /**
     * Assigns the result of a function to each cell.
     * 
     * @param y
     *            the secondary matrix to operate on.
     * @param function
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if
     *             <tt>columns() != other.columns() || rows() != other.rows()</tt>
     * @see cern.jet.math.tdcomplex.DComplexFunctions
     */
    public DComplexMatrix2D assign(final DComplexMatrix2D y, cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction function) {
        if (!this.isNoView)
            return super.assign(y, function);

        checkShape(y);

        if (function instanceof cern.jet.math.tdcomplex.DComplexPlusMult) {
            // x[i] = x[i] + alpha*y[i]
            final double[] alpha = ((cern.jet.math.tdcomplex.DComplexPlusMult) function).multiplicator;
            if (alpha[0] == 0 && alpha[1] == 1)
                return this; // nothing to do
            y.forEachNonZero(new cern.colt.function.tdcomplex.IntIntDComplexFunction() {
                public double[] apply(int i, int j, double[] value) {
                    setQuick(i, j, DComplex.plus(getQuick(i, j), DComplex.mult(alpha, value)));
                    return value;
                }
            });
            return this;
        }
        return super.assign(y, function);
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

    public void fft2() {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void ifft2(boolean scale) {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void fftRows() {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void ifftRows(boolean scale) {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void fftColumns() {
        throw new IllegalArgumentException("This method is not supported yet");
    }

    public void ifftColumns(boolean scale) {
        throw new IllegalArgumentException("This method is not supported yet");
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
    public double[] getQuick(int row, int column) {
        return this.elements.get(rowZero + row * rowStride + columnZero + column * columnStride);
    }

    /**
     * Returns the elements of this matrix.
     * 
     * @return the elements
     */
    public ConcurrentHashMap<Integer, double[]> elements() {
        return elements;
    }

    /**
     * Returns <tt>true</tt> if both matrices share common cells. More
     * formally, returns <tt>true</tt> if at least one of the following
     * conditions is met
     * <ul>
     * <li>the receiver is a view of the other matrix
     * <li>the other matrix is a view of the receiver
     * <li><tt>this == other</tt>
     * </ul>
     */
    protected boolean haveSharedCellsRaw(DComplexMatrix2D other) {
        if (other instanceof SelectedSparseDComplexMatrix2D) {
            SelectedSparseDComplexMatrix2D otherMatrix = (SelectedSparseDComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof SparseDComplexMatrix2D) {
            SparseDComplexMatrix2D otherMatrix = (SparseDComplexMatrix2D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
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
    public int index(int row, int column) {
        return rowZero + row * rowStride + columnZero + column * columnStride;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns. For
     * example, if the receiver is an instance of type
     * <tt>DenseComplexMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseComplexMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseComplexMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseComplexMatrix2D</tt>, etc. In general, the new matrix should
     * have internal parametrization as similar as possible.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */
    public DComplexMatrix2D like(int rows, int columns) {
        return new SparseDComplexMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseComplexMatrix2D</tt> the new
     * matrix must be of type <tt>DenseComplexMatrix1D</tt>, if the receiver
     * is an instance of type <tt>SparseComplexMatrix2D</tt> the new matrix
     * must be of type <tt>SparseComplexMatrix1D</tt>, etc.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */
    public DComplexMatrix1D like1D(int size) {
        return new SparseDComplexMatrix1D(size);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells. For example, if the receiver is an
     * instance of type <tt>DenseComplexMatrix2D</tt> the new matrix must be
     * of type <tt>DenseComplexMatrix1D</tt>, if the receiver is an instance
     * of type <tt>SparseComplexMatrix2D</tt> the new matrix must be of type
     * <tt>SparseComplexMatrix1D</tt>, etc.
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
    protected DComplexMatrix1D like1D(int size, int offset, int stride) {
        return new SparseDComplexMatrix1D(size, this.elements, offset, stride);
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
    public void setQuick(int row, int column, double[] value) {
        int index = rowZero + row * rowStride + columnZero + column * columnStride;
        if (value[0] == 0 && value[1] == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, value);
    }

    /**
     * Returns a vector obtained by stacking the columns of the matrix on top of
     * one another.
     * 
     * @return a vector obtained by stacking the columns of the matrix on top of
     *         one another.
     */
    public DComplexMatrix1D vectorize() {
        SparseDComplexMatrix1D v = new SparseDComplexMatrix1D(size());
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                v.setQuick(idx++, getQuick(c, r));
            }
        }
        return v;
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
     * @param re
     *            the real part of the value to be filled into the specified
     *            cell.
     * @param im
     *            the imaginary part of the value to be filled into the
     *            specified cell.
     * 
     */
    public void setQuick(int row, int column, double re, double im) {
        int index = rowZero + row * rowStride + columnZero + column * columnStride;
        if (re == 0 && im == 0)
            this.elements.remove(index);
        else
            this.elements.put(index, new double[] { re, im });

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
    protected DComplexMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        return new SelectedSparseDComplexMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
    }

    /**
     * Returns the imaginary part of this matrix
     * 
     * @return the imaginary part
     */
    public DoubleMatrix2D getImaginaryPart() {
        DoubleMatrix2D Im = new SparseDoubleMatrix2D(rows, columns);
        double[] tmp;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                tmp = getQuick(r, c);
                Im.setQuick(r, c, tmp[1]);
            }
        }
        return Im;
    }

    /**
     * Returns the real part of this matrix
     * 
     * @return the real part
     */
    public DoubleMatrix2D getRealPart() {
        DoubleMatrix2D R = new SparseDoubleMatrix2D(rows, columns);
        double[] tmp;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                tmp = getQuick(r, c);
                R.setQuick(r, c, tmp[0]);
            }
        }
        return R;
    }
}
