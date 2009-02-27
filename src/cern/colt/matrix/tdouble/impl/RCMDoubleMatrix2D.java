package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2D;

/**
 * Sparse row-compressed-modified 2-d matrix holding <tt>double</tt> elements.
 * Each row is stored as SparseDoubleMatrix1D.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class RCMDoubleMatrix2D extends WrapperDoubleMatrix2D {

    private SparseDoubleMatrix1D[] elements;

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
    public RCMDoubleMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new SparseDoubleMatrix1D[rows];
        for (int i = 0; i < rows; ++i)
            elements[i] = new SparseDoubleMatrix1D(columns);
    }

    public SparseDoubleMatrix1D[] elements() {
        return elements;
    }

    public double getQuick(int row, int column) {
        return elements[row].getQuick(column);
    }

    public void setQuick(int row, int column, double value) {
        elements[row].setQuick(column, value);
    }

    public void trimToSize() {
        for (int r = 0; r < rows; r++) {
            elements[r].trimToSize();
        }
    }

    public SparseDoubleMatrix1D viewRow(int row) {
        return elements[row];
    }

    protected DoubleMatrix2D getContent() {
        return this;
    }

    public DoubleMatrix2D like(int rows, int columns) {
        return new RCMDoubleMatrix2D(rows, columns);
    }
}
