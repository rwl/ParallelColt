package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2D;

/**
 * Sparse column-compressed-modified 2-d matrix holding <tt>double</tt>
 * elements. Each column is stored as SparseDoubleMatrix1D.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseCCMDoubleMatrix2D extends WrapperDoubleMatrix2D {

    private static final long serialVersionUID = 1L;
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
    public SparseCCMDoubleMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new SparseDoubleMatrix1D[columns];
        for (int i = 0; i < columns; ++i)
            elements[i] = new SparseDoubleMatrix1D(rows);
    }

    @Override
    public SparseDoubleMatrix1D[] elements() {
        return elements;
    }

    @Override
    public synchronized double getQuick(int row, int column) {
        return elements[column].getQuick(row);
    }

    @Override
    public synchronized void setQuick(int row, int column, double value) {
        elements[column].setQuick(row, value);
    }

    @Override
    public void trimToSize() {
        for (int c = 0; c < columns; c++) {
            elements[c].trimToSize();
        }
    }

    @Override
    public SparseDoubleMatrix1D viewColumn(int column) {
        return elements[column];
    }

    @Override
    protected DoubleMatrix2D getContent() {
        return this;
    }

    @Override
    public DoubleMatrix2D like(int rows, int columns) {
        return new SparseCCMDoubleMatrix2D(rows, columns);
    }
}
