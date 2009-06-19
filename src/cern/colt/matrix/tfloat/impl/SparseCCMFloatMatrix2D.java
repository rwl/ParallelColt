package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2D;

/**
 * Sparse column-compressed-modified 2-d matrix holding <tt>float</tt> elements.
 * Each column is stored as SparseFloatMatrix1D.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseCCMFloatMatrix2D extends WrapperFloatMatrix2D {

    private static final long serialVersionUID = 1L;
    private SparseFloatMatrix1D[] elements;

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
     *             <tt>rows<0 || columns<0 || (float)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseCCMFloatMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new SparseFloatMatrix1D[columns];
        for (int i = 0; i < columns; ++i)
            elements[i] = new SparseFloatMatrix1D(rows);
    }

    @Override
    public SparseFloatMatrix1D[] elements() {
        return elements;
    }

    @Override
    public synchronized float getQuick(int row, int column) {
        return elements[column].getQuick(row);
    }

    @Override
    public synchronized void setQuick(int row, int column, float value) {
        elements[column].setQuick(row, value);
    }

    @Override
    public void trimToSize() {
        for (int c = 0; c < columns; c++) {
            elements[c].trimToSize();
        }
    }

    @Override
    public SparseFloatMatrix1D viewColumn(int column) {
        return elements[column];
    }

    @Override
    protected FloatMatrix2D getContent() {
        return this;
    }

    @Override
    public FloatMatrix2D like(int rows, int columns) {
        return new SparseCCMFloatMatrix2D(rows, columns);
    }
}
