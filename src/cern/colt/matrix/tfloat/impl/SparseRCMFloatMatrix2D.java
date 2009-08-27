package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2D;

/**
 * Sparse row-compressed-modified 2-d matrix holding <tt>float</tt> elements.
 * Each row is stored as SparseFloatMatrix1D.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseRCMFloatMatrix2D extends WrapperFloatMatrix2D {

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
     *             <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>
     *             .
     */
    public SparseRCMFloatMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new SparseFloatMatrix1D[rows];
        for (int i = 0; i < rows; ++i)
            elements[i] = new SparseFloatMatrix1D(columns);
    }

    public SparseFloatMatrix1D[] elements() {
        return elements;
    }

    public float getQuick(int row, int column) {
        return elements[row].getQuick(column);
    }

    public void setQuick(int row, int column, float value) {
        elements[row].setQuick(column, value);
    }

    public void trimToSize() {
        for (int r = 0; r < rows; r++) {
            elements[r].trimToSize();
        }
    }

    public SparseFloatMatrix1D viewRow(int row) {
        return elements[row];
    }

    protected FloatMatrix2D getContent() {
        return this;
    }

    public FloatMatrix2D like(int rows, int columns) {
        return new SparseRCMFloatMatrix2D(rows, columns);
    }
}
