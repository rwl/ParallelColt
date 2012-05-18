/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2D;

/**
 * Sparse row-compressed-modified 2-d matrix holding <tt>complex</tt> elements.
 * Each row is stored as SparseDComplexMatrix1D.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseRCMDComplexMatrix2D extends WrapperDComplexMatrix2D {

    private static final long serialVersionUID = 1L;
    private SparseDComplexMatrix1D[] elements;

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
    public SparseRCMDComplexMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new SparseDComplexMatrix1D[rows];
        for (int i = 0; i < rows; ++i)
            elements[i] = new SparseDComplexMatrix1D(columns);
    }

    public SparseDComplexMatrix1D[] elements() {
        return elements;
    }

    public double[] getQuick(int row, int column) {
        return elements[row].getQuick(column);
    }

    public void setQuick(int row, int column, double[] value) {
        elements[row].setQuick(column, value);
    }

    public void setQuick(int row, int column, double re, double im) {
        elements[row].setQuick(column, re, im);
    }

    public void trimToSize() {
        for (int r = 0; r < rows; r++) {
            elements[r].trimToSize();
        }
    }

    public SparseDComplexMatrix1D viewRow(int row) {
        return elements[row];
    }

    protected DComplexMatrix2D getContent() {
        return this;
    }

    public DComplexMatrix2D like(int rows, int columns) {
        return new SparseRCMDComplexMatrix2D(rows, columns);
    }
}
