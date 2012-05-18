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
 * Sparse column-compressed-modified 2-d matrix holding <tt>complex</tt>
 * elements. Each column is stored as SparseDComplexMatrix1D.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseCCMDComplexMatrix2D extends WrapperDComplexMatrix2D {

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
    public SparseCCMDComplexMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new SparseDComplexMatrix1D[columns];
        for (int i = 0; i < columns; ++i)
            elements[i] = new SparseDComplexMatrix1D(rows);
    }

    public SparseDComplexMatrix1D[] elements() {
        return elements;
    }

    public double[] getQuick(int row, int column) {
        return elements[column].getQuick(row);
    }

    public void setQuick(int row, int column, double[] value) {
        elements[column].setQuick(row, value);
    }

    public void setQuick(int row, int column, double re, double im) {
        elements[column].setQuick(row, re, im);
    }

    public void trimToSize() {
        for (int c = 0; c < columns; c++) {
            elements[c].trimToSize();
        }
    }

    public SparseDComplexMatrix1D viewColumn(int column) {
        return elements[column];
    }

    protected DComplexMatrix2D getContent() {
        return this;
    }

    public DComplexMatrix2D like(int rows, int columns) {
        return new SparseCCMDComplexMatrix2D(rows, columns);
    }
}
