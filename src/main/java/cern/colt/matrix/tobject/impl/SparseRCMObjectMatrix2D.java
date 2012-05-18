/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import cern.colt.matrix.tobject.ObjectMatrix2D;

/**
 * Sparse row-compressed-modified 2-d matrix holding <tt>Object</tt> elements.
 * Each row is stored as SparseObjectMatrix1D.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class SparseRCMObjectMatrix2D extends WrapperObjectMatrix2D {

    private static final long serialVersionUID = 1L;
    private SparseObjectMatrix1D[] elements;

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
    public SparseRCMObjectMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new SparseObjectMatrix1D[rows];
        for (int i = 0; i < rows; ++i)
            elements[i] = new SparseObjectMatrix1D(columns);
    }

    public SparseObjectMatrix1D[] elements() {
        return elements;
    }

    public Object getQuick(int row, int column) {
        return elements[row].getQuick(column);
    }

    public void setQuick(int row, int column, Object value) {
        elements[row].setQuick(column, value);
    }

    public void trimToSize() {
        for (int r = 0; r < rows; r++) {
            elements[r].trimToSize();
        }
    }

    public SparseObjectMatrix1D viewRow(int row) {
        return elements[row];
    }

    protected ObjectMatrix2D getContent() {
        return this;
    }

    public ObjectMatrix2D like(int rows, int columns) {
        return new SparseRCMObjectMatrix2D(rows, columns);
    }
}
