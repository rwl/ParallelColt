/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix1D;

/**
 * 1-d matrix holding <tt>complex</tt> elements; either a view wrapping another
 * 2-d matrix and therefore delegating calls to it.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class DelegateFComplexMatrix1D extends FComplexMatrix1D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected FComplexMatrix2D content;

    /*
     * The row this view is bound to.
     */
    protected int row;

    /**
     * Creates new instance of DelegateFComplexMatrix1D
     * 
     * @param newContent
     *            the content
     * @param row
     *            the row this view is bound to
     */
    public DelegateFComplexMatrix1D(FComplexMatrix2D newContent, int row) {
        if (row < 0 || row >= newContent.rows())
            throw new IllegalArgumentException();
        setUp(newContent.columns());
        this.row = row;
        this.content = newContent;
    }

    public synchronized float[] getQuick(int index) {
        return content.getQuick(row, index);
    }

    public FComplexMatrix1D like(int size) {
        return content.like1D(size);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return content.like(rows, columns);
    }

    public synchronized void setQuick(int index, float[] value) {
        content.setQuick(row, index, value);
    }

    public synchronized void setQuick(int index, float re, float im) {
        content.setQuick(row, index, re, im);
    }

    public Object elements() {
        return content.elements();
    }

    public FComplexMatrix2D reshape(int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public FComplexMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    protected FComplexMatrix1D viewSelectionLike(int[] offsets) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public FloatMatrix1D getImaginaryPart() {
        return content.viewRow(row).getImaginaryPart();
    }

    public FloatMatrix1D getRealPart() {
        return content.viewRow(row).getRealPart();
    }

}
