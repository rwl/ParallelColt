/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix3D;

/**
 * 1-d matrix holding <tt>float</tt> elements; a view wrapping another 2-d
 * matrix and therefore delegating calls to it.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class DelegateFloatMatrix1D extends FloatMatrix1D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected FloatMatrix2D content;

    /*
     * The row this view is bound to.
     */
    protected int row;

    /**
     * Constructs a matrix view with a given content and row
     * 
     * @param newContent
     *            the content of this view
     * @param row
     *            the row this view is bound to
     */
    public DelegateFloatMatrix1D(FloatMatrix2D newContent, int row) {
        if (row < 0 || row >= newContent.rows())
            throw new IllegalArgumentException();
        setUp(newContent.columns());
        this.row = row;
        this.content = newContent;
    }

    public synchronized float getQuick(int index) {
        return content.getQuick(row, index);
    }

    public FloatMatrix1D like(int size) {
        return content.like1D(size);
    }

    public FloatMatrix2D like2D(int rows, int columns) {
        return content.like(rows, columns);
    }

    public synchronized void setQuick(int index, float value) {
        content.setQuick(row, index, value);
    }

    public Object elements() {
        return content.elements();
    }

    public FloatMatrix2D reshape(int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public FloatMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    protected FloatMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

}
