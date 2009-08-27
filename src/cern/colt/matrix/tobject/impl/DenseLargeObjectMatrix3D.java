/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import cern.colt.matrix.tobject.ObjectMatrix3D;

/**
 * Dense 3-d matrix holding <tt>Object</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * This data structure allows to store more than 2^31 elements. Internally holds
 * one three-dimensional array, elements[slices][rows][columns]. Note that this
 * implementation is not synchronized.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DenseLargeObjectMatrix3D extends WrapperObjectMatrix3D {

    private static final long serialVersionUID = 1L;

    private Object[][][] elements;

    public DenseLargeObjectMatrix3D(int slices, int rows, int columns) {
        super(null);
        try {
            setUp(slices, rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold slices*rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new Object[slices][rows][columns];
    }

    public Object getQuick(int slice, int row, int column) {
        return elements[slice][row][column];
    }

    public void setQuick(int slice, int row, int column, Object value) {
        elements[slice][row][column] = value;
    }

    public Object[][][] elements() {
        return elements;
    }

    protected ObjectMatrix3D getContent() {
        return this;
    }

    public ObjectMatrix3D like(int slices, int rows, int columns) {
        return new DenseLargeObjectMatrix3D(slices, rows, columns);
    }

}
