/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix1D;
import cern.colt.matrix.tlong.LongMatrix2D;

/**
 * Dense 2-d matrix holding <tt>long</tt> elements. First see the <a
 * href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * This data structure allows to store more than 2^31 elements. Internally holds
 * one two-dimensional array, elements[rows][columns]. Note that this
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
public class DenseLargeLongMatrix2D extends WrapperLongMatrix2D {

    private static final long serialVersionUID = 1L;

    private long[][] elements;

    public DenseLargeLongMatrix2D(int rows, int columns) {
        super(null);
        try {
            setUp(rows, columns);
        } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
            if (!"matrix too large".equals(exc.getMessage()))
                throw exc;
        }
        elements = new long[rows][columns];
        content = this;
    }

    public long getQuick(int row, int column) {
        return elements[row][column];
    }

    public void setQuick(int row, int column, long value) {
        elements[row][column] = value;
    }

    public long[][] elements() {
        return elements;
    }

    protected LongMatrix2D getContent() {
        return this;
    }

    public LongMatrix2D like(int rows, int columns) {
        return new DenseLargeLongMatrix2D(rows, columns);
    }

    public LongMatrix1D like1D(int size) {
        return new DenseLongMatrix1D(size);
    }
}
