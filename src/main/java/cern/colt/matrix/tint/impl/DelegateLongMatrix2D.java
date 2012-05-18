/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import cern.colt.matrix.tint.IntMatrix3D;

/**
 * 2-d matrix holding <tt>int</tt> elements; a view wrapping another 3-d matrix
 * and therefore delegating calls to it.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
class DelegateIntMatrix2D extends IntMatrix2D {
    private static final long serialVersionUID = 1L;

    /*
     * The elements of the matrix.
     */
    protected IntMatrix3D content;

    /*
     * The index this view is bound to.
     */
    protected int index;

    protected int axis; //0-2

    /**
     * Constructs a matrix view with a given content, axis and index
     * 
     * @param newContent
     *            the content of this view
     * @param axis
     *            the axis (0 to 2) this view is bound to
     * @param index
     *            the index this view is bound to
     */
    public DelegateIntMatrix2D(IntMatrix3D newContent, int axis, int index) {
        switch (axis) {
        case 0:
            if (index < 0 || index >= newContent.slices())
                throw new IllegalArgumentException();
            setUp(newContent.rows(), newContent.columns());
            break;
        case 1:
            if (index < 0 || index >= newContent.rows())
                throw new IllegalArgumentException();
            setUp(newContent.slices(), newContent.columns());
            break;
        case 2:
            if (index < 0 || index >= newContent.columns())
                throw new IllegalArgumentException();
            setUp(newContent.slices(), newContent.rows());
            break;
        default:
            throw new IllegalArgumentException();
        }
        this.axis = axis;
        this.index = index;
        this.content = newContent;
    }

    public synchronized int getQuick(int row, int column) {
        switch (axis) {
        case 0:
            return content.getQuick(index, row, column);
        case 1:
            return content.getQuick(row, index, column);
        case 2:
            return content.getQuick(row, column, index);
        default:
            throw new IllegalArgumentException();
        }
    }

    public IntMatrix2D like(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    public synchronized void setQuick(int row, int column, int value) {
        switch (axis) {
        case 0:
            content.setQuick(index, row, column, value);
            break;
        case 1:
            content.setQuick(row, index, column, value);
            break;
        case 2:
            content.setQuick(row, column, index, value);
            break;
        default:
            throw new IllegalArgumentException();
        }
    }

    public IntMatrix1D viewColumn(int column) {
        checkColumn(column);
        return new WrapperIntMatrix2D(this).viewColumn(column);
    }

    public Object elements() {
        return content.elements();
    }

    protected IntMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        throw new InternalError(); // should never get called
    }

    public IntMatrix1D like1D(int size) {
        throw new InternalError(); // should never get called
    }

    protected IntMatrix1D like1D(int size, int zero, int stride) {
        throw new InternalError(); // should never get called
    }

    public IntMatrix1D vectorize() {
        IntMatrix1D v = new DenseIntMatrix1D(rows * columns);
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                v.setQuick(idx++, getQuick(r, c));
            }
        }
        return v;
    }

}
