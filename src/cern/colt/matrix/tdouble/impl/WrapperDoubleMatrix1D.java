/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;

/**
 * 1-d matrix holding <tt>double</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperDoubleMatrix1D extends DoubleMatrix1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected DoubleMatrix1D content;

    public WrapperDoubleMatrix1D(DoubleMatrix1D newContent) {
        if (newContent != null)
            setUp((int) newContent.size());
        this.content = newContent;
    }

    @Override
    protected DoubleMatrix1D getContent() {
        return this.content;
    }

    @Override
    public double getQuick(int index) {
        return content.getQuick(index);
    }

    @Override
    public Object elements() {
        return content.elements();
    }

    @Override
    public DoubleMatrix1D like(int size) {
        return content.like(size);
    }

    @Override
    public DoubleMatrix2D like2D(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    @Override
    public DoubleMatrix2D reshape(int rows, int cols) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    @Override
    public DoubleMatrix3D reshape(int slices, int rows, int cols) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    @Override
    public void setQuick(int index, double value) {
        content.setQuick(index, value);
    }

    @Override
    public DoubleMatrix1D viewFlip() {
        DoubleMatrix1D view = new WrapperDoubleMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public double getQuick(int index) {
                return content.getQuick(size - 1 - index);
            }

            @Override
            public void setQuick(int index, double value) {
                content.setQuick(size - 1 - index, value);
            }

            @Override
            public double get(int index) {
                return content.get(size - 1 - index);
            }

            @Override
            public void set(int index, double value) {
                content.set(size - 1 - index, value);
            }
        };
        return view;
    }

    @Override
    public DoubleMatrix1D viewPart(final int index, int width) {
        checkRange(index, width);
        WrapperDoubleMatrix1D view = new WrapperDoubleMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public double getQuick(int i) {
                return content.getQuick(index + i);
            }

            @Override
            public void setQuick(int i, double value) {
                content.setQuick(index + i, value);
            }

            @Override
            public double get(int i) {
                return content.get(index + i);
            }

            @Override
            public void set(int i, double value) {
                content.set(index + i, value);
            }
        };
        view.size = width;
        return view;
    }

    @Override
    public DoubleMatrix1D viewSelection(int[] indexes) {
        // check for "all"
        if (indexes == null) {
            indexes = new int[size];
            for (int i = size; --i >= 0;)
                indexes[i] = i;
        }

        checkIndexes(indexes);
        final int[] idx = indexes;

        WrapperDoubleMatrix1D view = new WrapperDoubleMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public double getQuick(int i) {
                return content.getQuick(idx[i]);
            }

            @Override
            public void setQuick(int i, double value) {
                content.setQuick(idx[i], value);
            }

            @Override
            public double get(int i) {
                return content.get(idx[i]);
            }

            @Override
            public void set(int i, double value) {
                content.set(idx[i], value);
            }
        };
        view.size = indexes.length;
        return view;
    }

    @Override
    protected DoubleMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

    @Override
    public DoubleMatrix1D viewStrides(final int _stride) {
        if (stride <= 0)
            throw new IndexOutOfBoundsException("illegal stride: " + stride);
        WrapperDoubleMatrix1D view = new WrapperDoubleMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public double getQuick(int index) {
                return content.getQuick(index * _stride);
            }

            @Override
            public void setQuick(int index, double value) {
                content.setQuick(index * _stride, value);
            }

            @Override
            public double get(int index) {
                return content.get(index * _stride);
            }

            @Override
            public void set(int index, double value) {
                content.set(index * _stride, value);
            }
        };
        if (size != 0)
            view.size = (size - 1) / _stride + 1;
        return view;
    }
}
