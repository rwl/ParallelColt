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
 * 1-d matrix holding <tt>float</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperFloatMatrix1D extends FloatMatrix1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected FloatMatrix1D content;

    public WrapperFloatMatrix1D(FloatMatrix1D newContent) {
        if (newContent != null)
            setUp((int) newContent.size());
        this.content = newContent;
    }

    @Override
    protected FloatMatrix1D getContent() {
        return this.content;
    }

    @Override
    public float getQuick(int index) {
        return content.getQuick(index);
    }

    @Override
    public Object elements() {
        return content.elements();
    }

    @Override
    public FloatMatrix1D like(int size) {
        return content.like(size);
    }

    @Override
    public FloatMatrix2D like2D(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    @Override
    public FloatMatrix2D reshape(int rows, int cols) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    @Override
    public FloatMatrix3D reshape(int slices, int rows, int cols) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    @Override
    public void setQuick(int index, float value) {
        content.setQuick(index, value);
    }

    @Override
    public FloatMatrix1D viewFlip() {
        FloatMatrix1D view = new WrapperFloatMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public float getQuick(int index) {
                return content.getQuick(size - 1 - index);
            }

            @Override
            public void setQuick(int index, float value) {
                content.setQuick(size - 1 - index, value);
            }

            @Override
            public float get(int index) {
                return content.get(size - 1 - index);
            }

            @Override
            public void set(int index, float value) {
                content.set(size - 1 - index, value);
            }
        };
        return view;
    }

    @Override
    public FloatMatrix1D viewPart(final int index, int width) {
        checkRange(index, width);
        WrapperFloatMatrix1D view = new WrapperFloatMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public float getQuick(int i) {
                return content.getQuick(index + i);
            }

            @Override
            public void setQuick(int i, float value) {
                content.setQuick(index + i, value);
            }

            @Override
            public float get(int i) {
                return content.get(index + i);
            }

            @Override
            public void set(int i, float value) {
                content.set(index + i, value);
            }
        };
        view.size = width;
        return view;
    }

    @Override
    public FloatMatrix1D viewSelection(int[] indexes) {
        // check for "all"
        if (indexes == null) {
            indexes = new int[size];
            for (int i = size; --i >= 0;)
                indexes[i] = i;
        }

        checkIndexes(indexes);
        final int[] idx = indexes;

        WrapperFloatMatrix1D view = new WrapperFloatMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public float getQuick(int i) {
                return content.getQuick(idx[i]);
            }

            @Override
            public void setQuick(int i, float value) {
                content.setQuick(idx[i], value);
            }

            @Override
            public float get(int i) {
                return content.get(idx[i]);
            }

            @Override
            public void set(int i, float value) {
                content.set(idx[i], value);
            }
        };
        view.size = indexes.length;
        return view;
    }

    @Override
    protected FloatMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

    @Override
    public FloatMatrix1D viewStrides(final int _stride) {
        if (stride <= 0)
            throw new IndexOutOfBoundsException("illegal stride: " + stride);
        WrapperFloatMatrix1D view = new WrapperFloatMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            @Override
            public float getQuick(int index) {
                return content.getQuick(index * _stride);
            }

            @Override
            public void setQuick(int index, float value) {
                content.setQuick(index * _stride, value);
            }

            @Override
            public float get(int index) {
                return content.get(index * _stride);
            }

            @Override
            public void set(int index, float value) {
                content.set(index * _stride, value);
            }
        };
        if (size != 0)
            view.size = (size - 1) / _stride + 1;
        return view;
    }
}
