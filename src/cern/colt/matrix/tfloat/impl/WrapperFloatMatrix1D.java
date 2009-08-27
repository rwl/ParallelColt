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

    protected FloatMatrix1D getContent() {
        return this.content;
    }

    public synchronized float getQuick(int index) {
        return content.getQuick(index);
    }

    public Object elements() {
        return content.elements();
    }

    public FloatMatrix1D like(int size) {
        return content.like(size);
    }

    public FloatMatrix2D like2D(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    public FloatMatrix2D reshape(int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public FloatMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public synchronized void setQuick(int index, float value) {
        content.setQuick(index, value);
    }

    public FloatMatrix1D viewFlip() {
        FloatMatrix1D view = new WrapperFloatMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int index) {
                return content.getQuick(size - 1 - index);
            }

            public synchronized void setQuick(int index, float value) {
                content.setQuick(size - 1 - index, value);
            }

            public synchronized float get(int index) {
                return content.get(size - 1 - index);
            }

            public synchronized void set(int index, float value) {
                content.set(size - 1 - index, value);
            }
        };
        return view;
    }

    public FloatMatrix1D viewPart(final int index, int width) {
        checkRange(index, width);
        WrapperFloatMatrix1D view = new WrapperFloatMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int i) {
                return content.getQuick(index + i);
            }

            public synchronized void setQuick(int i, float value) {
                content.setQuick(index + i, value);
            }

            public synchronized float get(int i) {
                return content.get(index + i);
            }

            public synchronized void set(int i, float value) {
                content.set(index + i, value);
            }
        };
        view.size = width;
        return view;
    }

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

            public synchronized float getQuick(int i) {
                return content.getQuick(idx[i]);
            }

            public synchronized void setQuick(int i, float value) {
                content.setQuick(idx[i], value);
            }

            public synchronized float get(int i) {
                return content.get(idx[i]);
            }

            public synchronized void set(int i, float value) {
                content.set(idx[i], value);
            }
        };
        view.size = indexes.length;
        return view;
    }

    protected FloatMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

    public FloatMatrix1D viewStrides(final int _stride) {
        if (stride <= 0)
            throw new IndexOutOfBoundsException("illegal stride: " + stride);
        WrapperFloatMatrix1D view = new WrapperFloatMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int index) {
                return content.getQuick(index * _stride);
            }

            public synchronized void setQuick(int index, float value) {
                content.setQuick(index * _stride, value);
            }

            public synchronized float get(int index) {
                return content.get(index * _stride);
            }

            public synchronized void set(int index, float value) {
                content.set(index * _stride, value);
            }
        };
        if (size != 0)
            view.size = (size - 1) / _stride + 1;
        return view;
    }
}
