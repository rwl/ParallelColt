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
 * matrix or a matrix whose views are wrappers.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperFComplexMatrix1D extends FComplexMatrix1D {
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected FComplexMatrix1D content;

    public WrapperFComplexMatrix1D(FComplexMatrix1D newContent) {
        if (newContent != null)
            setUp((int) newContent.size());
        this.content = newContent;
    }

    @Override
    protected FComplexMatrix1D getContent() {
        return this.content;
    }

    @Override
    public float[] getQuick(int index) {
        return content.getQuick(index);
    }

    @Override
    public Object elements() {
        return content.elements();
    }

    @Override
    public FComplexMatrix1D like(int size) {
        return content.like(size);
    }

    @Override
    public FComplexMatrix2D like2D(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    @Override
    public FComplexMatrix2D reshape(int rows, int cols) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    @Override
    public FComplexMatrix3D reshape(int slices, int rows, int cols) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    @Override
    public void setQuick(int index, float[] value) {
        content.setQuick(index, value);
    }

    @Override
    public void setQuick(int index, float re, float im) {
        content.setQuick(index, re, im);
    }

    @Override
    public FComplexMatrix1D viewFlip() {
        WrapperFComplexMatrix1D view = new WrapperFComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int index) {
                return content.getQuick(size - 1 - index);
            }

            @Override
            public void setQuick(int index, float[] value) {
                content.setQuick(size - 1 - index, value);
            }

            @Override
            public void setQuick(int index, float re, float im) {
                content.setQuick(size - 1 - index, re, im);
            }

            @Override
            public float[] get(int index) {
                return content.get(size - 1 - index);
            }

            @Override
            public void set(int index, float[] value) {
                content.set(size - 1 - index, value);
            }

            @Override
            public void set(int index, float re, float im) {
                content.set(size - 1 - index, re, im);
            }
        };
        return view;
    }

    @Override
    public FComplexMatrix1D viewPart(final int index, int width) {
        checkRange(index, width);
        WrapperFComplexMatrix1D view = new WrapperFComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int i) {
                return content.getQuick(index + i);
            }

            @Override
            public void setQuick(int i, float[] value) {
                content.setQuick(index + i, value);
            }

            @Override
            public void setQuick(int i, float re, float im) {
                content.setQuick(index + i, re, im);
            }

            @Override
            public float[] get(int i) {
                return content.get(index + i);
            }

            @Override
            public void set(int i, float[] value) {
                content.set(index + i, value);
            }

            @Override
            public void set(int i, float re, float im) {
                content.set(index + i, re, im);
            }
        };
        view.size = width;
        return view;
    }

    @Override
    public FComplexMatrix1D viewSelection(int[] indexes) {
        // check for "all"
        if (indexes == null) {
            indexes = new int[size];
            for (int i = size; --i >= 0;)
                indexes[i] = i;
        }

        checkIndexes(indexes);
        final int[] idx = indexes;

        WrapperFComplexMatrix1D view = new WrapperFComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int i) {
                return content.getQuick(idx[i]);
            }

            @Override
            public void setQuick(int i, float[] value) {
                content.setQuick(idx[i], value);
            }

            @Override
            public void setQuick(int i, float re, float im) {
                content.setQuick(idx[i], re, im);
            }

            @Override
            public float[] get(int i) {
                return content.get(idx[i]);
            }

            @Override
            public void set(int i, float[] value) {
                content.set(idx[i], value);
            }

            @Override
            public void set(int i, float re, float im) {
                content.set(idx[i], re, im);
            }
        };
        view.size = indexes.length;
        return view;
    }

    @Override
    protected FComplexMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

    @Override
    public FComplexMatrix1D viewStrides(final int _stride) {
        if (stride <= 0)
            throw new IndexOutOfBoundsException("illegal stride: " + stride);
        WrapperFComplexMatrix1D view = new WrapperFComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            @Override
            public float[] getQuick(int index) {
                return content.getQuick(index * _stride);
            }

            @Override
            public void setQuick(int index, float[] value) {
                content.setQuick(index * _stride, value);
            }

            @Override
            public void setQuick(int index, float re, float im) {
                content.setQuick(index * _stride, re, im);
            }

            @Override
            public float[] get(int index) {
                return content.get(index * _stride);
            }

            @Override
            public void set(int index, float[] value) {
                content.set(index * _stride, value);
            }

            @Override
            public void set(int index, float re, float im) {
                content.set(index * _stride, re, im);
            }

        };
        if (size != 0)
            view.size = (size - 1) / _stride + 1;
        return view;
    }

    @Override
    public FloatMatrix1D getImaginaryPart() {
        return content.getImaginaryPart();
    }

    @Override
    public FloatMatrix1D getRealPart() {
        return content.getRealPart();
    }
}
