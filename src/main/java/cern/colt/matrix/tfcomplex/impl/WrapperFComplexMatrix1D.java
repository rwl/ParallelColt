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

    protected FComplexMatrix1D getContent() {
        return this.content;
    }

    public synchronized float[] getQuick(int index) {
        return content.getQuick(index);
    }

    public Object elements() {
        return content.elements();
    }

    public FComplexMatrix1D like(int size) {
        return content.like(size);
    }

    public FComplexMatrix2D like2D(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    public FComplexMatrix2D reshape(int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public FComplexMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public synchronized void setQuick(int index, float[] value) {
        content.setQuick(index, value);
    }

    public synchronized void setQuick(int index, float re, float im) {
        content.setQuick(index, re, im);
    }

    public FComplexMatrix1D viewFlip() {
        WrapperFComplexMatrix1D view = new WrapperFComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int index) {
                return content.getQuick(size - 1 - index);
            }

            public synchronized void setQuick(int index, float[] value) {
                content.setQuick(size - 1 - index, value);
            }

            public synchronized void setQuick(int index, float re, float im) {
                content.setQuick(size - 1 - index, re, im);
            }

            public synchronized float[] get(int index) {
                return content.get(size - 1 - index);
            }

            public synchronized void set(int index, float[] value) {
                content.set(size - 1 - index, value);
            }

            public synchronized void set(int index, float re, float im) {
                content.set(size - 1 - index, re, im);
            }
        };
        return view;
    }

    public FComplexMatrix1D viewPart(final int index, int width) {
        checkRange(index, width);
        WrapperFComplexMatrix1D view = new WrapperFComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int i) {
                return content.getQuick(index + i);
            }

            public synchronized void setQuick(int i, float[] value) {
                content.setQuick(index + i, value);
            }

            public synchronized void setQuick(int i, float re, float im) {
                content.setQuick(index + i, re, im);
            }

            public synchronized float[] get(int i) {
                return content.get(index + i);
            }

            public synchronized void set(int i, float[] value) {
                content.set(index + i, value);
            }

            public synchronized void set(int i, float re, float im) {
                content.set(index + i, re, im);
            }
        };
        view.size = width;
        return view;
    }

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

            public synchronized float[] getQuick(int i) {
                return content.getQuick(idx[i]);
            }

            public synchronized void setQuick(int i, float[] value) {
                content.setQuick(idx[i], value);
            }

            public synchronized void setQuick(int i, float re, float im) {
                content.setQuick(idx[i], re, im);
            }

            public synchronized float[] get(int i) {
                return content.get(idx[i]);
            }

            public synchronized void set(int i, float[] value) {
                content.set(idx[i], value);
            }

            public synchronized void set(int i, float re, float im) {
                content.set(idx[i], re, im);
            }
        };
        view.size = indexes.length;
        return view;
    }

    protected FComplexMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

    public FComplexMatrix1D viewStrides(final int _stride) {
        if (stride <= 0)
            throw new IndexOutOfBoundsException("illegal stride: " + stride);
        WrapperFComplexMatrix1D view = new WrapperFComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized float[] getQuick(int index) {
                return content.getQuick(index * _stride);
            }

            public synchronized void setQuick(int index, float[] value) {
                content.setQuick(index * _stride, value);
            }

            public synchronized void setQuick(int index, float re, float im) {
                content.setQuick(index * _stride, re, im);
            }

            public synchronized float[] get(int index) {
                return content.get(index * _stride);
            }

            public synchronized void set(int index, float[] value) {
                content.set(index * _stride, value);
            }

            public synchronized void set(int index, float re, float im) {
                content.set(index * _stride, re, im);
            }

        };
        if (size != 0)
            view.size = (size - 1) / _stride + 1;
        return view;
    }

    public FloatMatrix1D getImaginaryPart() {
        return content.getImaginaryPart();
    }

    public FloatMatrix1D getRealPart() {
        return content.getRealPart();
    }
}
