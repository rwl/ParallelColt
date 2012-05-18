/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;

/**
 * 1-d matrix holding <tt>complex</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperDComplexMatrix1D extends DComplexMatrix1D {
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected DComplexMatrix1D content;

    public WrapperDComplexMatrix1D(DComplexMatrix1D newContent) {
        if (newContent != null)
            setUp((int) newContent.size());
        this.content = newContent;
    }

    protected DComplexMatrix1D getContent() {
        return this.content;
    }

    public synchronized double[] getQuick(int index) {
        return content.getQuick(index);
    }

    public Object elements() {
        return content.elements();
    }

    public DComplexMatrix1D like(int size) {
        return content.like(size);
    }

    public DComplexMatrix2D like2D(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    public DComplexMatrix2D reshape(int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public DComplexMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public synchronized void setQuick(int index, double[] value) {
        content.setQuick(index, value);
    }

    public synchronized void setQuick(int index, double re, double im) {
        content.setQuick(index, re, im);
    }

    public DComplexMatrix1D viewFlip() {
        WrapperDComplexMatrix1D view = new WrapperDComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized double[] getQuick(int index) {
                return content.getQuick(size - 1 - index);
            }

            public synchronized void setQuick(int index, double[] value) {
                content.setQuick(size - 1 - index, value);
            }

            public synchronized void setQuick(int index, double re, double im) {
                content.setQuick(size - 1 - index, re, im);
            }

            public synchronized double[] get(int index) {
                return content.get(size - 1 - index);
            }

            public synchronized void set(int index, double[] value) {
                content.set(size - 1 - index, value);
            }

            public synchronized void set(int index, double re, double im) {
                content.set(size - 1 - index, re, im);
            }
        };
        return view;
    }

    public DComplexMatrix1D viewPart(final int index, int width) {
        checkRange(index, width);
        WrapperDComplexMatrix1D view = new WrapperDComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized double[] getQuick(int i) {
                return content.getQuick(index + i);
            }

            public synchronized void setQuick(int i, double[] value) {
                content.setQuick(index + i, value);
            }

            public synchronized void setQuick(int i, double re, double im) {
                content.setQuick(index + i, re, im);
            }

            public synchronized double[] get(int i) {
                return content.get(index + i);
            }

            public synchronized void set(int i, double[] value) {
                content.set(index + i, value);
            }

            public synchronized void set(int i, double re, double im) {
                content.set(index + i, re, im);
            }
        };
        view.size = width;
        return view;
    }

    public DComplexMatrix1D viewSelection(int[] indexes) {
        // check for "all"
        if (indexes == null) {
            indexes = new int[size];
            for (int i = size; --i >= 0;)
                indexes[i] = i;
        }

        checkIndexes(indexes);
        final int[] idx = indexes;

        WrapperDComplexMatrix1D view = new WrapperDComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized double[] getQuick(int i) {
                return content.getQuick(idx[i]);
            }

            public synchronized void setQuick(int i, double[] value) {
                content.setQuick(idx[i], value);
            }

            public synchronized void setQuick(int i, double re, double im) {
                content.setQuick(idx[i], re, im);
            }

            public synchronized double[] get(int i) {
                return content.get(idx[i]);
            }

            public synchronized void set(int i, double[] value) {
                content.set(idx[i], value);
            }

            public synchronized void set(int i, double re, double im) {
                content.set(idx[i], re, im);
            }
        };
        view.size = indexes.length;
        return view;
    }

    protected DComplexMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

    public DComplexMatrix1D viewStrides(final int _stride) {
        if (stride <= 0)
            throw new IndexOutOfBoundsException("illegal stride: " + stride);
        WrapperDComplexMatrix1D view = new WrapperDComplexMatrix1D(this) {
            private static final long serialVersionUID = 1L;

            public synchronized double[] getQuick(int index) {
                return content.getQuick(index * _stride);
            }

            public synchronized void setQuick(int index, double[] value) {
                content.setQuick(index * _stride, value);
            }

            public synchronized void setQuick(int index, double re, double im) {
                content.setQuick(index * _stride, re, im);
            }

            public synchronized double[] get(int index) {
                return content.get(index * _stride);
            }

            public synchronized void set(int index, double[] value) {
                content.set(index * _stride, value);
            }

            public synchronized void set(int index, double re, double im) {
                content.set(index * _stride, re, im);
            }

        };
        if (size != 0)
            view.size = (size - 1) / _stride + 1;
        return view;
    }

    public DoubleMatrix1D getImaginaryPart() {
        return content.getImaginaryPart();
    }

    public DoubleMatrix1D getRealPart() {
        return content.getRealPart();
    }
}
