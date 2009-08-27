/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import cern.colt.matrix.tobject.ObjectMatrix1D;
import cern.colt.matrix.tobject.ObjectMatrix2D;
import cern.colt.matrix.tobject.ObjectMatrix3D;

/**
 * 1-d matrix holding <tt>int</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperObjectMatrix1D extends ObjectMatrix1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected ObjectMatrix1D content;

    public WrapperObjectMatrix1D(ObjectMatrix1D newContent) {
        if (newContent != null)
            setUp((int) newContent.size());
        this.content = newContent;
    }

    /**
     * Returns the content of this matrix if it is a wrapper; or <tt>this</tt>
     * otherwise. Override this method in wrappers.
     */

    protected ObjectMatrix1D getContent() {
        return this.content;
    }

    /**
     * Returns the matrix cell value at coordinate <tt>index</tt>.
     * 
     * <p>
     * Provided with invalid parameters this method may return invalid objects
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @return the value of the specified cell.
     */

    public synchronized Object getQuick(int index) {
        return content.getQuick(index);
    }

    public Object elements() {
        return content.elements();
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified size. For example, if the receiver
     * is an instance of type <tt>DenseObjectMatrix1D</tt> the new matrix must
     * also be of type <tt>DenseObjectMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseObjectMatrix1D</tt> the new matrix must also be
     * of type <tt>SparseObjectMatrix1D</tt>, etc. In general, the new matrix
     * should have internal parametrization as similar as possible.
     * 
     * @param size
     *            the number of cell the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */

    public ObjectMatrix1D like(int size) {
        return content.like(size);
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseObjectMatrix1D</tt> the new matrix
     * must be of type <tt>DenseObjectMatrix2D</tt>, if the receiver is an
     * instance of type <tt>SparseObjectMatrix1D</tt> the new matrix must be of
     * type <tt>SparseObjectMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */

    public ObjectMatrix2D like2D(int rows, int columns) {
        return content.like2D(rows, columns);
    }

    public ObjectMatrix2D reshape(int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    public ObjectMatrix3D reshape(int slices, int rows, int columns) {
        throw new IllegalArgumentException("This method is not supported.");
    }

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
     * 
     * <p>
     * Provided with invalid parameters this method may access illegal indexes
     * without throwing any exception. <b>You should only use this method when
     * you are absolutely sure that the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     * 
     * @param index
     *            the index of the cell.
     * @param value
     *            the value to be filled into the specified cell.
     */

    public synchronized void setQuick(int index, Object value) {
        content.setQuick(index, value);
    }

    /**
     * Constructs and returns a new <i>flip view</i>. What used to be index
     * <tt>0</tt> is now index <tt>size()-1</tt>, ..., what used to be index
     * <tt>size()-1</tt> is now index <tt>0</tt>. The returned view is backed by
     * this matrix, so changes in the returned view are reflected in this
     * matrix, and vice-versa.
     * 
     * @return a new flip view.
     */

    public ObjectMatrix1D viewFlip() {
        ObjectMatrix1D view = new WrapperObjectMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized Object getQuick(int index) {
                return content.getQuick(size - 1 - index);
            }

            public synchronized void setQuick(int index, Object value) {
                content.setQuick(size - 1 - index, value);
            }

            public synchronized Object get(int index) {
                return content.get(size - 1 - index);
            }

            public synchronized void set(int index, Object value) {
                content.set(size - 1 - index, value);
            }
        };
        return view;
    }

    /**
     * Constructs and returns a new <i>sub-range view</i> that is a
     * <tt>width</tt> sub matrix starting at <tt>index</tt>.
     * 
     * Operations on the returned view can only be applied to the restricted
     * range. Any attempt to access coordinates not contained in the view will
     * throw an <tt>IndexOutOfBoundsException</tt>.
     * <p>
     * <b>Note that the view is really just a range restriction:</b> The
     * returned matrix is backed by this matrix, so changes in the returned
     * matrix are reflected in this matrix, and vice-versa.
     * <p>
     * The view contains the cells from <tt>index..index+width-1</tt>. and has
     * <tt>view.size() == width</tt>. A view's legal coordinates are again zero
     * based, as usual. In other words, legal coordinates of the view are
     * <tt>0 .. view.size()-1==width-1</tt>. As usual, any attempt to access a
     * cell at other coordinates will throw an
     * <tt>IndexOutOfBoundsException</tt>.
     * 
     * @param index
     *            The index of the first cell.
     * @param width
     *            The width of the range.
     * @throws IndexOutOfBoundsException
     *             if <tt>index<0 || width<0 || index+width>size()</tt>.
     * @return the new view.
     * 
     */

    public ObjectMatrix1D viewPart(final int index, int width) {
        checkRange(index, width);
        ObjectMatrix1D view = new WrapperObjectMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized Object getQuick(int i) {
                return content.getQuick(index + i);
            }

            public synchronized void setQuick(int i, Object value) {
                content.setQuick(index + i, value);
            }

            public synchronized Object get(int i) {
                return content.get(index + i);
            }

            public synchronized void set(int i, Object value) {
                content.set(index + i, value);
            }
        };
        view.setSize(width);
        return view;
    }

    /**
     * Constructs and returns a new <i>selection view</i> that is a matrix
     * holding the indicated cells. There holds
     * <tt>view.size() == indexes.length</tt> and
     * <tt>view.get(i) == this.get(indexes[i])</tt>. Indexes can occur multiple
     * times and can be in arbitrary order.
     * <p>
     * <b>Example:</b> <br>
     * 
     * <pre>
     *   this     = (0,0,8,0,7)
     *   indexes  = (0,2,4,2)
     *   --&gt;
     *   view     = (0,8,7,8)
     * 
     * </pre>
     * 
     * Note that modifying <tt>indexes</tt> after this call has returned has no
     * effect on the view. The returned view is backed by this matrix, so
     * changes in the returned view are reflected in this matrix, and
     * vice-versa.
     * 
     * @param indexes
     *            The indexes of the cells that shall be visible in the new
     *            view. To indicate that <i>all</i> cells shall be visible,
     *            simply set this parameter to <tt>null</tt>.
     * @return the new view.
     * @throws IndexOutOfBoundsException
     *             if <tt>!(0 <= indexes[i] < size())</tt> for any
     *             <tt>i=0..indexes.length()-1</tt>.
     */

    public ObjectMatrix1D viewSelection(int[] indexes) {
        // check for "all"
        if (indexes == null) {
            indexes = new int[size];
            for (int i = size; --i >= 0;)
                indexes[i] = i;
        }

        checkIndexes(indexes);
        final int[] idx = indexes;

        ObjectMatrix1D view = new WrapperObjectMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized Object getQuick(int i) {
                return content.getQuick(idx[i]);
            }

            public synchronized void setQuick(int i, Object value) {
                content.setQuick(idx[i], value);
            }

            public synchronized Object get(int i) {
                return content.get(idx[i]);
            }

            public synchronized void set(int i, Object value) {
                content.set(idx[i], value);
            }
        };
        view.setSize(indexes.length);
        return view;
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param offsets
     *            the offsets of the visible elements.
     * @return a new view.
     */

    protected ObjectMatrix1D viewSelectionLike(int[] offsets) {
        throw new InternalError(); // should never get called
    }

    /**
     * Constructs and returns a new <i>stride view</i> which is a sub matrix
     * consisting of every i-th cell. More specifically, the view has size
     * <tt>this.size()/stride</tt> holding cells <tt>this.get(i*stride)</tt> for
     * all <tt>i = 0..size()/stride - 1</tt>.
     * 
     * @param _stride
     *            the step factor.
     * @throws IndexOutOfBoundsException
     *             if <tt>stride <= 0</tt>.
     * @return the new view.
     * 
     */

    public ObjectMatrix1D viewStrides(final int _stride) {
        if (stride <= 0)
            throw new IndexOutOfBoundsException("illegal stride: " + stride);
        ObjectMatrix1D view = new WrapperObjectMatrix1D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized Object getQuick(int index) {
                return content.getQuick(index * _stride);
            }

            public synchronized void setQuick(int index, Object value) {
                content.setQuick(index * _stride, value);
            }

            public synchronized Object get(int index) {
                return content.get(index * _stride);
            }

            public synchronized void set(int index, Object value) {
                content.set(index * _stride, value);
            }
        };
        view.setSize(size);
        if (size != 0)
            view.setSize((size - 1) / _stride + 1);
        return view;
    }
}
