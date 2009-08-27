/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tobject.impl;

import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tobject.ObjectMatrix1D;
import cern.colt.matrix.tobject.ObjectMatrix2D;
import cern.colt.matrix.tobject.ObjectMatrix3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Dense 1-d matrix (aka <i>vector</i>) holding <tt>Object</tt> elements. First
 * see the <a href="package-summary.html">package summary</a> and javadoc <a
 * href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * <b>Implementation:</b>
 * <p>
 * Internally holds one single contigous one-dimensional array. Note that this
 * implementation is not synchronized.
 * <p>
 * <b>Memory requirements:</b>
 * <p>
 * <tt>memory [bytes] = 8*size()</tt>. Thus, a 1000000 matrix uses 8 MB.
 * <p>
 * <b>Time complexity:</b>
 * <p>
 * <tt>O(1)</tt> (i.e. constant time) for the basic operations <tt>get</tt>,
 * <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 * <p>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
public class DenseObjectMatrix1D extends ObjectMatrix1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /**
     * The elements of this matrix.
     */
    protected Object[] elements;

    /**
     * Constructs a matrix with a copy of the given values. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public DenseObjectMatrix1D(Object[] values) {
        this(values.length);
        assign(values);
    }

    /**
     * Constructs a matrix with a given number of cells. All entries are
     * initially <tt>0</tt>.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    public DenseObjectMatrix1D(int size) {
        setUp(size);
        this.elements = new Object[size];
    }

    /**
     * Constructs a matrix view with the given parameters.
     * 
     * @param size
     *            the number of cells the matrix shall have.
     * @param elements
     *            the cells.
     * @param zero
     *            the index of the first element.
     * @param stride
     *            the number of indexes between any two elements, i.e.
     *            <tt>index(i+1)-index(i)</tt>.
     * @param isView
     *            if true then a matrix view is constructed
     * @throws IllegalArgumentException
     *             if <tt>size<0</tt>.
     */
    protected DenseObjectMatrix1D(int size, Object[] elements, int zero, int stride, boolean isView) {
        setUp(size, zero, stride);
        this.elements = elements;
        this.isNoView = !isView;
    }
    
    public Object aggregate(final cern.colt.function.tobject.ObjectObjectFunction aggr,
            final cern.colt.function.tobject.ObjectFunction f) {
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        Object a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Object>() {
                    public Object call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        Object a = f.apply(elements[idx]);
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            a = aggr.apply(a, f.apply(elements[idx]));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero]);
            int idx = zero;
            for (int i = 1; i < size; i++) {
                idx += stride;
                a = aggr.apply(a, f.apply(elements[idx]));
            }
        }
        return a;
    }

    public Object aggregate(final cern.colt.function.tobject.ObjectObjectFunction aggr,
            final cern.colt.function.tobject.ObjectFunction f, final IntArrayList indexList) {
        if (size() == 0)
            throw new IllegalArgumentException("size == 0");
        final int size = indexList.size();
        final int[] indexElements = indexList.elements();
        Object a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Object>() {

                    public Object call() throws Exception {
                        int idx = zero + indexElements[firstIdx] * stride;
                        Object a = f.apply(elements[idx]);
                        Object elem;
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx = zero + indexElements[i] * stride;
                            elem = elements[idx];
                            a = aggr.apply(a, f.apply(elem));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            Object elem;
            int idx = zero + indexElements[0] * stride;
            a = f.apply(elements[idx]);
            for (int i = 1; i < size; i++) {
                idx = zero + indexElements[i] * stride;
                elem = elements[idx];
                a = aggr.apply(a, f.apply(elem));
            }
        }
        return a;
    }

    public Object aggregate(final ObjectMatrix1D other, final cern.colt.function.tobject.ObjectObjectFunction aggr,
            final cern.colt.function.tobject.ObjectObjectFunction f) {
        if (!(other instanceof DenseObjectMatrix1D)) {
            return super.aggregate(other, aggr, f);
        }
        checkSize(other);
        if (size == 0)
            throw new IllegalArgumentException("size == 0");
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        final Object[] elemsOther = (Object[]) other.elements();
        Object a = null;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Object>() {
                    public Object call() throws Exception {
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        Object a = f.apply(elements[idx], elemsOther[idxOther]);
                        for (int i = firstIdx + 1; i < lastIdx; i++) {
                            idx += stride;
                            idxOther += strideOther;
                            a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
                        }
                        return a;
                    }
                });
            }
            a = ConcurrencyUtils.waitForCompletion(futures, aggr);
        } else {
            a = f.apply(elements[zero], elemsOther[zeroOther]);
            int idx = zero;
            int idxOther = zeroOther;
            for (int i = 1; i < size; i++) {
                idx += stride;
                idxOther += strideOther;
                a = aggr.apply(a, f.apply(elements[idx], elemsOther[idxOther]));
            }
        }
        return a;
    }
    

    /**
     * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt>
     * is required to have the same number of cells as the receiver.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            the values to be filled into the cells.
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>values.length != size()</tt>.
     */

    public ObjectMatrix1D assign(Object[] values) {
        if (isNoView) {
            if (values.length != size)
                throw new IllegalArgumentException("Must have same number of cells: length=" + values.length
                        + "size()=" + size());
            System.arraycopy(values, 0, this.elements, 0, values.length);
        } else {
            super.assign(values);
        }
        return this;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[i] = function(x[i])</tt>. (Iterates downwards from
     * <tt>[size()-1]</tt> to <tt>[0]</tt>).
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * 	 // change each cell to its sine
     * 	 matrix =   0.5      1.5      2.5       3.5 
     * 	 matrix.assign(cern.jet.math.Functions.sin);
     * 	 --&gt;
     * 	 matrix ==  0.479426 0.997495 0.598472 -0.350783
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param function
     *            a function object taking as argument the current cell's value.
     * @return <tt>this</tt> (for convenience only).
     * @see cern.jet.math.tdouble.DoubleFunctions
     */

    public ObjectMatrix1D assign(final cern.colt.function.tobject.ObjectFunction function) {
        if (elements == null)
            throw new InternalError();

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        // the general case x[i] = f(x[i])
                        int idx = zero + firstIdx * stride;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            elements[idx] = function.apply(elements[idx]);
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            // the general case x[i] = f(x[i])
            int idx = zero;
            for (int k = 0; k < size; k++) {
                elements[idx] = function.apply(elements[idx]);
                idx += stride;
            }
        }
        return this;
    }

    /**
     * Replaces all cell values of the receiver with the values of another
     * matrix. Both matrices must have the same size. If both matrices share the
     * same cells (as is the case if they are views derived from the same
     * matrix) and intersect in an ambiguous way, then replaces <i>as if</i>
     * using an intermediate auxiliary deep copy of <tt>other</tt>.
     * 
     * @param source
     *            the source matrix to copy from (may be identical to the
     *            receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */

    public ObjectMatrix1D assign(ObjectMatrix1D source) {
        // overriden for performance only
        if (!(source instanceof DenseObjectMatrix1D)) {
            return super.assign(source);
        }
        DenseObjectMatrix1D other = (DenseObjectMatrix1D) source;
        if (other == this)
            return this;
        checkSize(other);
        if (isNoView && other.isNoView) { // quickest
            System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
            return this;
        }
        if (haveSharedCells(other)) {
            ObjectMatrix1D c = other.copy();
            if (!(c instanceof DenseObjectMatrix1D)) { // should not happen
                return super.assign(source);
            }
            other = (DenseObjectMatrix1D) c;
        }

        final Object[] elemsOther = other.elements;
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            elements[idx] = elemsOther[idxOther];
                            idx += stride;
                            idxOther += strideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int k = 0; k < size; k++) {
                elements[idx] = elemsOther[idxOther];
                idx += stride;
                idxOther += strideOther;
            }
        }
        return this;
    }

    /**
     * Assigns the result of a function to each cell;
     * <tt>x[i] = function(x[i],y[i])</tt>. (Iterates downwards from
     * <tt>[size()-1]</tt> to <tt>[0]</tt>).
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * 	 // assign x[i] = x[i]&lt;sup&gt;y[i]&lt;/sup&gt;
     * 	 m1 = 0 1 2 3;
     * 	 m2 = 0 2 4 6;
     * 	 m1.assign(m2, cern.jet.math.Functions.pow);
     * 	 --&gt;
     * 	 m1 == 1 1 16 729
     * 
     * 	 // for non-standard functions there is no shortcut: 
     * 	 m1.assign(m2,
     * 	    new ObjectobjectFunction() {
     * 	       public Object apply(Object x, Object y) { return Math.pow(x,y); }
     * 	    }
     * 	 );
     * 
     * </pre>
     * 
     * For further examples, see the <a
     * href="package-summary.html#FunctionObjects">package doc</a>.
     * 
     * @param y
     *            the secondary matrix to operate on.
     * @param function
     *            a function object taking as first argument the current cell's
     *            value of <tt>this</tt>, and as second argument the current
     *            cell's value of <tt>y</tt>,
     * @return <tt>this</tt> (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>size() != y.size()</tt>.
     * @see cern.jet.math.tdouble.DoubleFunctions
     */

    public ObjectMatrix1D assign(final ObjectMatrix1D y, final cern.colt.function.tobject.ObjectObjectFunction function) {
        // overriden for performance only
        if (!(y instanceof DenseObjectMatrix1D)) {
            return super.assign(y, function);
        }
        final int zeroOther = (int) y.index(0);
        final int strideOther = y.stride();
        final Object[] elemsOther = (Object[]) y.elements();
        if (elements == null || elemsOther == null)
            throw new InternalError();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        int idxOther;
                        // the general case x[i] = f(x[i],y[i])
                        idx = zero + firstIdx * stride;
                        idxOther = zeroOther + firstIdx * strideOther;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
                            idx += stride;
                            idxOther += strideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            // specialized for speed
            int idx;
            int idxOther;
            // the general case x[i] = f(x[i],y[i])
            idx = zero;
            idxOther = zeroOther;
            for (int k = 0; k < size; k++) {
                elements[idx] = function.apply(elements[idx], elemsOther[idxOther]);
                idx += stride;
                idxOther += strideOther;
            }
        }
        return this;
    }
    
    public ObjectMatrix1D assign(final Object value) {
        final Object[] elems = this.elements;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstIdx * stride;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            elems[idx] = value;
                            idx += stride;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int i = 0; i < size; i++) {
                elems[idx] = value;
                idx += stride;
            }
        }
        return this;
    }

    public Object elements() {
        return elements;
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

    public Object getQuick(int index) {
        // if (debug) if (index<0 || index>=size) checkIndex(index);
        // return elements[index(index)];
        // manually inlined:
        return elements[zero + index * stride];
    }

    /**
     * Returns <tt>true</tt> if both matrices share at least one identical cell.
     */

    protected boolean haveSharedCellsRaw(ObjectMatrix1D other) {
        if (other instanceof SelectedDenseObjectMatrix1D) {
            SelectedDenseObjectMatrix1D otherMatrix = (SelectedDenseObjectMatrix1D) other;
            return this.elements == otherMatrix.elements;
        } else if (other instanceof DenseObjectMatrix1D) {
            DenseObjectMatrix1D otherMatrix = (DenseObjectMatrix1D) other;
            return this.elements == otherMatrix.elements;
        }
        return false;
    }

    /**
     * Returns the position of the element with the given relative rank within
     * the (virtual or non-virtual) internal 1-dimensional array. You may want
     * to override this method for performance.
     * 
     * @param rank
     *            the rank of the element.
     */

    public long index(int rank) {
        // overriden for manual inlining only
        // return _offset(_rank(rank));
        return zero + rank * stride;
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified size. For example, if the receiver
     * is an instance of type <tt>DenseObjectMatrix1D</tt> the new matrix must
     * also be of type <tt>DenseObjectMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseObjectMatrix1D</tt> the new matrix must also
     * be of type <tt>SparseObjectMatrix1D</tt>, etc. In general, the new matrix
     * should have internal parametrization as similar as possible.
     * 
     * @param size
     *            the number of cell the matrix shall have.
     * @return a new empty matrix of the same dynamic type.
     */

    public ObjectMatrix1D like(int size) {
        return new DenseObjectMatrix1D(size);
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver. For example, if the
     * receiver is an instance of type <tt>DenseObjectMatrix1D</tt> the new
     * matrix must be of type <tt>DenseObjectMatrix2D</tt>, if the receiver is
     * an instance of type <tt>SparseObjectMatrix1D</tt> the new matrix must be
     * of type <tt>SparseObjectMatrix2D</tt>, etc.
     * 
     * @param rows
     *            the number of rows the matrix shall have.
     * @param columns
     *            the number of columns the matrix shall have.
     * @return a new matrix of the corresponding dynamic type.
     */

    public ObjectMatrix2D like2D(int rows, int columns) {
        return new DenseObjectMatrix2D(rows, columns);
    }
    
    public ObjectMatrix2D reshape(final int rows, final int columns) {
        if (rows * columns != size) {
            throw new IllegalArgumentException("rows*columns != size");
        }
        ObjectMatrix2D M = new DenseObjectMatrix2D(rows, columns);
        final Object[] elemsOther = (Object[]) M.elements();
        final int zeroOther = (int) M.index(0, 0);
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        int idxOther;
                        for (int c = firstColumn; c < lastColumn; c++) {
                            idxOther = zeroOther + c * colStrideOther;
                            idx = zero + (c * rows) * stride;
                            for (int r = 0; r < rows; r++) {
                                elemsOther[idxOther] = elements[idx];
                                idxOther += rowStrideOther;
                                idx += stride;
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxOther;
            int idx = zero;
            for (int c = 0; c < columns; c++) {
                idxOther = zeroOther + c * colStrideOther;
                for (int r = 0; r < rows; r++) {
                    elemsOther[idxOther] = elements[idx];
                    idxOther += rowStrideOther;
                    idx += stride;
                }
            }
        }
        return M;
    }

    public ObjectMatrix3D reshape(final int slices, final int rows, final int columns) {
        if (slices * rows * columns != size) {
            throw new IllegalArgumentException("slices*rows*columns != size");
        }
        ObjectMatrix3D M = new DenseObjectMatrix3D(slices, rows, columns);
        final Object[] elemsOther = (Object[]) M.elements();
        final int zeroOther = (int) M.index(0, 0, 0);
        final int sliceStrideOther = M.sliceStride();
        final int rowStrideOther = M.rowStride();
        final int colStrideOther = M.columnStride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstSlice = j * k;
                final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx;
                        int idxOther;
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
                                idx = zero + (s * rows * columns + c * rows) * stride;
                                for (int r = 0; r < rows; r++) {
                                    elemsOther[idxOther] = elements[idx];
                                    idxOther += rowStrideOther;
                                    idx += stride;
                                }
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idxOther;
            int idx = zero;
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    idxOther = zeroOther + s * sliceStrideOther + c * colStrideOther;
                    for (int r = 0; r < rows; r++) {
                        elemsOther[idxOther] = elements[idx];
                        idxOther += rowStrideOther;
                        idx += stride;
                    }
                }
            }
        }
        return M;
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

    public void setQuick(int index, Object value) {
        // if (debug) if (index<0 || index>=size) checkIndex(index);
        // elements[index(index)] = value;
        // manually inlined:
        elements[zero + index * stride] = value;
    }

    /**
     * Swaps each element <tt>this[i]</tt> with <tt>other[i]</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>size() != other.size()</tt>.
     */

    public void swap(ObjectMatrix1D other) {
        // overriden for performance only
        if (!(other instanceof DenseObjectMatrix1D)) {
            super.swap(other);
        }
        DenseObjectMatrix1D y = (DenseObjectMatrix1D) other;
        if (y == this)
            return;
        checkSize(y);
        final Object[] elemsOther = (Object[]) y.elements();
        if (elements == null || elemsOther == null)
            throw new InternalError();
        final int zeroOther = (int) other.index(0);
        final int strideOther = other.stride();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;

                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = zero + firstIdx * stride;
                        int idxOther = zeroOther + firstIdx * strideOther;
                        for (int k = firstIdx; k < lastIdx; k++) {
                            Object tmp = elements[idx];
                            elements[idx] = elemsOther[idxOther];
                            elemsOther[idxOther] = tmp;
                            idx += stride;
                            idxOther += strideOther;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            int idxOther = zeroOther;
            for (int k = 0; k < size; k++) {
                Object tmp = elements[idx];
                elements[idx] = elemsOther[idxOther];
                elemsOther[idxOther] = tmp;
                idx += stride;
                idxOther += strideOther;
            }
        }
    }

    /**
     * Fills the cell values into the specified 1-dimensional array. The values
     * are copied. So subsequent changes in <tt>values</tt> are not reflected in
     * the matrix, and vice-versa. After this call returns the array
     * <tt>values</tt> has the form <br>
     * <tt>for (int i=0; i < size(); i++) values[i] = get(i);</tt>
     * 
     * @throws IllegalArgumentException
     *             if <tt>values.length < size()</tt>.
     */

    public void toArray(Object[] values) {
        if (values.length < size)
            throw new IllegalArgumentException("values too small");
        if (this.isNoView)
            System.arraycopy(this.elements, 0, values, 0, this.elements.length);
        else
            super.toArray(values);
    }

    /**
     * Construct and returns a new selection view.
     * 
     * @param offsets
     *            the offsets of the visible elements.
     * @return a new view.
     */

    protected ObjectMatrix1D viewSelectionLike(int[] offsets) {
        return new SelectedDenseObjectMatrix1D(this.elements, offsets);
    }
}
