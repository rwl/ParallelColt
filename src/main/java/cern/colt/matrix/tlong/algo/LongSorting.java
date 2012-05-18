/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong.algo;

import cern.colt.function.tint.IntComparator;
import cern.colt.matrix.AbstractFormatter;
import cern.colt.matrix.tlong.LongMatrix1D;
import cern.colt.matrix.tlong.LongMatrix2D;
import cern.colt.matrix.tlong.LongMatrix3D;
import cern.colt.matrix.tlong.impl.DenseLongMatrix1D;

/**
 * Matrix quicksorts and mergesorts. Use idioms like
 * <tt>Sorting.quickSort.sort(...)</tt> and <tt>Sorting.mergeSort.sort(...)</tt>
 * .
 * <p>
 * This is another case demonstrating one primary goal of this library:
 * Delivering easy to use, yet very efficient APIs. The sorts return convenient
 * <i>sort views</i>. This enables the usage of algorithms which scale well with
 * the problem size: For example, sorting a 1000000 x 10000 or a 1000000 x 100 x
 * 100 matrix performs just as fast as sorting a 1000000 x 1 matrix. This is so,
 * because internally the algorithms only move around integer indexes, they do
 * not physically move around entire rows or slices. The original matrix is left
 * unaffected.
 * <p>
 * The quicksort is a derivative of the JDK 1.2 V1.26 algorithms (which are, in
 * turn, based on Bentley's and McIlroy's fine work). The mergesort is a
 * derivative of the JAL algorithms, with optimisations taken from the JDK
 * algorithms. Mergesort is <i>stable</i> (by definition), while quicksort is
 * not. A stable sort is, for example, helpful, if matrices are sorted
 * successively by multiple columns. It preserves the relative position of equal
 * elements.
 * 
 * @see cern.colt.GenericSorting
 * @see cern.colt.Sorting
 * @see java.util.Arrays
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.1, 25/May/2000
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class LongSorting extends cern.colt.PersistentObject {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /**
     * A prefabricated quicksort.
     */
    public static final LongSorting quickSort = new LongSorting();

    /**
     * A prefabricated mergesort.
     */
    public static final LongSorting mergeSort = new LongSorting() {
        /**
         * 
         */
        private static final long serialVersionUID = 1L;

        protected void runSort(int[] a, int fromIndex, int toIndex, IntComparator c) {
            cern.colt.Sorting.mergeSort(a, fromIndex, toIndex, c);
        }

        protected void runSort(int fromIndex, int toIndex, IntComparator c, cern.colt.Swapper swapper) {
            cern.colt.GenericSorting.mergeSort(fromIndex, toIndex, c, swapper);
        }
    };

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected LongSorting() {
    }

    protected void runSort(int[] a, int fromIndex, int toIndex, IntComparator c) {
        cern.colt.Sorting.parallelQuickSort(a, fromIndex, toIndex, c);
    }

    protected void runSort(int fromIndex, int toIndex, IntComparator c, cern.colt.Swapper swapper) {
        cern.colt.GenericSorting.quickSort(fromIndex, toIndex, c, swapper);
    }

    /**
     * Sorts the vector into ascending order, according to the <i>natural
     * ordering</i>. The returned view is backed by this matrix, so changes in
     * the returned view are reflected in this matrix, and vice-versa. To sort
     * ranges use sub-ranging views. To sort descending, use flip views ...
     * <p>
     * <b>Example:</b>
     * <table border="1" cellspacing="0">
     * <tr nowrap>
     * <td valign="top"><tt> 7, 1, 3, 1<br>
     </tt></td>
     * <td valign="top">
     * <p>
     * <tt> ==&gt; 1, 1, 3, 7<br>
     The vector IS NOT SORTED.<br>
     The new VIEW IS SORTED.</tt>
     * </p>
     * </td>
     * </tr>
     * </table>
     * 
     * @param vector
     *            the vector to be sorted.
     * @return a new sorted vector (matrix) view. <b>Note that the original
     *         matrix is left unaffected.</b>
     */
    public LongMatrix1D sort(final LongMatrix1D vector) {
        return vector.viewSelection(sortIndex(vector));
    }

    /**
     * Sorts indexes of the <code>vector</code> into ascending order.
     * 
     * @param vector
     * @return sorted indexes
     */
    public int[] sortIndex(final LongMatrix1D vector) {
        int[] indexes = new int[(int) vector.size()]; // row indexes to reorder
        // instead of matrix itself
        for (int i = indexes.length; --i >= 0;)
            indexes[i] = i;
        IntComparator comp = null;
        if (vector instanceof DenseLongMatrix1D) {
            final long[] velems = (long[]) vector.elements();
            final int zero = (int) vector.index(0);
            final int stride = vector.stride();
            comp = new IntComparator() {
                public int compare(int a, int b) {
                    int idxa = zero + a * stride;
                    int idxb = zero + b * stride;
                    long av = velems[idxa];
                    long bv = velems[idxb];
                    return av < bv ? -1 : (av == bv ? 0 : 1);
                }
            };
        } else {
            comp = new IntComparator() {
                public int compare(int a, int b) {
                    long av = vector.getQuick(a);
                    long bv = vector.getQuick(b);
                    return av < bv ? -1 : (av == bv ? 0 : 1);
                }
            };
        }

        runSort(indexes, 0, indexes.length, comp);

        return indexes;
    }

    /**
     * Sorts the vector into ascending order, according to the order induced by
     * the specified comparator. The returned view is backed by this matrix, so
     * changes in the returned view are reflected in this matrix, and
     * vice-versa. The algorithm compares two cells at a time, determinining
     * whether one is smaller, equal or larger than the other. To sort ranges
     * use sub-ranging views. To sort descending, use flip views ...
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * // sort by sinus of cells
     * IntComparator comp = new IntComparator() {
     *     public int compare(int a, int b) {
     *         int as = Math.sin(a);
     *         int bs = Math.sin(b);
     *         return as &lt; bs ? -1 : as == bs ? 0 : 1;
     *     }
     * };
     * sorted = quickSort(vector, comp);
     * </pre>
     * 
     * @param vector
     *            the vector to be sorted.
     * @param c
     *            the comparator to determine the order.
     * @return a new matrix view sorted as specified. <b>Note that the original
     *         vector (matrix) is left unaffected.</b>
     */
    public LongMatrix1D sort(final LongMatrix1D vector, final cern.colt.function.tlong.LongComparator c) {
        return vector.viewSelection(sortIndex(vector, c));
    }

    /**
     * Sorts indexes of the <code>vector</code> according to the comparator
     * <code>c</code>.
     * 
     * @param vector
     * @param c
     * @return sorted indexes
     */
    public int[] sortIndex(final LongMatrix1D vector, final cern.colt.function.tlong.LongComparator c) {
        int[] indexes = new int[(int) vector.size()]; // row indexes to reorder
        // instead of matrix itself
        for (int i = indexes.length; --i >= 0;)
            indexes[i] = i;
        IntComparator comp = null;
        if (vector instanceof DenseLongMatrix1D) {
            final long[] velems = (long[]) vector.elements();
            final int zero = (int) vector.index(0);
            final int stride = vector.stride();
            comp = new IntComparator() {
                public int compare(int a, int b) {
                    int idxa = zero + a * stride;
                    int idxb = zero + b * stride;
                    return c.compare(velems[idxa], velems[idxb]);
                }
            };
        } else {
            comp = new IntComparator() {
                public int compare(int a, int b) {
                    return c.compare(vector.getQuick(a), vector.getQuick(b));
                }
            };
        }

        runSort(indexes, 0, indexes.length, comp);

        return indexes;
    }

    /**
     * Sorts the matrix rows into ascending order, according to the <i>natural
     * ordering</i> of the matrix values in the virtual column
     * <tt>aggregates</tt>; Particularly efficient when comparing expensive
     * aggregates, because aggregates need not be recomputed time and again, as
     * is the case for comparator based sorts. Essentially, this algorithm makes
     * expensive comparisons cheap. Normally each element of <tt>aggregates</tt>
     * is a summary measure of a row. Speedup over comparator based sorting =
     * <tt>2*log(rows)</tt>, on average. For this operation, quicksort is
     * usually faster.
     * <p>
     * The returned view is backed by this matrix, so changes in the returned
     * view are reflected in this matrix, and vice-versa. To sort ranges use
     * sub-ranging views. To sort columns by rows, use dice views. To sort
     * descending, use flip views ...
     * <p>
     * <b>Example:</b> Each aggregate is the sum of a row
     * <table border="1" * cellspacing="0">
     * <tr nowrap>
     * <td valign="top"><tt>4 x 2 matrix: <br>
     1, 1<br>
     5, 4<br>
     3, 0<br>
     4, 4 <br>
     </tt></td>
     * <td align="left" valign="top"> <tt>aggregates=<br>
     2<br>
     9<br>
     3<br>
     8<br>
     ==></tt></td>
     * <td valign="top">
     * <p>
     * <tt>4 x 2 matrix:<br>
     1, 1<br>
     3, 0<br>
     4, 4<br>
     5, 4</tt><br>
     * The matrix IS NOT SORTED.<br>
     * The new VIEW IS SORTED.
     * </p>
     * </td>
     * </tr>
     * </table>
     * 
     * <table>
     * <td class="PRE">
     * 
     * <pre>
     * // sort 10000 x 1000 matrix by sum of logarithms in a row (i.e. by geometric mean)
     * LongMatrix2D matrix = new DenseLongMatrix2D(10000, 1000);
     * matrix.assign(new cern.jet.random.engine.MersenneTwister()); // initialized randomly
     * cern.jet.math.Functions F = cern.jet.math.Functions.functions; // alias for convenience
     * 
     * // THE QUICK VERSION (takes some 3 secs)
     * // aggregates[i] = Sum(log(row));
     * int[] aggregates = new int[matrix.rows()];
     * for (int i = matrix.rows(); --i &gt;= 0;)
     *     aggregates[i] = matrix.viewRow(i).aggregate(F.plus, F.log);
     * LongMatrix2D sorted = quickSort(matrix, aggregates);
     * 
     * // THE SLOW VERSION (takes some 90 secs)
     * LongMatrix1DComparator comparator = new LongMatrix1DComparator() {
     *     public int compare(LongMatrix1D x, LongMatrix1D y) {
     *         int a = x.aggregate(F.plus, F.log);
     *         int b = y.aggregate(F.plus, F.log);
     *         return a &lt; b ? -1 : a == b ? 0 : 1;
     *     }
     * };
     * LongMatrix2D sorted = quickSort(matrix, comparator);
     * </pre>
     * 
     * </td>
     * </table>
     * 
     * @param matrix
     *            the matrix to be sorted.
     * @param aggregates
     *            the values to sort on. (As a side effect, this array will also
     *            get sorted).
     * @return a new matrix view having rows sorted. <b>Note that the original
     *         matrix is left unaffected.</b>
     * @throws IndexOutOfBoundsException
     *             if <tt>aggregates.length != matrix.rows()</tt>.
     */
    public LongMatrix2D sort(LongMatrix2D matrix, final long[] aggregates) {
        int rows = matrix.rows();
        if (aggregates.length != rows)
            throw new IndexOutOfBoundsException("aggregates.length != matrix.rows()");

        // set up index reordering
        final int[] indexes = new int[rows];
        for (int i = rows; --i >= 0;)
            indexes[i] = i;

        // compares two aggregates at a time
        cern.colt.function.tint.IntComparator comp = new cern.colt.function.tint.IntComparator() {
            public int compare(int x, int y) {
                long a = aggregates[x];
                long b = aggregates[y];
                return a < b ? -1 : (a == b) ? 0 : 1;
            }
        };
        // swaps aggregates and reorders indexes
        cern.colt.Swapper swapper = new cern.colt.Swapper() {
            public void swap(int x, int y) {
                int t1;
                long t2;
                t1 = indexes[x];
                indexes[x] = indexes[y];
                indexes[y] = t1;
                t2 = aggregates[x];
                aggregates[x] = aggregates[y];
                aggregates[y] = t2;
            }
        };

        // sort indexes and aggregates
        runSort(0, rows, comp, swapper);

        // view the matrix according to the reordered row indexes
        // take all columns in the original order
        return matrix.viewSelection(indexes, null);
    }

    /**
     * Sorts the matrix rows into ascending order, according to the <i>natural
     * ordering</i> of the matrix values in the given column. The returned view
     * is backed by this matrix, so changes in the returned view are reflected
     * in this matrix, and vice-versa. To sort ranges use sub-ranging views. To
     * sort columns by rows, use dice views. To sort descending, use flip views
     * ...
     * <p>
     * <b>Example:</b>
     * <table border="1" cellspacing="0">
     * <tr nowrap>
     * <td valign="top"><tt>4 x 2 matrix: <br>
     7, 6<br>
     5, 4<br>
     3, 2<br>
     1, 0 <br>
     </tt></td>
     * <td align="left" valign="top">
     * <p>
     * <tt>column = 0;<br>
     view = quickSort(matrix,column);<br>
     System.out.println(view); </tt><tt><br>
     ==> </tt>
     * </p>
     * </td>
     * <td valign="top">
     * <p>
     * <tt>4 x 2 matrix:<br>
     1, 0<br>
     3, 2<br>
     5, 4<br>
     7, 6</tt><br>
     * The matrix IS NOT SORTED.<br>
     * The new VIEW IS SORTED.
     * </p>
     * </td>
     * </tr>
     * </table>
     * 
     * @param matrix
     *            the matrix to be sorted.
     * @param column
     *            the index of the column inducing the order.
     * @return a new matrix view having rows sorted by the given column. <b>Note
     *         that the original matrix is left unaffected.</b>
     * @throws IndexOutOfBoundsException
     *             if <tt>column < 0 || column >= matrix.columns()</tt>.
     */
    public LongMatrix2D sort(LongMatrix2D matrix, int column) {
        if (column < 0 || column >= matrix.columns())
            throw new IndexOutOfBoundsException("column=" + column + ", matrix=" + AbstractFormatter.shape(matrix));

        int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder
        // instead of matrix itself
        for (int i = rowIndexes.length; --i >= 0;)
            rowIndexes[i] = i;

        final LongMatrix1D col = matrix.viewColumn(column);
        IntComparator comp = new IntComparator() {
            public int compare(int a, int b) {
                long av = col.getQuick(a);
                long bv = col.getQuick(b);
                return av < bv ? -1 : (av == bv ? 0 : 1);
            }
        };

        runSort(rowIndexes, 0, rowIndexes.length, comp);

        // view the matrix according to the reordered row indexes
        // take all columns in the original order
        return matrix.viewSelection(rowIndexes, null);
    }

    /**
     * Sorts the matrix rows according to the order induced by the specified
     * comparator. The returned view is backed by this matrix, so changes in the
     * returned view are reflected in this matrix, and vice-versa. The algorithm
     * compares two rows (1-d matrices) at a time, determinining whether one is
     * smaller, equal or larger than the other. To sort ranges use sub-ranging
     * views. To sort columns by rows, use dice views. To sort descending, use
     * flip views ...
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * // sort by sum of values in a row
     * LongMatrix1DComparator comp = new LongMatrix1DComparator() {
     *     public int compare(LongMatrix1D a, LongMatrix1D b) {
     *         int as = a.zSum();
     *         int bs = b.zSum();
     *         return as &lt; bs ? -1 : as == bs ? 0 : 1;
     *     }
     * };
     * sorted = quickSort(matrix, comp);
     * </pre>
     * 
     * @param matrix
     *            the matrix to be sorted.
     * @param c
     *            the comparator to determine the order.
     * @return a new matrix view having rows sorted as specified. <b>Note that
     *         the original matrix is left unaffected.</b>
     */
    public LongMatrix2D sort(final LongMatrix2D matrix, final LongMatrix1DComparator c) {
        int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder
        // instead of matrix itself
        for (int i = rowIndexes.length; --i >= 0;)
            rowIndexes[i] = i;

        final LongMatrix1D[] views = new LongMatrix1D[matrix.rows()]; // precompute
        // views
        // for
        // speed
        for (int i = views.length; --i >= 0;)
            views[i] = matrix.viewRow(i);

        IntComparator comp = new IntComparator() {
            public int compare(int a, int b) {
                // return c.compare(matrix.viewRow(a), matrix.viewRow(b));
                return c.compare(views[a], views[b]);
            }
        };

        runSort(rowIndexes, 0, rowIndexes.length, comp);

        // view the matrix according to the reordered row indexes
        // take all columns in the original order
        return matrix.viewSelection(rowIndexes, null);
    }

    /**
     * Sorts the matrix slices into ascending order, according to the <i>natural
     * ordering</i> of the matrix values in the given <tt>[row,column]</tt>
     * position. The returned view is backed by this matrix, so changes in the
     * returned view are reflected in this matrix, and vice-versa. To sort
     * ranges use sub-ranging views. To sort by other dimensions, use dice
     * views. To sort descending, use flip views ...
     * <p>
     * The algorithm compares two 2-d slices at a time, determinining whether
     * one is smaller, equal or larger than the other. Comparison is based on
     * the cell <tt>[row,column]</tt> within a slice. Let <tt>A</tt> and
     * <tt>B</tt> be two 2-d slices. Then we have the following rules
     * <ul>
     * <li><tt>A &lt;  B  iff A.get(row,column) &lt;  B.get(row,column)</tt>
     * <li><tt>A == B iff A.get(row,column) == B.get(row,column)</tt>
     * <li><tt>A &gt;  B  iff A.get(row,column) &gt;  B.get(row,column)</tt>
     * </ul>
     * 
     * @param matrix
     *            the matrix to be sorted.
     * @param row
     *            the index of the row inducing the order.
     * @param column
     *            the index of the column inducing the order.
     * @return a new matrix view having slices sorted by the values of the slice
     *         view <tt>matrix.viewRow(row).viewColumn(column)</tt>. <b>Note
     *         that the original matrix is left unaffected.</b>
     * @throws IndexOutOfBoundsException
     *             if
     *             <tt>row < 0 || row >= matrix.rows() || column < 0 || column >= matrix.columns()</tt>
     *             .
     */
    public LongMatrix3D sort(LongMatrix3D matrix, int row, int column) {
        if (row < 0 || row >= matrix.rows())
            throw new IndexOutOfBoundsException("row=" + row + ", matrix=" + AbstractFormatter.shape(matrix));
        if (column < 0 || column >= matrix.columns())
            throw new IndexOutOfBoundsException("column=" + column + ", matrix=" + AbstractFormatter.shape(matrix));

        int[] sliceIndexes = new int[matrix.slices()]; // indexes to reorder
        // instead of matrix
        // itself
        for (int i = sliceIndexes.length; --i >= 0;)
            sliceIndexes[i] = i;

        final LongMatrix1D sliceView = matrix.viewRow(row).viewColumn(column);
        IntComparator comp = new IntComparator() {
            public int compare(int a, int b) {
                long av = sliceView.getQuick(a);
                long bv = sliceView.getQuick(b);
                return av < bv ? -1 : (av == bv ? 0 : 1);
            }
        };

        runSort(sliceIndexes, 0, sliceIndexes.length, comp);

        // view the matrix according to the reordered slice indexes
        // take all rows and columns in the original order
        return matrix.viewSelection(sliceIndexes, null, null);
    }

    /**
     * Sorts the matrix slices according to the order induced by the specified
     * comparator. The returned view is backed by this matrix, so changes in the
     * returned view are reflected in this matrix, and vice-versa. The algorithm
     * compares two slices (2-d matrices) at a time, determinining whether one
     * is smaller, equal or larger than the other. To sort ranges use
     * sub-ranging views. To sort by other dimensions, use dice views. To sort
     * descending, use flip views ...
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * // sort by sum of values in a slice
     * LongMatrix2DComparator comp = new LongMatrix2DComparator() {
     *     public int compare(LongMatrix2D a, LongMatrix2D b) {
     *         int as = a.zSum();
     *         int bs = b.zSum();
     *         return as &lt; bs ? -1 : as == bs ? 0 : 1;
     *     }
     * };
     * sorted = quickSort(matrix, comp);
     * </pre>
     * 
     * @param matrix
     *            the matrix to be sorted.
     * @param c
     *            the comparator to determine the order.
     * @return a new matrix view having slices sorted as specified. <b>Note that
     *         the original matrix is left unaffected.</b>
     */
    public LongMatrix3D sort(final LongMatrix3D matrix, final LongMatrix2DComparator c) {
        int[] sliceIndexes = new int[matrix.slices()]; // indexes to reorder
        // instead of matrix
        // itself
        for (int i = sliceIndexes.length; --i >= 0;)
            sliceIndexes[i] = i;

        final LongMatrix2D[] views = new LongMatrix2D[matrix.slices()]; // precompute
        // views
        // for
        // speed
        for (int i = views.length; --i >= 0;)
            views[i] = matrix.viewSlice(i);

        IntComparator comp = new IntComparator() {
            public int compare(int a, int b) {
                // return c.compare(matrix.viewSlice(a), matrix.viewSlice(b));
                return c.compare(views[a], views[b]);
            }
        };

        runSort(sliceIndexes, 0, sliceIndexes.length, comp);

        // view the matrix according to the reordered slice indexes
        // take all rows and columns in the original order
        return matrix.viewSelection(sliceIndexes, null, null);
    }
}
