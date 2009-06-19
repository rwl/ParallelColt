/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdcomplex;

import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix2D;

/**
 * Factory for convenient construction of 2-d matrices holding <tt>complex</tt>
 * cells. Also provides convenient methods to compose (concatenate) and
 * decompose (split) matrices from/to constituent blocks. </p>
 * <p>
 * &nbsp;
 * </p>
 * <table border="0" cellspacing="0">
 * <tr align="left" valign="top">
 * <td><i>Construction</i></td>
 * <td>Use idioms like <tt>ComplexFactory2D.dense.make(4,4)</tt> to construct
 * dense matrices, <tt>ComplexFactory2D.sparse.make(4,4)</tt> to construct
 * sparse matrices.</td>
 * </tr>
 * <tr align="left" valign="top">
 * <td><i> Construction with initial values </i></td>
 * <td>Use other <tt>make</tt> methods to construct matrices with given initial
 * values.</td>
 * </tr>
 * <tr align="left" valign="top">
 * <td><i> Appending rows and columns </i></td>
 * <td>Use methods {@link #appendColumns(DComplexMatrix2D,DComplexMatrix2D)
 * appendColumns}, {@link #appendColumns(DComplexMatrix2D,DComplexMatrix2D)
 * appendRows} and {@link #repeat(DComplexMatrix2D,int,int) repeat} to append
 * rows and columns.</td>
 * </tr>
 * <tr align="left" valign="top">
 * <td><i> General block matrices </i></td>
 * <td>Use methods {@link #compose(DComplexMatrix2D[][]) compose} and
 * {@link #decompose(DComplexMatrix2D[][],DComplexMatrix2D) decompose} to work
 * with general block matrices.</td>
 * </tr>
 * <tr align="left" valign="top">
 * <td><i> Diagonal matrices </i></td>
 * <td>Use methods {@link #diagonal(DComplexMatrix1D) diagonal(vector)},
 * {@link #diagonal(DComplexMatrix2D) diagonal(matrix)} and
 * {@link #identity(int) identity} to work with diagonal matrices.</td>
 * </tr>
 * <tr align="left" valign="top">
 * <td><i> Diagonal block matrices </i></td>
 * <td>Use method
 * {@link #composeDiagonal(DComplexMatrix2D,DComplexMatrix2D,DComplexMatrix2D)
 * composeDiagonal} to work with diagonal block matrices.</td>
 * </tr>
 * <tr align="left" valign="top">
 * <td><i>Random</i></td>
 * <td>Use methods {@link #random(int,int) random} and
 * {@link #sample(int,int,double[],double) sample} to construct random matrices.
 * </td>
 * </tr>
 * </table>
 * <p>
 * &nbsp;
 * </p>
 * <p>
 * If the factory is used frequently it might be useful to streamline the
 * notation. For example by aliasing:
 * </p>
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 *  ComplexFactory2D F = ComplexFactory2D.dense;
 *  F.make(4,4);
 *  F.random(4,4);
 *  ...
 * </pre>
 * 
 * </td>
 * </table>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DComplexFactory2D extends cern.colt.PersistentObject {
    private static final long serialVersionUID = 1L;

    /**
     * A factory producing dense matrices.
     */
    public static final DComplexFactory2D dense = new DComplexFactory2D();

    /**
     * A factory producing sparse hash matrices.
     */
    public static final DComplexFactory2D sparse = new DComplexFactory2D();

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected DComplexFactory2D() {
    }

    /**
     * C = A||B; Constructs a new matrix which is the column-wise concatenation
     * of two other matrices.
     * 
     * <pre>
     * 	 0 1 2
     * 	 3 4 5
     * 	 appendColumns
     * 	 6 7
     * 	 8 9
     * 	 --&gt;
     * 	 0 1 2 6 7 
     * 	 3 4 5 8 9
     * 
     * </pre>
     */
    public DComplexMatrix2D appendColumns(DComplexMatrix2D A, DComplexMatrix2D B) {
        // force both to have maximal shared number of rows.
        if (B.rows() > A.rows())
            B = B.viewPart(0, 0, A.rows(), B.columns());
        else if (B.rows() < A.rows())
            A = A.viewPart(0, 0, B.rows(), A.columns());

        // concatenate
        int ac = A.columns();
        int bc = B.columns();
        int r = A.rows();
        DComplexMatrix2D matrix = make(r, ac + bc);
        matrix.viewPart(0, 0, r, ac).assign(A);
        matrix.viewPart(0, ac, r, bc).assign(B);
        return matrix;
    }

    /**
     * C = A||b; Constructs a new matrix which is the column-wise concatenation
     * of two other matrices.
     * 
     * <pre>
     *   0 1 2
     *   3 4 5
     *   appendColumn
     *   6 
     *   8 
     *   --&gt;
     *   0 1 2 6  
     *   3 4 5 8
     * 
     * </pre>
     */
    public DComplexMatrix2D appendColumn(DComplexMatrix2D A, DComplexMatrix1D b) {
        // force both to have maximal shared number of rows.
        if (b.size() > A.rows())
            b = b.viewPart(0, A.rows());
        else if (b.size() < A.rows())
            A = A.viewPart(0, 0, (int) b.size(), A.columns());

        // concatenate
        int ac = A.columns();
        int bc = 1;
        int r = A.rows();
        DComplexMatrix2D matrix = make(r, ac + bc);
        matrix.viewPart(0, 0, r, ac).assign(A);
        matrix.viewColumn(ac).assign(b);
        return matrix;
    }

    /**
     * C = A||B; Constructs a new matrix which is the row-wise concatenation of
     * two other matrices.
     * 
     */
    public DComplexMatrix2D appendRows(DComplexMatrix2D A, DComplexMatrix2D B) {
        // force both to have maximal shared number of columns.
        if (B.columns() > A.columns())
            B = B.viewPart(0, 0, B.rows(), A.columns());
        else if (B.columns() < A.columns())
            A = A.viewPart(0, 0, A.rows(), B.columns());

        // concatenate
        int ar = A.rows();
        int br = B.rows();
        int c = A.columns();
        DComplexMatrix2D matrix = make(ar + br, c);
        matrix.viewPart(0, 0, ar, c).assign(A);
        matrix.viewPart(ar, 0, br, c).assign(B);
        return matrix;
    }

    /**
     * C = A||b; Constructs a new matrix which is the row-wise concatenation of
     * two other matrices.
     * 
     */
    public DComplexMatrix2D appendRow(DComplexMatrix2D A, DComplexMatrix1D b) {
        // force both to have maximal shared number of columns.
        if (b.size() > A.columns())
            b = b.viewPart(0, A.columns());
        else if (b.size() < A.columns())
            A = A.viewPart(0, 0, A.rows(), (int) b.size());

        // concatenate
        int ar = A.rows();
        int br = 1;
        int c = A.columns();
        DComplexMatrix2D matrix = make(ar + br, c);
        matrix.viewPart(0, 0, ar, c).assign(A);
        matrix.viewRow(ar).assign(b);
        return matrix;
    }

    /**
     * Checks whether the given array is rectangular, that is, whether all rows
     * have the same number of columns.
     * 
     * @throws IllegalArgumentException
     *             if the array is not rectangular.
     */
    protected static void checkRectangularShape(double[][] array) {
        int columns = -1;
        for (int r = 0; r < array.length; r++) {
            if (array[r] != null) {
                if (columns == -1)
                    columns = array[r].length;
                if (array[r].length != columns)
                    throw new IllegalArgumentException("All rows of array must have same number of columns.");
            }
        }
    }

    /**
     * Checks whether the given array is rectangular, that is, whether all rows
     * have the same number of columns.
     * 
     * @throws IllegalArgumentException
     *             if the array is not rectangular.
     */
    protected static void checkRectangularShape(DComplexMatrix2D[][] array) {
        int columns = -1;
        for (int r = 0; r < array.length; r++) {
            if (array[r] != null) {
                if (columns == -1)
                    columns = array[r].length;
                if (array[r].length != columns)
                    throw new IllegalArgumentException("All rows of array must have same number of columns.");
            }
        }
    }

    /**
     * Constructs a block matrix made from the given parts. The inverse to
     * method {@link #decompose(DComplexMatrix2D[][], DComplexMatrix2D)}.
     * <p>
     * All matrices of a given column within <tt>parts</tt> must have the same
     * number of columns. All matrices of a given row within <tt>parts</tt> must
     * have the same number of rows. Otherwise an
     * <tt>IllegalArgumentException</tt> is thrown. Note that <tt>null</tt>s
     * within <tt>parts[row,col]</tt> are an exception to this rule: they are
     * ignored. Cells are copied.
     * 
     * @throws IllegalArgumentException
     *             subject to the conditions outlined above.
     */
    public DComplexMatrix2D compose(DComplexMatrix2D[][] parts) {
        checkRectangularShape(parts);
        int rows = parts.length;
        int columns = 0;
        if (parts.length > 0)
            columns = parts[0].length;
        DComplexMatrix2D empty = make(0, 0);

        if (rows == 0 || columns == 0)
            return empty;

        // determine maximum column width of each column
        int[] maxWidths = new int[columns];
        for (int c = 0; c < columns; c++) {
            int maxWidth = 0;
            for (int r = 0; r < rows; r++) {
                DComplexMatrix2D part = parts[r][c];
                if (part != null) {
                    int width = part.columns();
                    if (maxWidth > 0 && width > 0 && width != maxWidth)
                        throw new IllegalArgumentException("Different number of columns.");
                    maxWidth = Math.max(maxWidth, width);
                }
            }
            maxWidths[c] = maxWidth;
        }

        // determine row height of each row
        int[] maxHeights = new int[rows];
        for (int r = 0; r < rows; r++) {
            int maxHeight = 0;
            for (int c = 0; c < columns; c++) {
                DComplexMatrix2D part = parts[r][c];
                if (part != null) {
                    int height = part.rows();
                    if (maxHeight > 0 && height > 0 && height != maxHeight)
                        throw new IllegalArgumentException("Different number of rows.");
                    maxHeight = Math.max(maxHeight, height);
                }
            }
            maxHeights[r] = maxHeight;
        }

        // shape of result
        int resultRows = 0;
        for (int r = 0; r < rows; r++)
            resultRows += maxHeights[r];
        int resultCols = 0;
        for (int c = 0; c < columns; c++)
            resultCols += maxWidths[c];

        DComplexMatrix2D matrix = make(resultRows, resultCols);

        // copy
        int idxr = 0;
        for (int r = 0; r < rows; r++) {
            int idxc = 0;
            for (int c = 0; c < columns; c++) {
                DComplexMatrix2D part = parts[r][c];
                if (part != null) {
                    matrix.viewPart(idxr, idxc, part.rows(), part.columns()).assign(part);
                }
                idxc += maxWidths[c];
            }
            idxr += maxHeights[r];
        }

        return matrix;
    }

    /**
     * Constructs a diagonal block matrix from the given parts (the <i>direct
     * sum</i> of two matrices). That is the concatenation
     * 
     * <pre>
     * 	 A 0
     * 	 0 B
     * 
     * </pre>
     * 
     * (The direct sum has <tt>A.rows()+B.rows()</tt> rows and
     * <tt>A.columns()+B.columns()</tt> columns). Cells are copied.
     * 
     * @return a new matrix which is the direct sum.
     */
    public DComplexMatrix2D composeDiagonal(DComplexMatrix2D A, DComplexMatrix2D B) {
        int ar = A.rows();
        int ac = A.columns();
        int br = B.rows();
        int bc = B.columns();
        DComplexMatrix2D sum = make(ar + br, ac + bc);
        sum.viewPart(0, 0, ar, ac).assign(A);
        sum.viewPart(ar, ac, br, bc).assign(B);
        return sum;
    }

    /**
     * Constructs a diagonal block matrix from the given parts. The
     * concatenation has the form
     * 
     * <pre>
     * 	 A 0 0
     * 	 0 B 0
     * 	 0 0 C
     * 
     * </pre>
     * 
     * from the given parts. Cells are copied.
     */
    public DComplexMatrix2D composeDiagonal(DComplexMatrix2D A, DComplexMatrix2D B, DComplexMatrix2D C) {
        DComplexMatrix2D diag = make(A.rows() + B.rows() + C.rows(), A.columns() + B.columns() + C.columns());
        diag.viewPart(0, 0, A.rows(), A.columns()).assign(A);
        diag.viewPart(A.rows(), A.columns(), B.rows(), B.columns()).assign(B);
        diag.viewPart(A.rows() + B.rows(), A.columns() + B.columns(), C.rows(), C.columns()).assign(C);
        return diag;
    }

    /**
     * Constructs a bidiagonal block matrix from the given parts.
     * 
     * from the given parts. Cells are copied.
     */
    public DComplexMatrix2D composeBidiagonal(DComplexMatrix2D A, DComplexMatrix2D B) {
        int ar = A.rows();
        int ac = A.columns();
        int br = B.rows();
        int bc = B.columns();
        DComplexMatrix2D sum = make(ar + br - 1, ac + bc);
        sum.viewPart(0, 0, ar, ac).assign(A);
        sum.viewPart(ar - 1, ac, br, bc).assign(B);
        return sum;
    }

    /**
     * Splits a block matrix into its constituent blocks; Copies blocks of a
     * matrix into the given parts. The inverse to method
     * {@link #compose(DComplexMatrix2D[][])}.
     * <p>
     * All matrices of a given column within <tt>parts</tt> must have the same
     * number of columns. All matrices of a given row within <tt>parts</tt> must
     * have the same number of rows. Otherwise an
     * <tt>IllegalArgumentException</tt> is thrown. Note that <tt>null</tt>s
     * within <tt>parts[row,col]</tt> are an exception to this rule: they are
     * ignored. Cells are copied.
     * 
     * @throws IllegalArgumentException
     *             subject to the conditions outlined above.
     */
    public void decompose(DComplexMatrix2D[][] parts, DComplexMatrix2D matrix) {
        checkRectangularShape(parts);
        int rows = parts.length;
        int columns = 0;
        if (parts.length > 0)
            columns = parts[0].length;
        if (rows == 0 || columns == 0)
            return;

        // determine maximum column width of each column
        int[] maxWidths = new int[columns];
        for (int c = 0; c < columns; c++) {
            int maxWidth = 0;
            for (int r = 0; r < rows; r++) {
                DComplexMatrix2D part = parts[r][c];
                if (part != null) {
                    int width = part.columns();
                    if (maxWidth > 0 && width > 0 && width != maxWidth)
                        throw new IllegalArgumentException("Different number of columns.");
                    maxWidth = Math.max(maxWidth, width);
                }
            }
            maxWidths[c] = maxWidth;
        }

        // determine row height of each row
        int[] maxHeights = new int[rows];
        for (int r = 0; r < rows; r++) {
            int maxHeight = 0;
            for (int c = 0; c < columns; c++) {
                DComplexMatrix2D part = parts[r][c];
                if (part != null) {
                    int height = part.rows();
                    if (maxHeight > 0 && height > 0 && height != maxHeight)
                        throw new IllegalArgumentException("Different number of rows.");
                    maxHeight = Math.max(maxHeight, height);
                }
            }
            maxHeights[r] = maxHeight;
        }

        // shape of result parts
        int resultRows = 0;
        for (int r = 0; r < rows; r++)
            resultRows += maxHeights[r];
        int resultCols = 0;
        for (int c = 0; c < columns; c++)
            resultCols += maxWidths[c];

        if (matrix.rows() < resultRows || matrix.columns() < resultCols)
            throw new IllegalArgumentException("Parts larger than matrix.");

        // copy
        int idxr = 0;
        for (int r = 0; r < rows; r++) {
            int idxc = 0;
            for (int c = 0; c < columns; c++) {
                DComplexMatrix2D part = parts[r][c];
                if (part != null) {
                    part.assign(matrix.viewPart(idxr, idxc, part.rows(), part.columns()));
                }
                idxc += maxWidths[c];
            }
            idxr += maxHeights[r];
        }

    }

    /**
     * Demonstrates usage of this class.
     */
    public void demo1() {
        System.out.println("\n\n");
        DComplexMatrix2D[][] parts1 = { { null, make(2, 2, new double[] { 1, 2 }), null },
                { make(4, 4, new double[] { 3, 4 }), null, make(4, 3, new double[] { 5, 6 }) },
                { null, make(2, 2, new double[] { 7, 8 }), null } };
        System.out.println("\n" + compose(parts1));
    }

    /**
     * Demonstrates usage of this class.
     */
    public void demo2() {
        System.out.println("\n\n");
        DComplexMatrix2D matrix;
        DComplexMatrix2D A, B, C, D;
        DComplexMatrix2D _ = null;
        A = make(2, 2, new double[] { 1, 2 });
        B = make(4, 4, new double[] { 3, 4 });
        C = make(4, 3, new double[] { 5, 6 });
        D = make(2, 2, new double[] { 7, 8 });
        DComplexMatrix2D[][] parts1 = { { _, A, _ }, { B, _, C }, { _, D, _ } };
        matrix = compose(parts1);
        System.out.println("\n" + matrix);

        A.assign(9, 9);
        B.assign(9, 9);
        C.assign(9, 9);
        D.assign(9, 9);
        decompose(parts1, matrix);
        System.out.println(A);
        System.out.println(B);
        System.out.println(C);
        System.out.println(D);
    }

    /**
     * Constructs a new diagonal matrix whose diagonal elements are the elements
     * of <tt>vector</tt>. Cells values are copied. The new matrix is not a
     * view.
     * 
     * @return a new matrix.
     */
    public DComplexMatrix2D diagonal(DComplexMatrix1D vector) {
        int size = (int) vector.size();
        DComplexMatrix2D diag = make(size, size);
        for (int i = 0; i < size; i++) {
            diag.setQuick(i, i, vector.getQuick(i));
        }
        return diag;
    }

    /**
     * Constructs a new vector consisting of the diagonal elements of <tt>A</tt>
     * . Cells values are copied. The new vector is not a view.
     * 
     * @param A
     *            the matrix, need not be square.
     * @return a new vector.
     */
    public DComplexMatrix1D diagonal(DComplexMatrix2D A) {
        int min = Math.min(A.rows(), A.columns());
        DComplexMatrix1D diag = make1D(min);
        for (int i = 0; i < min; i++) {
            diag.setQuick(i, A.getQuick(i, i));
        }
        return diag;
    }

    /**
     * Constructs an identity matrix (having ones on the diagonal and zeros
     * elsewhere).
     */
    public DComplexMatrix2D identity(int rowsAndColumns) {
        DComplexMatrix2D matrix = make(rowsAndColumns, rowsAndColumns);
        double[] one = new double[] { 1, 0 };
        for (int i = rowsAndColumns; --i >= 0;) {
            matrix.setQuick(i, i, one);
        }
        return matrix;
    }

    /**
     * Constructs a matrix with the given cell values. <tt>values</tt> is
     * required to have the form <tt>values[row][column]</tt> and have exactly
     * the same number of columns in every row.
     * <p>
     * The values are copied. So subsequent changes in <tt>values</tt> are not
     * reflected in the matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     * @throws IllegalArgumentException
     *             if
     *             <tt>for any 1 &lt;= row &lt; values.length: values[row].length != values[row-1].length</tt>
     *             .
     */
    public DComplexMatrix2D make(double[][] values) {
        if (this == sparse) {
            return new SparseDComplexMatrix2D(values);
        } else {
            return new DenseDComplexMatrix2D(values);
        }
    }

    /**
     * Constructs a matrix with the given shape, each cell initialized with
     * zero.
     */
    public DComplexMatrix2D make(int rows, int columns) {
        if (this == sparse)
            return new SparseDComplexMatrix2D(rows, columns);
        else
            return new DenseDComplexMatrix2D(rows, columns);
    }

    /**
     * Constructs a matrix with the given shape, each cell initialized with the
     * given value.
     */
    public DComplexMatrix2D make(int rows, int columns, double[] initialValue) {
        if (initialValue[0] == 0 && initialValue[1] == 0)
            return make(rows, columns);
        return make(rows, columns).assign(initialValue);
    }

    /**
     * Constructs a 1d matrix of the right dynamic type.
     */
    protected DComplexMatrix1D make1D(int size) {
        return make(0, 0).like1D(size);
    }

    /**
     * Constructs a matrix with uniformly distributed values in <tt>(0,1)</tt>
     * (exclusive).
     */
    public DComplexMatrix2D random(int rows, int columns) {
        return make(rows, columns).assign(cern.jet.math.tdcomplex.DComplexFunctions.random());
    }

    /**
     * C = A||A||..||A; Constructs a new matrix which is duplicated both along
     * the row and column dimension.
     */
    public DComplexMatrix2D repeat(DComplexMatrix2D A, int rowRepeat, int columnRepeat) {
        int r = A.rows();
        int c = A.columns();
        DComplexMatrix2D matrix = make(r * rowRepeat, c * columnRepeat);
        for (int i = 0; i < rowRepeat; i++) {
            for (int j = 0; j < columnRepeat; j++) {
                matrix.viewPart(r * i, c * j, r, c).assign(A);
            }
        }
        return matrix;
    }

    /**
     * Constructs a randomly sampled matrix with the given shape. Randomly picks
     * exactly <tt>Math.round(rows*columns*nonZeroFraction)</tt> cells and
     * initializes them to <tt>value</tt>, all the rest will be initialized to
     * zero. Note that this is not the same as setting each cell with
     * probability <tt>nonZeroFraction</tt> to <tt>value</tt>. Note: The random
     * seed is a constant.
     * 
     * @throws IllegalArgumentException
     *             if <tt>nonZeroFraction < 0 || nonZeroFraction > 1</tt>.
     * @see cern.jet.random.tdouble.sampling.DoubleRandomSampler
     */
    public DComplexMatrix2D sample(int rows, int columns, double[] value, double nonZeroFraction) {
        DComplexMatrix2D matrix = make(rows, columns);
        sample(matrix, value, nonZeroFraction);
        return matrix;
    }

    /**
     * Modifies the given matrix to be a randomly sampled matrix. Randomly picks
     * exactly <tt>Math.round(rows*columns*nonZeroFraction)</tt> cells and
     * initializes them to <tt>value</tt>, all the rest will be initialized to
     * zero. Note that this is not the same as setting each cell with
     * probability <tt>nonZeroFraction</tt> to <tt>value</tt>. Note: The random
     * seed is a constant.
     * 
     * @throws IllegalArgumentException
     *             if <tt>nonZeroFraction < 0 || nonZeroFraction > 1</tt>.
     * @see cern.jet.random.tdouble.sampling.DoubleRandomSampler
     */
    public DComplexMatrix2D sample(DComplexMatrix2D matrix, double[] value, double nonZeroFraction) {
        int rows = matrix.rows();
        int columns = matrix.columns();
        double epsilon = 1e-09;
        if (nonZeroFraction < 0 - epsilon || nonZeroFraction > 1 + epsilon)
            throw new IllegalArgumentException();
        if (nonZeroFraction < 0)
            nonZeroFraction = 0;
        if (nonZeroFraction > 1)
            nonZeroFraction = 1;

        matrix.assign(0, 0);

        int size = rows * columns;
        int n = (int) Math.round(size * nonZeroFraction);
        if (n == 0)
            return matrix;

        cern.jet.random.tdouble.sampling.DoubleRandomSamplingAssistant sampler = new cern.jet.random.tdouble.sampling.DoubleRandomSamplingAssistant(
                n, size, new cern.jet.random.tdouble.engine.DoubleMersenneTwister());
        for (int i = 0; i < size; i++) {
            if (sampler.sampleNextElement()) {
                int row = (i / columns);
                int column = (i % columns);
                matrix.set(row, column, value);
            }
        }

        return matrix;
    }
}
