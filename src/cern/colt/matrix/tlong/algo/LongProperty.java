/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong.algo;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.matrix.AbstractFormatter;
import cern.colt.matrix.tlong.LongMatrix1D;
import cern.colt.matrix.tlong.LongMatrix2D;
import cern.colt.matrix.tlong.LongMatrix3D;
import cern.jet.math.tlong.LongFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Tests matrices for linear algebraic properties (equality, tridiagonality,
 * symmetry, singularity, etc).
 * <p>
 * 
 * Note that this implementation is not synchronized.
 * <p>
 * Here are some example properties
 * <table border="1" cellspacing="0">
 * <tr align="left" valign="top">
 * <td valign="middle" align="left"><tt>matrix</tt></td>
 * <td> <tt>4&nbsp;x&nbsp;4&nbsp;<br>
 0&nbsp;0&nbsp;0&nbsp;0<br>
 0&nbsp;0&nbsp;0&nbsp;0<br>
 0&nbsp;0&nbsp;0&nbsp;0<br>
 0&nbsp;0&nbsp;0&nbsp;0 </tt></td>
 * <td><tt>4&nbsp;x&nbsp;4<br>
 1&nbsp;0&nbsp;0&nbsp;0<br>
 0&nbsp;0&nbsp;0&nbsp;0<br>
 0&nbsp;0&nbsp;0&nbsp;0<br>
 0&nbsp;0&nbsp;0&nbsp;1 </tt></td>
 * <td><tt>4&nbsp;x&nbsp;4<br>
 1&nbsp;1&nbsp;0&nbsp;0<br>
 1&nbsp;1&nbsp;1&nbsp;0<br>
 0&nbsp;1&nbsp;1&nbsp;1<br>
 0&nbsp;0&nbsp;1&nbsp;1 </tt></td>
 * <td><tt> 4&nbsp;x&nbsp;4<br>
 0&nbsp;1&nbsp;1&nbsp;1<br>
 0&nbsp;1&nbsp;1&nbsp;1<br>
 0&nbsp;0&nbsp;0&nbsp;1<br>
 0&nbsp;0&nbsp;0&nbsp;1 </tt></td>
 * <td><tt> 4&nbsp;x&nbsp;4<br>
 0&nbsp;0&nbsp;0&nbsp;0<br>
 1&nbsp;1&nbsp;0&nbsp;0<br>
 1&nbsp;1&nbsp;0&nbsp;0<br>
 1&nbsp;1&nbsp;1&nbsp;1 </tt></td>
 * <td><tt>4&nbsp;x&nbsp;4<br>
 1&nbsp;1&nbsp;0&nbsp;0<br>
 0&nbsp;1&nbsp;1&nbsp;0<br>
 0&nbsp;1&nbsp;0&nbsp;1<br>
 1&nbsp;0&nbsp;1&nbsp;1 </tt><tt> </tt></td>
 * <td><tt>4&nbsp;x&nbsp;4<br>
 1&nbsp;1&nbsp;1&nbsp;0<br>
 0&nbsp;1&nbsp;0&nbsp;0<br>
 1&nbsp;1&nbsp;0&nbsp;1<br>
 0&nbsp;0&nbsp;1&nbsp;1 </tt></td>
 * </tr>
 * <tr align="center" valign="middle">
 * <td><tt>upperBandwidth</tt></td>
 * <td><div align="center"><tt>0</tt></div></td>
 * <td><div align="center"><tt>0</tt></div></td>
 * <td><div align="center"><tt>1</tt></div></td>
 * <td><tt>3</tt></td>
 * <td align="center" valign="middle"><tt>0</tt></td>
 * <td align="center" valign="middle"><div align="center"><tt>1</tt></div></td>
 * <td align="center" valign="middle"><div align="center"><tt>2</tt></div></td>
 * </tr>
 * <tr align="center" valign="middle">
 * <td><tt>lowerBandwidth</tt></td>
 * <td><div align="center"><tt>0</tt></div></td>
 * <td><div align="center"><tt>0</tt></div></td>
 * <td><div align="center"><tt>1</tt></div></td>
 * <td><tt>0</tt></td>
 * <td align="center" valign="middle"><tt>3</tt></td>
 * <td align="center" valign="middle"><div align="center"><tt>3</tt></div></td>
 * <td align="center" valign="middle"><div align="center"><tt>2</tt></div></td>
 * </tr>
 * <tr align="center" valign="middle">
 * <td><tt>semiBandwidth</tt></td>
 * <td><div align="center"><tt>1</tt></div></td>
 * <td><div align="center"><tt>1</tt></div></td>
 * <td><div align="center"><tt>2</tt></div></td>
 * <td><tt>4</tt></td>
 * <td align="center" valign="middle"><tt>4</tt></td>
 * <td align="center" valign="middle"><div align="center"><tt>4</tt></div></td>
 * <td align="center" valign="middle"><div align="center"><tt>3</tt></div></td>
 * </tr>
 * <tr align="center" valign="middle">
 * <td><tt>description</tt></td>
 * <td><div align="center"><tt>zero</tt></div></td>
 * <td><div align="center"><tt>diagonal</tt></div></td>
 * <td><div align="center"><tt>tridiagonal</tt></div></td>
 * <td><tt>upper triangular</tt></td>
 * <td align="center" valign="middle"><tt>lower triangular</tt></td>
 * <td align="center" valign="middle"><div align="center"><tt>unstructured</tt>
 * </div></td>
 * <td align="center" valign="middle"><div align="center"><tt>unstructured</tt>
 * </div></td>
 * </tr>
 * </table>
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class LongProperty extends cern.colt.PersistentObject {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /**
     * The default Property object;.
     */
    public static final LongProperty DEFAULT = new LongProperty();

    /**
     * Constructs an instance with a tolerance of
     * <tt>Math.abs(newTolerance)</tt>.
     */
    public LongProperty() {
    }

    /**
     * Returns a String with <tt>length</tt> blanks.
     */
    protected static String blanks(int length) {
        if (length < 0)
            length = 0;
        StringBuffer buf = new StringBuffer(length);
        for (int k = 0; k < length; k++) {
            buf.append(' ');
        }
        return buf.toString();
    }

    /**
     * Checks whether the given matrix <tt>A</tt> is <i>rectangular</i>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>A.rows() < A.columns()</tt>.
     */
    public void checkRectangular(LongMatrix2D A) {
        if (A.rows() < A.columns()) {
            throw new IllegalArgumentException("Matrix must be rectangular: " + AbstractFormatter.shape(A));
        }
    }

    /**
     * Checks whether the given matrix <tt>A</tt> is <i>square</i>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>A.rows() != A.columns()</tt>.
     */
    public void checkSquare(LongMatrix2D A) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("Matrix must be square: " + AbstractFormatter.shape(A));
    }

    /**
     * Returns the matrix's fraction of non-zero cells;
     * <tt>A.cardinality() / A.size()</tt>.
     */
    public int density(LongMatrix2D A) {
        return A.cardinality() / (int) A.size();
    }

    /**
     * Returns whether all cells of the given matrix <tt>A</tt> are equal to the
     * given value. The result is <tt>true</tt> if and only if
     * <tt>A != null</tt> and <tt>! (Math.abs(value - A[i]) > tolerance())</tt>
     * holds for all coordinates.
     * 
     * @param A
     *            the first matrix to compare.
     * @param value
     *            the value to compare against.
     * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final LongMatrix1D A, final long value) {
        if (A == null)
            return false;
        int size = (int) A.size();
        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            if (!(A.getQuick(i) == value))
                                return false;
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            for (int i = 0; i < size; i++) {
                if (!(A.getQuick(i) == value))
                    return false;
            }
            return true;
        }
    }

    /**
     * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal.
     * The result is <tt>true</tt> if <tt>A==B</tt>. Otherwise, the result is
     * <tt>true</tt> if and only if both arguments are <tt>!= null</tt>, have
     * the same size and <tt>! (Math.abs(A[i] - B[i]) > tolerance())</tt> holds
     * for all indexes.
     * 
     * @param A
     *            the first matrix to compare.
     * @param B
     *            the second matrix to compare.
     * @return <tt>true</tt> if both matrices are equal; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final LongMatrix1D A, final LongMatrix1D B) {
        if (A == B)
            return true;
        if (!(A != null && B != null))
            return false;
        int size = (int) A.size();
        if (size != B.size())
            return false;

        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            if (!(A.getQuick(i) == B.getQuick(i)))
                                return false;
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            for (int i = 0; i < size; i++) {
                if (!(A.getQuick(i) == B.getQuick(i)))
                    return false;
            }
            return true;
        }
    }

    /**
     * Returns whether all cells of the given matrix <tt>A</tt> are equal to the
     * given value. The result is <tt>true</tt> if and only if
     * <tt>A != null</tt> and
     * <tt>! (Math.abs(value - A[row,col]) > tolerance())</tt> holds for all
     * coordinates.
     * 
     * @param A
     *            the first matrix to compare.
     * @param value
     *            the value to compare against.
     * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final LongMatrix2D A, final long value) {
        if (A == null)
            return false;
        final int rows = A.rows();
        final int columns = A.columns();
        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, A.rows());
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = A.rows() / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? A.rows() : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                if (!(A.getQuick(r, c) == value))
                                    return false;
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    if (!(A.getQuick(r, c) == value))
                        return false;
                }
            }
            return true;
        }
    }

    /**
     * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal.
     * The result is <tt>true</tt> if <tt>A==B</tt>. Otherwise, the result is
     * <tt>true</tt> if and only if both arguments are <tt>!= null</tt>, have
     * the same number of columns and rows and
     * <tt>! (Math.abs(A[row,col] - B[row,col]) > tolerance())</tt> holds for
     * all coordinates.
     * 
     * @param A
     *            the first matrix to compare.
     * @param B
     *            the second matrix to compare.
     * @return <tt>true</tt> if both matrices are equal; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final LongMatrix2D A, final LongMatrix2D B) {
        if (A == B)
            return true;
        if (!(A != null && B != null))
            return false;
        final int rows = A.rows();
        final int columns = A.columns();
        if (columns != B.columns() || rows != B.rows())
            return false;
        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, A.rows());
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = A.rows() / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? A.rows() : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                if (!(A.getQuick(r, c) == B.getQuick(r, c)))
                                    return false;
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    if (!(A.getQuick(r, c) == B.getQuick(r, c)))
                        return false;
                }
            }
            return true;
        }
    }

    /**
     * Returns whether all cells of the given matrix <tt>A</tt> are equal to the
     * given value. The result is <tt>true</tt> if and only if
     * <tt>A != null</tt> and
     * <tt>! (Math.abs(value - A[slice,row,col]) > tolerance())</tt> holds for
     * all coordinates.
     * 
     * @param A
     *            the first matrix to compare.
     * @param value
     *            the value to compare against.
     * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final LongMatrix3D A, final long value) {
        if (A == null)
            return false;
        final int slices = A.slices();
        final int rows = A.rows();
        final int columns = A.columns();
        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int startslice = j * k;
                final int stopslice;
                if (j == nthreads - 1) {
                    stopslice = slices;
                } else {
                    stopslice = startslice + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        for (int s = startslice; s < stopslice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    if (!(A.getQuick(s, r, c) == value))
                                        return false;
                                }
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        if (!(A.getQuick(s, r, c) == value))
                            return false;
                    }
                }
            }
            return true;
        }
    }

    /**
     * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal.
     * The result is <tt>true</tt> if <tt>A==B</tt>. Otherwise, the result is
     * <tt>true</tt> if and only if both arguments are <tt>!= null</tt>, have
     * the same number of columns, rows and slices, and
     * <tt>! (Math.abs(A[slice,row,col] - B[slice,row,col]) > tolerance())</tt>
     * holds for all coordinates.
     * 
     * @param A
     *            the first matrix to compare.
     * @param B
     *            the second matrix to compare.
     * @return <tt>true</tt> if both matrices are equal; <tt>false</tt>
     *         otherwise.
     */
    public boolean equals(final LongMatrix3D A, final LongMatrix3D B) {
        if (A == B)
            return true;
        if (!(A != null && B != null))
            return false;
        final int slices = A.slices();
        final int rows = A.rows();
        final int columns = A.columns();
        if (columns != B.columns() || rows != B.rows() || slices != B.slices())
            return false;
        boolean result = false;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (A.size() >= ConcurrencyUtils.getThreadsBeginN_3D())) {
            nthreads = Math.min(nthreads, slices);
            Future<?>[] futures = new Future[nthreads];
            Boolean[] results = new Boolean[nthreads];
            int k = slices / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int startslice = j * k;
                final int stopslice;
                if (j == nthreads - 1) {
                    stopslice = slices;
                } else {
                    stopslice = startslice + k;
                }
                futures[j] = ConcurrencyUtils.submit(new Callable<Boolean>() {
                    public Boolean call() throws Exception {
                        for (int s = startslice; s < stopslice; s++) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < columns; c++) {
                                    if (!(A.getQuick(s, r, c) == B.getQuick(s, r, c)))
                                        return false;
                                }
                            }
                        }
                        return true;
                    }
                });
            }
            try {
                for (int j = 0; j < nthreads; j++) {
                    results[j] = (Boolean) futures[j].get();
                }
                result = results[0].booleanValue();
                for (int j = 1; j < nthreads; j++) {
                    result = result && results[j].booleanValue();
                }
            } catch (ExecutionException ex) {
                ex.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return result;
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        if (!(A.getQuick(s, r, c) == B.getQuick(s, r, c)))
                            return false;
                    }
                }
            }
            return true;
        }
    }

    /**
     * Modifies the given matrix square matrix <tt>A</tt> such that it is
     * diagonally dominant by row and column, hence non-singular, hence
     * invertible. For testing purposes only.
     * 
     * @param A
     *            the square matrix to modify.
     * @throws IllegalArgumentException
     *             if <tt>!isSquare(A)</tt>.
     */
    public void generateNonSingular(LongMatrix2D A) {
        checkSquare(A);
        cern.jet.math.tlong.LongFunctions F = cern.jet.math.tlong.LongFunctions.longFunctions;
        int min = Math.min(A.rows(), A.columns());
        for (int i = min; --i >= 0;) {
            A.setQuick(i, i, 0);
        }
        for (int i = min; --i >= 0;) {
            long rowSum = A.viewRow(i).aggregate(LongFunctions.plus, LongFunctions.abs);
            long colSum = A.viewColumn(i).aggregate(LongFunctions.plus, LongFunctions.abs);
            A.setQuick(i, i, Math.max(rowSum, colSum) + i + 1);
        }
    }

    /**
     */
    protected static String get(cern.colt.list.tobject.ObjectArrayList list, int index) {
        return ((String) list.get(index));
    }

    /**
     * A matrix <tt>A</tt> is <i>diagonal</i> if <tt>A[i,j] == 0</tt> whenever
     * <tt>i != j</tt>. Matrix may but need not be square.
     */
    public boolean isDiagonal(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = columns; --column >= 0;) {
                if (row != column && A.getQuick(row, column) != 0)
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>diagonally dominant by column</i> if the
     * absolute value of each diagonal element is larger than the sum of the
     * absolute values of the off-diagonal elements in the corresponding column.
     * 
     * <tt>returns true if for all i: abs(A[i,i]) &gt; Sum(abs(A[j,i])); j != i.</tt>
     * Matrix may but need not be square.
     * <p>
     * Note: Ignores tolerance.
     */
    public boolean isDiagonallyDominantByColumn(LongMatrix2D A) {
        cern.jet.math.tlong.LongFunctions F = cern.jet.math.tlong.LongFunctions.longFunctions;
        int min = Math.min(A.rows(), A.columns());
        for (int i = min; --i >= 0;) {
            long diag = Math.abs(A.getQuick(i, i));
            diag += diag;
            if (diag <= A.viewColumn(i).aggregate(LongFunctions.plus, LongFunctions.abs))
                return false;
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>diagonally dominant by row</i> if the absolute
     * value of each diagonal element is larger than the sum of the absolute
     * values of the off-diagonal elements in the corresponding row.
     * <tt>returns true if for all i: abs(A[i,i]) &gt; Sum(abs(A[i,j])); j != i.</tt>
     * Matrix may but need not be square.
     * <p>
     * Note: Ignores tolerance.
     */
    public boolean isDiagonallyDominantByRow(LongMatrix2D A) {
        cern.jet.math.tlong.LongFunctions F = cern.jet.math.tlong.LongFunctions.longFunctions;
        int min = Math.min(A.rows(), A.columns());
        for (int i = min; --i >= 0;) {
            long diag = Math.abs(A.getQuick(i, i));
            diag += diag;
            if (diag <= A.viewRow(i).aggregate(LongFunctions.plus, LongFunctions.abs))
                return false;
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is an <i>identity</i> matrix if <tt>A[i,i] == 1</tt>
     * and all other cells are zero. Matrix may but need not be square.
     */
    public boolean isIdentity(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = columns; --column >= 0;) {
                long v = A.getQuick(row, column);
                if (row == column) {
                    if (v != 1)
                        return false;
                } else if (v != 0)
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>lower bidiagonal</i> if <tt>A[i,j]==0</tt>
     * unless <tt>i==j || i==j+1</tt>. Matrix may but need not be square.
     */
    public boolean isLowerBidiagonal(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = columns; --column >= 0;) {
                if (!(row == column || row == column + 1)) {
                    if (A.getQuick(row, column) != 0)
                        return false;
                }
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>lower triangular</i> if <tt>A[i,j]==0</tt>
     * whenever <tt>i &lt; j</tt>. Matrix may but need not be square.
     */
    public boolean isLowerTriangular(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int column = columns; --column >= 0;) {
            for (int row = Math.min(column, rows); --row >= 0;) {
                if (A.getQuick(row, column) != 0)
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>non-negative</i> if <tt>A[i,j] &gt;= 0</tt>
     * holds for all cells.
     * <p>
     * Note: Ignores tolerance.
     */
    public boolean isNonNegative(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = columns; --column >= 0;) {
                if (!(A.getQuick(row, column) >= 0))
                    return false;
            }
        }
        return true;
    }

    /**
     * A square matrix <tt>A</tt> is <i>orthogonal</i> if
     * <tt>A*transpose(A) = I</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>!isSquare(A)</tt>.
     */
    public boolean isOrthogonal(LongMatrix2D A) {
        checkSquare(A);
        return equals(A.zMult(A, null, 1, 0, false, true), cern.colt.matrix.tlong.LongFactory2D.dense
                .identity(A.rows()));
    }

    /**
     * A matrix <tt>A</tt> is <i>positive</i> if <tt>A[i,j] &gt; 0</tt> holds
     * for all cells.
     * <p>
     * Note: Ignores tolerance.
     */
    public boolean isPositive(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = columns; --column >= 0;) {
                if (!(A.getQuick(row, column) > 0))
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>singular</i> if it has no inverse, that is, iff
     * <tt>det(A)==0</tt>.
     */
    //    public boolean isSingular(LongMatrix2D A) {
    //        return !(Math.abs(LongAlgebra.DEFAULT.det(A)) >= tolerance());
    //    }
    /**
     * A square matrix <tt>A</tt> is <i>skew-symmetric</i> if
     * <tt>A = -transpose(A)</tt>, that is <tt>A[i,j] == -A[j,i]</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>!isSquare(A)</tt>.
     */
    public boolean isSkewSymmetric(LongMatrix2D A) {
        checkSquare(A);
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = rows; --column >= 0;) {
                if (A.getQuick(row, column) != -A.getQuick(column, row))
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>square</i> if it has the same number of rows
     * and columns.
     */
    public boolean isSquare(LongMatrix2D A) {
        return A.rows() == A.columns();
    }

    /**
     * A matrix <tt>A</tt> is <i>strictly lower triangular</i> if
     * <tt>A[i,j]==0</tt> whenever <tt>i &lt;= j</tt>. Matrix may but need not
     * be square.
     */
    public boolean isStrictlyLowerTriangular(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int column = columns; --column >= 0;) {
            for (int row = Math.min(rows, column + 1); --row >= 0;) {
                if (A.getQuick(row, column) != 0)
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>strictly triangular</i> if it is triangular and
     * its diagonal elements all equal 0. Matrix may but need not be square.
     */
    public boolean isStrictlyTriangular(LongMatrix2D A) {
        if (!isTriangular(A))
            return false;

        for (int i = Math.min(A.rows(), A.columns()); --i >= 0;) {
            if (A.getQuick(i, i) != 0)
                return false;
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>strictly upper triangular</i> if
     * <tt>A[i,j]==0</tt> whenever <tt>i &gt;= j</tt>. Matrix may but need not
     * be square.
     */
    public boolean isStrictlyUpperTriangular(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int column = columns; --column >= 0;) {
            for (int row = rows; --row >= column;) {
                if (A.getQuick(row, column) != 0)
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>symmetric</i> if <tt>A = tranpose(A)</tt>, that
     * is <tt>A[i,j] == A[j,i]</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>!isSquare(A)</tt>.
     */
    public boolean isSymmetric(LongMatrix2D A) {
        checkSquare(A);
        return equals(A, A.viewDice());
    }

    /**
     * A matrix <tt>A</tt> is <i>triangular</i> iff it is either upper or lower
     * triangular. Matrix may but need not be square.
     */
    public boolean isTriangular(LongMatrix2D A) {
        return isLowerTriangular(A) || isUpperTriangular(A);
    }

    /**
     * A matrix <tt>A</tt> is <i>tridiagonal</i> if <tt>A[i,j]==0</tt> whenever
     * <tt>Math.abs(i-j) > 1</tt>. Matrix may but need not be square.
     */
    public boolean isTridiagonal(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = columns; --column >= 0;) {
                if (Math.abs(row - column) > 1) {
                    if (A.getQuick(row, column) != 0)
                        return false;
                }
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>unit triangular</i> if it is triangular and its
     * diagonal elements all equal 1. Matrix may but need not be square.
     */
    public boolean isUnitTriangular(LongMatrix2D A) {
        if (!isTriangular(A))
            return false;

        for (int i = Math.min(A.rows(), A.columns()); --i >= 0;) {
            if (A.getQuick(i, i) != 1)
                return false;
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>upper bidiagonal</i> if <tt>A[i,j]==0</tt>
     * unless <tt>i==j || i==j-1</tt>. Matrix may but need not be square.
     */
    public boolean isUpperBidiagonal(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int row = rows; --row >= 0;) {
            for (int column = columns; --column >= 0;) {
                if (!(row == column || row == column - 1)) {
                    if (A.getQuick(row, column) != 0)
                        return false;
                }
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>upper triangular</i> if <tt>A[i,j]==0</tt>
     * whenever <tt>i &gt; j</tt>. Matrix may but need not be square.
     */
    public boolean isUpperTriangular(LongMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int column = columns; --column >= 0;) {
            for (int row = rows; --row > column;) {
                if (A.getQuick(row, column) != 0)
                    return false;
            }
        }
        return true;
    }

    /**
     * A matrix <tt>A</tt> is <i>zero</i> if all its cells are zero.
     */
    public boolean isZero(LongMatrix2D A) {
        return equals(A, 0);
    }

    /**
     * The <i>lower bandwidth</i> of a square matrix <tt>A</tt> is the maximum
     * <tt>i-j</tt> for which <tt>A[i,j]</tt> is nonzero and <tt>i &gt; j</tt>.
     * A <i>banded</i> matrix has a "band" about the diagonal. Diagonal,
     * tridiagonal and triangular matrices are special cases.
     * 
     * @param A
     *            the square matrix to analyze.
     * @return the lower bandwith.
     * @throws IllegalArgumentException
     *             if <tt>!isSquare(A)</tt>.
     * @see #semiBandwidth(LongMatrix2D)
     * @see #upperBandwidth(LongMatrix2D)
     */
    public int lowerBandwidth(LongMatrix2D A) {
        checkSquare(A);
        int rows = A.rows();

        for (int k = rows; --k >= 0;) {
            for (int i = rows - k; --i >= 0;) {
                int j = i + k;
                if (A.getQuick(j, i) != 0)
                    return k;
            }
        }
        return 0;
    }

    /**
     * Returns the <i>semi-bandwidth</i> of the given square matrix <tt>A</tt>.
     * A <i>banded</i> matrix has a "band" about the diagonal. It is a matrix
     * with all cells equal to zero, with the possible exception of the cells
     * along the diagonal line, the <tt>k</tt> diagonal lines above the
     * diagonal, and the <tt>k</tt> diagonal lines below the diagonal. The
     * <i>semi-bandwith l</i> is the number <tt>k+1</tt>. The <i>bandwidth p</i>
     * is the number <tt>2*k + 1</tt>. For example, a tridiagonal matrix
     * corresponds to <tt>k=1, l=2, p=3</tt>, a diagonal or zero matrix
     * corresponds to <tt>k=0, l=1, p=1</tt>,
     * <p>
     * The <i>upper bandwidth</i> is the maximum <tt>j-i</tt> for which
     * <tt>A[i,j]</tt> is nonzero and <tt>j &gt; i</tt>. The <i>lower
     * bandwidth</i> is the maximum <tt>i-j</tt> for which <tt>A[i,j]</tt> is
     * nonzero and <tt>i &gt; j</tt>. Diagonal, tridiagonal and triangular
     * matrices are special cases.
     * <p>
     * Examples:
     * <table border="1" cellspacing="0">
     * <tr align="left" valign="top">
     * <td valign="middle" align="left"><tt>matrix</tt></td>
     * <td> <tt>4&nbsp;x&nbsp;4&nbsp;<br>
     0&nbsp;0&nbsp;0&nbsp;0<br>
     0&nbsp;0&nbsp;0&nbsp;0<br>
     0&nbsp;0&nbsp;0&nbsp;0<br>
     0&nbsp;0&nbsp;0&nbsp;0 </tt></td>
     * <td><tt>4&nbsp;x&nbsp;4<br>
     1&nbsp;0&nbsp;0&nbsp;0<br>
     0&nbsp;0&nbsp;0&nbsp;0<br>
     0&nbsp;0&nbsp;0&nbsp;0<br>
     0&nbsp;0&nbsp;0&nbsp;1 </tt></td>
     * <td><tt>4&nbsp;x&nbsp;4<br>
     1&nbsp;1&nbsp;0&nbsp;0<br>
     1&nbsp;1&nbsp;1&nbsp;0<br>
     0&nbsp;1&nbsp;1&nbsp;1<br>
     0&nbsp;0&nbsp;1&nbsp;1 </tt></td>
     * <td><tt> 4&nbsp;x&nbsp;4<br>
     0&nbsp;1&nbsp;1&nbsp;1<br>
     0&nbsp;1&nbsp;1&nbsp;1<br>
     0&nbsp;0&nbsp;0&nbsp;1<br>
     0&nbsp;0&nbsp;0&nbsp;1 </tt></td>
     * <td><tt> 4&nbsp;x&nbsp;4<br>
     0&nbsp;0&nbsp;0&nbsp;0<br>
     1&nbsp;1&nbsp;0&nbsp;0<br>
     1&nbsp;1&nbsp;0&nbsp;0<br>
     1&nbsp;1&nbsp;1&nbsp;1 </tt></td>
     * <td><tt>4&nbsp;x&nbsp;4<br>
     1&nbsp;1&nbsp;0&nbsp;0<br>
     0&nbsp;1&nbsp;1&nbsp;0<br>
     0&nbsp;1&nbsp;0&nbsp;1<br>
     1&nbsp;0&nbsp;1&nbsp;1 </tt><tt> </tt></td>
     * <td><tt>4&nbsp;x&nbsp;4<br>
     1&nbsp;1&nbsp;1&nbsp;0<br>
     0&nbsp;1&nbsp;0&nbsp;0<br>
     1&nbsp;1&nbsp;0&nbsp;1<br>
     0&nbsp;0&nbsp;1&nbsp;1 </tt></td>
     * </tr>
     * <tr align="center" valign="middle">
     * <td><tt>upperBandwidth</tt></td>
     * <td><div align="center"><tt>0</tt></div></td>
     * <td><div align="center"><tt>0</tt></div></td>
     * <td><div align="center"><tt>1</tt></div></td>
     * <td><tt>3</tt></td>
     * <td align="center" valign="middle"><tt>0</tt></td>
     * <td align="center" valign="middle"><div align="center"><tt>1</tt></div></td>
     * <td align="center" valign="middle"><div align="center"><tt>2</tt></div></td>
     * </tr>
     * <tr align="center" valign="middle">
     * <td><tt>lowerBandwidth</tt></td>
     * <td><div align="center"><tt>0</tt></div></td>
     * <td><div align="center"><tt>0</tt></div></td>
     * <td><div align="center"><tt>1</tt></div></td>
     * <td><tt>0</tt></td>
     * <td align="center" valign="middle"><tt>3</tt></td>
     * <td align="center" valign="middle"><div align="center"><tt>3</tt></div></td>
     * <td align="center" valign="middle"><div align="center"><tt>2</tt></div></td>
     * </tr>
     * <tr align="center" valign="middle">
     * <td><tt>semiBandwidth</tt></td>
     * <td><div align="center"><tt>1</tt></div></td>
     * <td><div align="center"><tt>1</tt></div></td>
     * <td><div align="center"><tt>2</tt></div></td>
     * <td><tt>4</tt></td>
     * <td align="center" valign="middle"><tt>4</tt></td>
     * <td align="center" valign="middle"><div align="center"><tt>4</tt></div></td>
     * <td align="center" valign="middle"><div align="center"><tt>3</tt></div></td>
     * </tr>
     * <tr align="center" valign="middle">
     * <td><tt>description</tt></td>
     * <td><div align="center"><tt>zero</tt></div></td>
     * <td><div align="center"><tt>diagonal</tt></div></td>
     * <td><div align="center"><tt>tridiagonal</tt></div></td>
     * <td><tt>upper triangular</tt></td>
     * <td align="center" valign="middle"><tt>lower triangular</tt></td>
     * <td align="center" valign="middle"><div align="center">
     * <tt>unstructured</tt></div></td>
     * <td align="center" valign="middle"><div align="center">
     * <tt>unstructured</tt></div></td>
     * </tr>
     * </table>
     * 
     * @param A
     *            the square matrix to analyze.
     * @return the semi-bandwith <tt>l</tt>.
     * @throws IllegalArgumentException
     *             if <tt>!isSquare(A)</tt>.
     * @see #lowerBandwidth(LongMatrix2D)
     * @see #upperBandwidth(LongMatrix2D)
     */
    public int semiBandwidth(LongMatrix2D A) {
        checkSquare(A);
        int rows = A.rows();

        for (int k = rows; --k >= 0;) {
            for (int i = rows - k; --i >= 0;) {
                int j = i + k;
                if (A.getQuick(j, i) != 0)
                    return k + 1;
                if (A.getQuick(i, j) != 0)
                    return k + 1;
            }
        }
        return 1;
    }

    /**
     * Returns summary information about the given matrix <tt>A</tt>. That is a
     * String with (propertyName, propertyValue) pairs. Useful for debugging or
     * to quickly get the rough picture of a matrix. For example,
     * 
     * <pre>
     *   density                      : 0.9
     *   isDiagonal                   : false
     *   isDiagonallyDominantByRow    : false
     *   isDiagonallyDominantByColumn : false
     *   isIdentity                   : false
     *   isLowerBidiagonal            : false
     *   isLowerTriangular            : false
     *   isNonNegative                : true
     *   isOrthogonal                 : Illegal operation or error: Matrix must be square.
     *   isPositive                   : true
     *   isSingular                   : Illegal operation or error: Matrix must be square.
     *   isSkewSymmetric              : Illegal operation or error: Matrix must be square.
     *   isSquare                     : false
     *   isStrictlyLowerTriangular    : false
     *   isStrictlyTriangular         : false
     *   isStrictlyUpperTriangular    : false
     *   isSymmetric                  : Illegal operation or error: Matrix must be square.
     *   isTriangular                 : false
     *   isTridiagonal                : false
     *   isUnitTriangular             : false
     *   isUpperBidiagonal            : false
     *   isUpperTriangular            : false
     *   isZero                       : false
     *   lowerBandwidth               : Illegal operation or error: Matrix must be square.
     *   semiBandwidth                : Illegal operation or error: Matrix must be square.
     *   upperBandwidth               : Illegal operation or error: Matrix must be square.
     * 
     * </pre>
     */
    public String toString(LongMatrix2D A) {
        final cern.colt.list.tobject.ObjectArrayList names = new cern.colt.list.tobject.ObjectArrayList();
        final cern.colt.list.tobject.ObjectArrayList values = new cern.colt.list.tobject.ObjectArrayList();
        String unknown = "Illegal operation or error: ";

        // determine properties
        names.add("density");
        try {
            values.add(String.valueOf(density(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        // determine properties
        names.add("isDiagonal");
        try {
            values.add(String.valueOf(isDiagonal(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        // determine properties
        names.add("isDiagonallyDominantByRow");
        try {
            values.add(String.valueOf(isDiagonallyDominantByRow(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        // determine properties
        names.add("isDiagonallyDominantByColumn");
        try {
            values.add(String.valueOf(isDiagonallyDominantByColumn(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isIdentity");
        try {
            values.add(String.valueOf(isIdentity(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isLowerBidiagonal");
        try {
            values.add(String.valueOf(isLowerBidiagonal(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isLowerTriangular");
        try {
            values.add(String.valueOf(isLowerTriangular(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isNonNegative");
        try {
            values.add(String.valueOf(isNonNegative(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isOrthogonal");
        try {
            values.add(String.valueOf(isOrthogonal(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isPositive");
        try {
            values.add(String.valueOf(isPositive(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        //        names.add("isSingular");
        //        try {
        //            values.add(String.valueOf(isSingular(A)));
        //        } catch (IllegalArgumentException exc) {
        //            values.add(unknown + exc.getMessage());
        //        }

        names.add("isSkewSymmetric");
        try {
            values.add(String.valueOf(isSkewSymmetric(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isSquare");
        try {
            values.add(String.valueOf(isSquare(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isStrictlyLowerTriangular");
        try {
            values.add(String.valueOf(isStrictlyLowerTriangular(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isStrictlyTriangular");
        try {
            values.add(String.valueOf(isStrictlyTriangular(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isStrictlyUpperTriangular");
        try {
            values.add(String.valueOf(isStrictlyUpperTriangular(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isSymmetric");
        try {
            values.add(String.valueOf(isSymmetric(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isTriangular");
        try {
            values.add(String.valueOf(isTriangular(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isTridiagonal");
        try {
            values.add(String.valueOf(isTridiagonal(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isUnitTriangular");
        try {
            values.add(String.valueOf(isUnitTriangular(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isUpperBidiagonal");
        try {
            values.add(String.valueOf(isUpperBidiagonal(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isUpperTriangular");
        try {
            values.add(String.valueOf(isUpperTriangular(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("isZero");
        try {
            values.add(String.valueOf(isZero(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("lowerBandwidth");
        try {
            values.add(String.valueOf(lowerBandwidth(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("semiBandwidth");
        try {
            values.add(String.valueOf(semiBandwidth(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("upperBandwidth");
        try {
            values.add(String.valueOf(upperBandwidth(A)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        // sort ascending by property name
        cern.colt.function.tint.IntComparator comp = new cern.colt.function.tint.IntComparator() {
            public int compare(int a, int b) {
                return get(names, a).compareTo(get(names, b));
            }
        };
        cern.colt.Swapper swapper = new cern.colt.Swapper() {
            public void swap(int a, int b) {
                Object tmp;
                tmp = names.get(a);
                names.set(a, names.get(b));
                names.set(b, tmp);
                tmp = values.get(a);
                values.set(a, values.get(b));
                values.set(b, tmp);
            }
        };
        cern.colt.GenericSorting.quickSort(0, names.size(), comp, swapper);

        // determine padding for nice formatting
        int maxLength = 0;
        for (int i = 0; i < names.size(); i++) {
            int length = ((String) names.get(i)).length();
            maxLength = Math.max(length, maxLength);
        }

        // finally, format properties
        StringBuffer buf = new StringBuffer();
        for (int i = 0; i < names.size(); i++) {
            String name = ((String) names.get(i));
            buf.append(name);
            buf.append(blanks(maxLength - name.length()));
            buf.append(" : ");
            buf.append(values.get(i));
            if (i < names.size() - 1)
                buf.append('\n');
        }

        return buf.toString();
    }

    /**
     * The <i>upper bandwidth</i> of a square matrix <tt>A</tt> is the maximum
     * <tt>j-i</tt> for which <tt>A[i,j]</tt> is nonzero and <tt>j &gt; i</tt>.
     * A <i>banded</i> matrix has a "band" about the diagonal. Diagonal,
     * tridiagonal and triangular matrices are special cases.
     * 
     * @param A
     *            the square matrix to analyze.
     * @return the upper bandwith.
     * @throws IllegalArgumentException
     *             if <tt>!isSquare(A)</tt>.
     * @see #semiBandwidth(LongMatrix2D)
     * @see #lowerBandwidth(LongMatrix2D)
     */
    public int upperBandwidth(LongMatrix2D A) {
        checkSquare(A);
        int rows = A.rows();

        for (int k = rows; --k >= 0;) {
            for (int i = rows - k; --i >= 0;) {
                int j = i + k;
                if (A.getQuick(i, j) != 0)
                    return k;
            }
        }
        return 0;
    }
}
