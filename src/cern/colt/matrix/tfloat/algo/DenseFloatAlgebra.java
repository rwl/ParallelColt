/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.algo;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.Norm;
import cern.colt.matrix.tfloat.FloatFactory2D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatCholeskyDecomposition;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatEigenvalueDecomposition;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatLUDecomposition;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatQRDecomposition;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatSingularValueDecomposition;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix3D;
import cern.colt.matrix.tfloat.impl.SparseCCFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.SparseRCFloatMatrix2D;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import cern.colt.matrix.tint.impl.DenseIntMatrix1D;
import cern.colt.matrix.tint.impl.DenseIntMatrix2D;
import cern.jet.math.tfloat.FloatFunctions;
import cern.jet.math.tint.IntFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Linear algebraic matrix operations operating on dense matrices.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DenseFloatAlgebra extends cern.colt.PersistentObject {
    private static final long serialVersionUID = 1L;

    /**
     * A default Algebra object; has {@link FloatProperty#DEFAULT} attached for
     * tolerance. Allows ommiting to construct an Algebra object time and again.
     * 
     * Note that this Algebra object is immutable. Any attempt to assign a new
     * Property object to it (via method <tt>setProperty</tt>), or to alter the
     * tolerance of its property object (via
     * <tt>property().setTolerance(...)</tt>) will throw an exception.
     */
    public static final DenseFloatAlgebra DEFAULT;

    /**
     * A default Algebra object; has {@link FloatProperty#ZERO} attached for
     * tolerance. Allows ommiting to construct an Algebra object time and again.
     * 
     * Note that this Algebra object is immutable. Any attempt to assign a new
     * Property object to it (via method <tt>setProperty</tt>), or to alter the
     * tolerance of its property object (via
     * <tt>property().setTolerance(...)</tt>) will throw an exception.
     */
    public static final DenseFloatAlgebra ZERO;

    /**
     * The property object attached to this instance.
     */
    protected FloatProperty property;

    static {
        // don't use new Algebra(Property.DEFAULT.tolerance()), because then
        // property object would be mutable.
        DEFAULT = new DenseFloatAlgebra();
        DEFAULT.property = FloatProperty.DEFAULT; // immutable property object

        ZERO = new DenseFloatAlgebra();
        ZERO.property = FloatProperty.ZERO; // immutable property object
    }

    /**
     * Constructs a new instance with an equality tolerance given by
     * <tt>Property.DEFAULT.tolerance()</tt>.
     */
    public DenseFloatAlgebra() {
        this(FloatProperty.DEFAULT.tolerance());
    }

    /**
     * Constructs a new instance with the given equality tolerance.
     * 
     * @param tolerance
     *            the tolerance to be used for equality operations.
     */
    public DenseFloatAlgebra(float tolerance) {
        setProperty(new FloatProperty(tolerance));
    }

    /**
     * Constructs and returns the cholesky-decomposition of the given matrix.
     */
    public DenseFloatCholeskyDecomposition chol(FloatMatrix2D matrix) {
        return new DenseFloatCholeskyDecomposition(matrix);
    }

    /**
     * Returns a copy of the receiver. The attached property object is also
     * copied. Hence, the property object of the copy is mutable.
     * 
     * @return a copy of the receiver.
     */

    public Object clone() {
        return new DenseFloatAlgebra(property.tolerance());
    }

    /**
     * Returns the condition of matrix <tt>A</tt>, which is the ratio of largest
     * to smallest singular value.
     */
    public float cond(FloatMatrix2D A) {
        return svd(A).cond();
    }

    /**
     * Returns the determinant of matrix <tt>A</tt>.
     * 
     * @return the determinant.
     */
    public float det(FloatMatrix2D A) {
        return lu(A).det();
    }

    /**
     * Constructs and returns the Eigenvalue-decomposition of the given matrix.
     */
    public DenseFloatEigenvalueDecomposition eig(FloatMatrix2D matrix) {
        return new DenseFloatEigenvalueDecomposition(matrix);
    }

    /**
     * Returns sqrt(a^2 + b^2) without under/overflow.
     */
    public static float hypot(float a, float b) {
        float r;
        if (Math.abs(a) > Math.abs(b)) {
            r = b / a;
            r = Math.abs(a) * (float) Math.sqrt(1 + r * r);
        } else if (b != 0) {
            r = a / b;
            r = Math.abs(b) * (float) Math.sqrt(1 + r * r);
        } else {
            r = 0;
        }
        return r;
    }

    /**
     * Returns sqrt(a^2 + b^2) without under/overflow.
     */
    public static cern.colt.function.tfloat.FloatFloatFunction hypotFunction() {
        return new cern.colt.function.tfloat.FloatFloatFunction() {
            public final float apply(float a, float b) {
                return hypot(a, b);
            }
        };
    }

    /**
     * Returns the inverse or pseudo-inverse of matrix <tt>A</tt>.
     * 
     * @return a new independent matrix; inverse(matrix) if the matrix is
     *         square, pseudoinverse otherwise.
     */
    public FloatMatrix2D inverse(FloatMatrix2D A) {
        if (property.isSquare(A) && property.isDiagonal(A)) {
            FloatMatrix2D inv = A.copy();
            boolean isNonSingular = true;
            for (int i = inv.rows(); --i >= 0;) {
                float v = inv.getQuick(i, i);
                isNonSingular &= (v != 0);
                inv.setQuick(i, i, 1 / v);
            }
            if (!isNonSingular)
                throw new IllegalArgumentException("A is singular.");
            return inv;
        }
        return solve(A, FloatFactory2D.dense.identity(A.rows()));
    }

    /**
     * Constructs and returns the LU-decomposition of the given matrix.
     */
    public DenseFloatLUDecomposition lu(FloatMatrix2D matrix) {
        return new DenseFloatLUDecomposition(matrix);
    }

    /**
     * Computes the Kronecker product of two real matrices.
     * 
     * @param x
     * @param y
     * @return the Kronecker product of two real matrices
     */
    public FloatMatrix1D kron(final FloatMatrix1D x, final FloatMatrix1D y) {
        final int size_x = (int) x.size();
        final int size_y = (int) y.size();
        final FloatMatrix1D C = new DenseFloatMatrix1D(size_x * size_y);
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size_x >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            ConcurrencyUtils.setThreadsBeginN_1D(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, size_x);
            Future<?>[] futures = new Future[nthreads];
            int k = size_x / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = j * k;
                final int lastIdx = (j == nthreads - 1) ? size_x : firstIdx + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            C.viewPart(i * size_y, size_y).assign(y, FloatFunctions.multSecond(x.getQuick(i)));
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
        } else {
            for (int i = 0; i < size_x; i++) {
                C.viewPart(i * size_y, size_y).assign(y, FloatFunctions.multSecond(x.getQuick(i)));
            }
        }
        return C;
    }

    /**
     * Computes the Kronecker product of two real matrices.
     * 
     * @param X
     * @param Y
     * @return the Kronecker product of two real matrices
     */
    public FloatMatrix2D kron(final FloatMatrix2D X, final FloatMatrix2D Y) {
        final int rows_x = X.rows();
        final int columns_x = X.columns();
        final int rows_y = Y.rows();
        final int columns_y = Y.columns();
        if ((X.getClass().getName().indexOf("Dense", 0) != -1 && Y.getClass().getName().indexOf("Dense", 0) != -1)) {//both are dense 
            final FloatMatrix2D C = new DenseFloatMatrix2D(rows_x * rows_y, columns_x * columns_y);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (X.size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                ConcurrencyUtils.setThreadsBeginN_1D(Integer.MAX_VALUE);
                nthreads = Math.min(nthreads, rows_x);
                Future<?>[] futures = new Future[nthreads];
                int k = rows_x / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows_x : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {
                        public void run() {
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int c = 0; c < columns_x; c++) {
                                    C.viewPart(r * rows_y, c * columns_y, rows_y, columns_y).assign(Y,
                                            FloatFunctions.multSecond(X.getQuick(r, c)));
                                }
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
                ConcurrencyUtils.resetThreadsBeginN();
            } else {
                for (int r = 0; r < rows_x; r++) {
                    for (int c = 0; c < columns_x; c++) {
                        C.viewPart(r * rows_y, c * columns_y, rows_y, columns_y).assign(Y,
                                FloatFunctions.multSecond(X.getQuick(r, c)));
                    }
                }
            }
            return C;
        } else {
            IntArrayList iaList = new IntArrayList();
            IntArrayList jaList = new IntArrayList();
            FloatArrayList saList = new FloatArrayList();
            IntArrayList ibList = new IntArrayList();
            IntArrayList jbList = new IntArrayList();
            FloatArrayList sbList = new FloatArrayList();
            X.getNonZeros(iaList, jaList, saList);
            Y.getNonZeros(ibList, jbList, sbList);
            iaList.trimToSize();
            jaList.trimToSize();
            saList.trimToSize();
            ibList.trimToSize();
            jbList.trimToSize();
            sbList.trimToSize();
            IntMatrix1D ia = new DenseIntMatrix1D(iaList.elements());
            IntMatrix1D ja = new DenseIntMatrix1D(jaList.elements());
            FloatMatrix1D sa = new DenseFloatMatrix1D(saList.elements());
            IntMatrix1D ib = new DenseIntMatrix1D(ibList.elements());
            IntMatrix1D jb = new DenseIntMatrix1D(jbList.elements());
            FloatMatrix1D sb = new DenseFloatMatrix1D(sbList.elements());

            ia.assign(IntFunctions.mult(rows_y));
            IntMatrix2D ik = new DenseIntMatrix2D(sbList.size(), (int) ia.size());
            for (int i = 0; i < sbList.size(); i++) {
                ik.viewRow(i).assign(ia).assign(IntFunctions.plus(ib.getQuick(i)));
            }
            ja.assign(IntFunctions.mult(columns_y));
            IntMatrix2D jk = new DenseIntMatrix2D(sbList.size(), (int) ja.size());
            for (int i = 0; i < sbList.size(); i++) {
                jk.viewRow(i).assign(ja).assign(IntFunctions.plus(jb.getQuick(i)));
            }
            FloatMatrix2D sk = multOuter(sa, sb, null);
            if (X instanceof SparseCCFloatMatrix2D || Y instanceof SparseCCFloatMatrix2D) {
                return new SparseCCFloatMatrix2D(rows_x * rows_y, columns_x * columns_y, (int[]) ik.vectorize()
                        .elements(), (int[]) jk.vectorize().elements(), (float[]) sk.viewDice().vectorize().elements(),
                        false, false, false);
            } else {
                return new SparseRCFloatMatrix2D(rows_x * rows_y, columns_x * columns_y, (int[]) ik.vectorize()
                        .elements(), (int[]) jk.vectorize().elements(), (float[]) sk.viewDice().vectorize().elements(),
                        false, false, false);
            }

        }
    }

    /**
     * Inner product of two vectors; <tt>Sum(x[i] * y[i])</tt>. Also known as
     * dot product. <br>
     * Equivalent to <tt>x.zDotProduct(y)</tt>.
     * 
     * @param x
     *            the first source vector.
     * @param y
     *            the second source matrix.
     * @return the inner product.
     * 
     * @throws IllegalArgumentException
     *             if <tt>x.size() != y.size()</tt>.
     */
    public float mult(FloatMatrix1D x, FloatMatrix1D y) {
        return x.zDotProduct(y);
    }

    /**
     * Linear algebraic matrix-vector multiplication; <tt>z = A * y</tt>.
     * <tt>z[i] = Sum(A[i,j] * y[j]), i=0..A.rows()-1, j=0..y.size()-1</tt>.
     * 
     * @param A
     *            the source matrix.
     * @param y
     *            the source vector.
     * @return <tt>z</tt>; a new vector with <tt>z.size()==A.rows()</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>A.columns() != y.size()</tt>.
     */
    public FloatMatrix1D mult(FloatMatrix2D A, FloatMatrix1D y) {
        return A.zMult(y, null);
    }

    /**
     * Linear algebraic matrix-matrix multiplication; <tt>C = A x B</tt>.
     * <tt>C[i,j] = Sum(A[i,k] * B[k,j]), k=0..n-1</tt>. <br>
     * Matrix shapes: <tt>A(m x n), B(n x p), C(m x p)</tt>.
     * 
     * @param A
     *            the first source matrix.
     * @param B
     *            the second source matrix.
     * @return <tt>C</tt>; a new matrix holding the results, with
     *         <tt>C.rows()=A.rows(), C.columns()==B.columns()</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>B.rows() != A.columns()</tt>.
     */
    public FloatMatrix2D mult(FloatMatrix2D A, FloatMatrix2D B) {
        return A.zMult(B, null);
    }

    /**
     * Outer product of two vectors; Sets <tt>A[i,j] = x[i] * y[j]</tt>.
     * 
     * @param x
     *            the first source vector.
     * @param y
     *            the second source vector.
     * @param A
     *            the matrix to hold the results. Set this parameter to
     *            <tt>null</tt> to indicate that a new result matrix shall be
     *            constructed.
     * @return A (for convenience only).
     * @throws IllegalArgumentException
     *             if <tt>A.rows() != x.size() || A.columns() != y.size()</tt>.
     */
    public FloatMatrix2D multOuter(final FloatMatrix1D x, final FloatMatrix1D y, FloatMatrix2D A) {
        int rows = (int) x.size();
        int columns = (int) y.size();
        final FloatMatrix2D AA;
        if (A == null) {
            AA = x.like2D(rows, columns);
        } else {
            AA = A;
        }
        if (AA.rows() != rows || AA.columns() != columns)
            throw new IllegalArgumentException();

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (rows >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            ConcurrencyUtils.setThreadsBeginN_1D(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int r = firstRow; r < lastRow; r++) {
                            AA.viewRow(r).assign(y);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
        } else {
            for (int r = rows; --r >= 0;) {
                AA.viewRow(r).assign(y);
            }
        }

        if ((nthreads > 1) && (columns >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            ConcurrencyUtils.setThreadsBeginN_1D(Integer.MAX_VALUE);
            ConcurrencyUtils.setThreadsBeginN_1D(Integer.MAX_VALUE);
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstColumn = j * k;
                final int lastColumn = (j == nthreads - 1) ? columns : firstColumn + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            AA.viewColumn(c).assign(x, FloatFunctions.mult);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
            ConcurrencyUtils.resetThreadsBeginN();
        } else {
            for (int c = columns; --c >= 0;) {
                AA.viewColumn(c).assign(x, FloatFunctions.mult);
            }
        }
        return AA;
    }

    /**
     * Returns the one-norm of vector <tt>x</tt>, which is
     * <tt>Sum(abs(x[i]))</tt>.
     */
    public float norm1(FloatMatrix1D x) {
        if (x.size() == 0)
            return 0;
        return x.aggregate(cern.jet.math.tfloat.FloatFunctions.plus, cern.jet.math.tfloat.FloatFunctions.abs);
    }

    /**
     * Returns the one-norm of matrix <tt>A</tt>, which is the maximum absolute
     * column sum.
     */
    public float norm1(FloatMatrix2D A) {
        float max = 0;
        for (int column = A.columns(); --column >= 0;) {
            max = Math.max(max, norm1(A.viewColumn(column)));
        }
        return max;
    }

    /**
     * Returns the two-norm (aka <i>euclidean norm</i>) of vector <tt>x</tt>;
     * equivalent to <tt>Sqrt(mult(x,x))</tt>.
     */
    public float norm2(FloatMatrix1D x) {
        return (float) Math.sqrt(x.zDotProduct(x));
    }

    /**
     * Returns the two-norm (aka <i>euclidean norm</i>) of vector
     * <tt>X.vectorize()</tt>;
     */
    public float vectorNorm2(final FloatMatrix2D X) {
        if (X.isView() == true || !(X instanceof DenseFloatMatrix2D)) {
            final int rows = X.rows();
            final int columns = X.columns();
            float sum = 0;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, rows);
                Future<?>[] futures = new Future[nthreads];
                Float result;
                int k = rows / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstRow = j * k;
                    final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                    futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {
                        public Float call() throws Exception {
                            float sum = 0;
                            float elem;
                            for (int r = firstRow; r < lastRow; r++) {
                                for (int c = 0; c < columns; c++) {
                                    elem = X.getQuick(r, c);
                                    sum += (elem * elem);
                                }
                            }
                            return sum;
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        result = (Float) futures[j].get();
                        sum += result;
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                float elem;
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        elem = X.getQuick(r, c);
                        sum += (elem * elem);
                    }
                }
            }
            return (float) Math.sqrt(sum);
        } else {
            final float[] elems = ((DenseFloatMatrix2D) X).elements();
            float sum = 0;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (elems.length >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, elems.length);
                Future<?>[] futures = new Future[nthreads];
                Float result;
                int k = elems.length / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == nthreads - 1) ? elems.length : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {
                        public Float call() throws Exception {
                            float sum = 0;
                            for (int l = firstIdx; l < lastIdx; l++) {
                                sum += (elems[l] * elems[l]);
                            }
                            return sum;
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        result = (Float) futures[j].get();
                        sum += result;
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                for (int l = 0; l < elems.length; l++) {
                    sum += (elems[l] * elems[l]);
                }
            }
            return (float) Math.sqrt(sum);

        }
    }

    /**
     * Returns the two-norm (aka <i>euclidean norm</i>) of vector
     * <tt>X.vectorize()</tt>;
     */
    public float vectorNorm2(final FloatMatrix3D X) {
        if (X.isView() == true || !(X instanceof DenseFloatMatrix3D)) {
            final int slices = X.slices();
            final int rows = X.rows();
            final int columns = X.columns();
            float sum = 0;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, slices);
                Future<?>[] futures = new Future[nthreads];
                Float result;
                int k = slices / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstSlice = j * k;
                    final int lastSlice = (j == nthreads - 1) ? slices : firstSlice + k;
                    futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {
                        public Float call() throws Exception {
                            float sum = 0;
                            float elem;
                            for (int s = firstSlice; s < lastSlice; s++) {
                                for (int r = 0; r < rows; r++) {
                                    for (int c = 0; c < columns; c++) {
                                        elem = X.getQuick(s, r, c);
                                        sum += (elem * elem);
                                    }
                                }
                            }
                            return sum;
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        result = (Float) futures[j].get();
                        sum += result;
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                float elem;
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < columns; c++) {
                            elem = X.getQuick(s, r, c);
                            sum += (elem * elem);
                        }
                    }
                }
            }
            return (float) Math.sqrt(sum);
        } else {
            final float[] elems = ((DenseFloatMatrix3D) X).elements();
            float sum = 0;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (elems.length >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, elems.length);
                Future<?>[] futures = new Future[nthreads];
                Float result;
                int k = elems.length / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == nthreads - 1) ? elems.length : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Callable<Float>() {
                        public Float call() throws Exception {
                            float sum = 0;
                            for (int l = firstIdx; l < lastIdx; l++) {
                                sum += (elems[l] * elems[l]);
                            }
                            return sum;
                        }
                    });
                }
                try {
                    for (int j = 0; j < nthreads; j++) {
                        result = (Float) futures[j].get();
                        sum += result;
                    }
                } catch (ExecutionException ex) {
                    ex.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                for (int l = 0; l < elems.length; l++) {
                    sum += (elems[l] * elems[l]);
                }
            }
            return (float) Math.sqrt(sum);

        }
    }

    public float norm(FloatMatrix2D A, Norm type) {
        switch (type) {
        case Frobenius:
            return DEFAULT.normF(A);
        case Infinity:
            return DEFAULT.normInfinity(A);
        case One:
            return DEFAULT.norm1(A);
        case Two:
            return DEFAULT.norm2(A);
        default:
            return 0;
        }

    }

    public float norm(FloatMatrix1D x, Norm type) {
        switch (type) {
        case Frobenius:
            return DEFAULT.normF(x);
        case Infinity:
            return DEFAULT.normInfinity(x);
        case One:
            return DEFAULT.norm1(x);
        case Two:
            return DEFAULT.norm2(x);
        default:
            return 0;
        }

    }

    /**
     * Returns the two-norm of matrix <tt>A</tt>, which is the maximum singular
     * value; obtained from SVD.
     */
    public float norm2(FloatMatrix2D A) {
        return svd(A).norm2();
    }

    /**
     * Returns the Frobenius norm of matrix <tt>A</tt>, which is
     * <tt>Sqrt(Sum(A[i,j]<sup>2</sup>))</tt>.
     */
    public float normF(FloatMatrix2D A) {
        if (A.size() == 0)
            return 0;
        return A.aggregate(hypotFunction(), cern.jet.math.tfloat.FloatFunctions.identity);
    }

    /**
     * Returns the Frobenius norm of matrix <tt>A</tt>, which is
     * <tt>Sqrt(Sum(A[i]<sup>2</sup>))</tt>.
     */
    public float normF(FloatMatrix1D A) {
        if (A.size() == 0)
            return 0;
        return A.aggregate(hypotFunction(), cern.jet.math.tfloat.FloatFunctions.identity);
    }

    /**
     * Returns the infinity norm of vector <tt>x</tt>, which is
     * <tt>Max(abs(x[i]))</tt>.
     */
    public float normInfinity(FloatMatrix1D x) {
        if (x.size() == 0)
            return 0;
        return x.aggregate(cern.jet.math.tfloat.FloatFunctions.max, cern.jet.math.tfloat.FloatFunctions.abs);
    }

    /**
     * Returns the infinity norm of matrix <tt>A</tt>, which is the maximum
     * absolute row sum.
     */
    public float normInfinity(FloatMatrix2D A) {
        float max = 0;
        for (int row = A.rows(); --row >= 0;) {
            max = Math.max(max, norm1(A.viewRow(row)));
        }
        return max;
    }

    /**
     * Modifies the given vector <tt>A</tt> such that it is permuted as
     * specified; Useful for pivoting. Cell <tt>A[i]</tt> will go into cell
     * <tt>A[indexes[i]]</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * 	 Reordering
     * 	 [A,B,C,D,E] with indexes [0,4,2,3,1] yields 
     * 	 [A,E,C,D,B]
     * 	 In other words A[0]&lt;--A[0], A[1]&lt;--A[4], A[2]&lt;--A[2], A[3]&lt;--A[3], A[4]&lt;--A[1].
     * 
     * 	 Reordering
     * 	 [A,B,C,D,E] with indexes [0,4,1,2,3] yields 
     * 	 [A,E,B,C,D]
     * 	 In other words A[0]&lt;--A[0], A[1]&lt;--A[4], A[2]&lt;--A[1], A[3]&lt;--A[2], A[4]&lt;--A[3].
     * 
     * </pre>
     * 
     * @param A
     *            the vector to permute.
     * @param indexes
     *            the permutation indexes, must satisfy
     *            <tt>indexes.length==A.size() && indexes[i] >= 0 && indexes[i] < A.size()</tt>
     *            ;
     * @param work
     *            the working storage, must satisfy
     *            <tt>work.length >= A.size()</tt>; set <tt>work==null</tt> if
     *            you don't care about performance.
     * @return the modified <tt>A</tt> (for convenience only).
     * @throws IndexOutOfBoundsException
     *             if <tt>indexes.length != A.size()</tt>.
     */
    public FloatMatrix1D permute(FloatMatrix1D A, int[] indexes, float[] work) {
        // check validity
        int size = (int) A.size();
        if (indexes.length != size)
            throw new IndexOutOfBoundsException("invalid permutation");

        if (work == null || size > work.length) {
            work = A.toArray();
        } else {
            A.toArray(work);
        }
        for (int i = size; --i >= 0;)
            A.setQuick(i, work[indexes[i]]);
        return A;
    }

    /**
     * Constructs and returns a new row and column permuted <i>selection
     * view</i> of matrix <tt>A</tt>; equivalent to
     * {@link FloatMatrix2D#viewSelection(int[],int[])}. The returned matrix is
     * backed by this matrix, so changes in the returned matrix are reflected in
     * this matrix, and vice-versa. Use idioms like
     * <tt>result = permute(...).copy()</tt> to generate an independent sub
     * matrix.
     * 
     * @return the new permuted selection view.
     */
    public FloatMatrix2D permute(FloatMatrix2D A, int[] rowIndexes, int[] columnIndexes) {
        return A.viewSelection(rowIndexes, columnIndexes);
    }

    /**
     * Modifies the given matrix <tt>A</tt> such that it's columns are permuted
     * as specified; Useful for pivoting. Column <tt>A[i]</tt> will go into
     * column <tt>A[indexes[i]]</tt>. Equivalent to
     * <tt>permuteRows(transpose(A), indexes, work)</tt>.
     * 
     * @param A
     *            the matrix to permute.
     * @param indexes
     *            the permutation indexes, must satisfy
     *            <tt>indexes.length==A.columns() && indexes[i] >= 0 && indexes[i] < A.columns()</tt>
     *            ;
     * @param work
     *            the working storage, must satisfy
     *            <tt>work.length >= A.columns()</tt>; set <tt>work==null</tt>
     *            if you don't care about performance.
     * @return the modified <tt>A</tt> (for convenience only).
     * @throws IndexOutOfBoundsException
     *             if <tt>indexes.length != A.columns()</tt>.
     */
    public FloatMatrix2D permuteColumns(FloatMatrix2D A, int[] indexes, int[] work) {
        return permuteRows(A.viewDice(), indexes, work);
    }

    /**
     * Modifies the given matrix <tt>A</tt> such that it's rows are permuted as
     * specified; Useful for pivoting. Row <tt>A[i]</tt> will go into row
     * <tt>A[indexes[i]]</tt>.
     * <p>
     * <b>Example:</b>
     * 
     * <pre>
     * 	 Reordering
     * 	 [A,B,C,D,E] with indexes [0,4,2,3,1] yields 
     * 	 [A,E,C,D,B]
     * 	 In other words A[0]&lt;--A[0], A[1]&lt;--A[4], A[2]&lt;--A[2], A[3]&lt;--A[3], A[4]&lt;--A[1].
     * 
     * 	 Reordering
     * 	 [A,B,C,D,E] with indexes [0,4,1,2,3] yields 
     * 	 [A,E,B,C,D]
     * 	 In other words A[0]&lt;--A[0], A[1]&lt;--A[4], A[2]&lt;--A[1], A[3]&lt;--A[2], A[4]&lt;--A[3].
     * 
     * </pre>
     * 
     * @param A
     *            the matrix to permute.
     * @param indexes
     *            the permutation indexes, must satisfy
     *            <tt>indexes.length==A.rows() && indexes[i] >= 0 && indexes[i] < A.rows()</tt>
     *            ;
     * @param work
     *            the working storage, must satisfy
     *            <tt>work.length >= A.rows()</tt>; set <tt>work==null</tt> if
     *            you don't care about performance.
     * @return the modified <tt>A</tt> (for convenience only).
     * @throws IndexOutOfBoundsException
     *             if <tt>indexes.length != A.rows()</tt>.
     */
    public FloatMatrix2D permuteRows(final FloatMatrix2D A, int[] indexes, int[] work) {
        // check validity
        int size = A.rows();
        if (indexes.length != size)
            throw new IndexOutOfBoundsException("invalid permutation");

        int columns = A.columns();
        if (columns < size / 10) { // quicker
            float[] floatWork = new float[size];
            for (int j = A.columns(); --j >= 0;)
                permute(A.viewColumn(j), indexes, floatWork);
            return A;
        }

        cern.colt.Swapper swapper = new cern.colt.Swapper() {
            public void swap(int a, int b) {
                A.viewRow(a).swap(A.viewRow(b));
            }
        };

        cern.colt.GenericPermuting.permute(indexes, swapper, work, null);
        return A;
    }

    /**
     * Linear algebraic matrix power;
     * <tt>B = A<sup>k</sup> <==> B = A*A*...*A</tt>.
     * <ul>
     * <li><tt>p &gt;= 1: B = A*A*...*A</tt>.</li>
     * <li><tt>p == 0: B = identity matrix</tt>.</li>
     * <li><tt>p &lt;  0: B = pow(inverse(A),-p)</tt>.</li>
     * </ul>
     * Implementation: Based on logarithms of 2, memory usage minimized.
     * 
     * @param A
     *            the source matrix; must be square; stays unaffected by this
     *            operation.
     * @param p
     *            the exponent, can be any number.
     * @return <tt>B</tt>, a newly constructed result matrix;
     *         storage-independent of <tt>A</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>!property().isSquare(A)</tt>.
     */
    public FloatMatrix2D pow(FloatMatrix2D A, int p) {
        // matrix multiplication based on log2 method: A*A*....*A is slow, ((A *
        // A)^2)^2 * ... is faster
        // allocates two auxiliary matrices as work space

        FloatBlas blas = new SmpFloatBlas(); // for parallel matrix mult; if
        // not
        // initialized defaults to sequential blas
        FloatProperty.DEFAULT.checkSquare(A);
        if (p < 0) {
            A = inverse(A);
            p = -p;
        }
        if (p == 0)
            return FloatFactory2D.dense.identity(A.rows());
        FloatMatrix2D T = A.like(); // temporary
        if (p == 1)
            return T.assign(A); // safes one auxiliary matrix allocation
        if (p == 2) {
            blas.dgemm(false, false, 1, A, A, 0, T); // mult(A,A); // safes
            // one auxiliary matrix
            // allocation
            return T;
        }

        int k = cern.colt.matrix.tbit.QuickBitVector.mostSignificantBit(p);
        /* index of highest bit in state "true" */

        /*
         * this is the naive version: FloatMatrix2D B = A.copy(); for (int i=0;
         * i<p-1; i++) { B = mult(B,A); } return B;
         */

        // here comes the optimized version:
        // cern.colt.Timer timer = new cern.colt.Timer().start();
        int i = 0;
        while (i <= k && (p & (1 << i)) == 0) { // while (bit i of p == false)
            // A = mult(A,A); would allocate a lot of temporary memory
            blas.dgemm(false, false, 1, A, A, 0, T); // A.zMult(A,T);
            FloatMatrix2D swap = A;
            A = T;
            T = swap; // swap A with T
            i++;
        }

        FloatMatrix2D B = A.copy();
        i++;
        for (; i <= k; i++) {
            // A = mult(A,A); would allocate a lot of temporary memory
            blas.dgemm(false, false, 1, A, A, 0, T); // A.zMult(A,T);
            FloatMatrix2D swap = A;
            A = T;
            T = swap; // swap A with T

            if ((p & (1 << i)) != 0) { // if (bit i of p == true)
                // B = mult(B,A); would allocate a lot of temporary memory
                blas.dgemm(false, false, 1, B, A, 0, T); // B.zMult(A,T);
                swap = B;
                B = T;
                T = swap; // swap B with T
            }
        }
        // timer.stop().display();
        return B;
    }

    /**
     * Returns the property object attached to this Algebra, defining tolerance.
     * 
     * @return the Property object.
     * @see #setProperty(FloatProperty)
     */
    public FloatProperty property() {
        return property;
    }

    /**
     * Constructs and returns the QR-decomposition of the given matrix.
     */
    public DenseFloatQRDecomposition qr(FloatMatrix2D matrix) {
        return new DenseFloatQRDecomposition(matrix);
    }

    /**
     * Returns the effective numerical rank of matrix <tt>A</tt>, obtained from
     * Singular Value Decomposition.
     */
    public int rank(FloatMatrix2D A) {
        return svd(A).rank();
    }

    /**
     * Attaches the given property object to this Algebra, defining tolerance.
     * 
     * @param property
     *            the Property object to be attached.
     * @throws UnsupportedOperationException
     *             if <tt>this==DEFAULT && property!=this.property()</tt> - The
     *             DEFAULT Algebra object is immutable.
     * @throws UnsupportedOperationException
     *             if <tt>this==ZERO && property!=this.property()</tt> - The
     *             ZERO Algebra object is immutable.
     * @see #property
     */
    public void setProperty(FloatProperty property) {
        if (this == DEFAULT && property != this.property)
            throw new IllegalArgumentException("Attempted to modify immutable object.");
        if (this == ZERO && property != this.property)
            throw new IllegalArgumentException("Attempted to modify immutable object.");
        this.property = property;
    }

    /**
     * Solves the upper triangular system U*x=b;
     * 
     * @param U
     *            upper triangular matrix
     * @param b
     *            right-hand side
     * @return x, a new independent matrix;
     */
    public FloatMatrix1D backwardSolve(final FloatMatrix2D U, final FloatMatrix1D b) {
        final int rows = U.rows();
        final FloatMatrix1D x = b.like();
        x.setQuick(rows - 1, b.getQuick(rows - 1) / U.getQuick(rows - 1, rows - 1));
        float sum;
        for (int r = rows - 2; r >= 0; r--) {
            sum = U.viewRow(r).zDotProduct(x);
            x.setQuick(r, (b.getQuick(r) - sum) / U.getQuick(r, r));
        }
        return x;
    }

    /**
     * Solves the lower triangular system U*x=b;
     * 
     * @param L
     *            lower triangular matrix
     * @param b
     *            right-hand side
     * @return x, a new independent matrix;
     */
    public FloatMatrix1D forwardSolve(final FloatMatrix2D L, final FloatMatrix1D b) {
        final int rows = L.rows();
        final FloatMatrix1D x = b.like();
        float sum;
        x.setQuick(0, b.getQuick(0) / L.getQuick(0, 0));
        for (int r = 1; r < rows; r++) {
            sum = L.viewRow(r).zDotProduct(x);
            x.setQuick(r, (b.getQuick(r) - sum) / L.getQuick(r, r));
        }
        return x;
    }

    /**
     * Solves A*x = b.
     * 
     * @return x; a new independent matrix; solution if A is square, least
     *         squares solution otherwise.
     */
    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b) {
        if (A.rows() == A.columns()) {
            return lu(A).solve(b);
        } else {
            FloatMatrix1D x = b.copy();
            qr(A).solve(x);
            return x.viewPart(0, A.columns()).copy();
        }
    }

    /**
     * Solves A*X = B.
     * 
     * @return X; a new independent matrix; solution if A is square, least
     *         squares solution otherwise.
     */
    public FloatMatrix2D solve(FloatMatrix2D A, FloatMatrix2D B) {
        if (A.rows() == A.columns()) {
            return lu(A).solve(B);
        } else {
            FloatMatrix2D X = B.copy();
            qr(A).solve(X);
            return X.viewPart(0, 0, A.columns(), B.columns()).copy();
        }
    }

    /**
     * Solves X*A = B, which is also A'*X' = B'.
     * 
     * @return X; a new independent matrix; solution if A is square, least
     *         squares solution otherwise.
     */
    public FloatMatrix2D solveTranspose(FloatMatrix2D A, FloatMatrix2D B) {
        return solve(transpose(A), transpose(B));
    }

    /**
     * Copies the columns of the indicated rows into a new sub matrix.
     * 
     * <tt>sub[0..rowIndexes.length-1,0..columnTo-columnFrom] = A[rowIndexes(:),columnFrom..columnTo]</tt>
     * ; The returned matrix is <i>not backed</i> by this matrix, so changes in
     * the returned matrix are <i>not reflected</i> in this matrix, and
     * vice-versa.
     * 
     * @param A
     *            the source matrix to copy from.
     * @param rowIndexes
     *            the indexes of the rows to copy. May be unsorted.
     * @param columnFrom
     *            the index of the first column to copy (inclusive).
     * @param columnTo
     *            the index of the last column to copy (inclusive).
     * @return a new sub matrix; with
     *         <tt>sub.rows()==rowIndexes.length; sub.columns()==columnTo-columnFrom+1</tt>
     *         .
     * @throws IndexOutOfBoundsException
     *             if
     * 
     *             <tt>columnFrom<0 || columnTo-columnFrom+1<0 || columnTo+1>matrix.columns() || for any row=rowIndexes[i]: row < 0 || row >= matrix.rows()</tt>
     *             .
     */
    public FloatMatrix2D subMatrix(FloatMatrix2D A, int[] rowIndexes, int columnFrom, int columnTo) {
        int width = columnTo - columnFrom + 1;
        int rows = A.rows();
        A = A.viewPart(0, columnFrom, rows, width);
        FloatMatrix2D sub = A.like(rowIndexes.length, width);

        for (int r = rowIndexes.length; --r >= 0;) {
            int row = rowIndexes[r];
            if (row < 0 || row >= rows)
                throw new IndexOutOfBoundsException("Illegal Index");
            sub.viewRow(r).assign(A.viewRow(row));
        }
        return sub;
    }

    /**
     * Copies the rows of the indicated columns into a new sub matrix.
     * 
     * <tt>sub[0..rowTo-rowFrom,0..columnIndexes.length-1] = A[rowFrom..rowTo,columnIndexes(:)]</tt>
     * ; The returned matrix is <i>not backed</i> by this matrix, so changes in
     * the returned matrix are <i>not reflected</i> in this matrix, and
     * vice-versa.
     * 
     * @param A
     *            the source matrix to copy from.
     * @param rowFrom
     *            the index of the first row to copy (inclusive).
     * @param rowTo
     *            the index of the last row to copy (inclusive).
     * @param columnIndexes
     *            the indexes of the columns to copy. May be unsorted.
     * @return a new sub matrix; with
     *         <tt>sub.rows()==rowTo-rowFrom+1; sub.columns()==columnIndexes.length</tt>
     *         .
     * @throws IndexOutOfBoundsException
     *             if
     * 
     *             <tt>rowFrom<0 || rowTo-rowFrom+1<0 || rowTo+1>matrix.rows() || for any col=columnIndexes[i]: col < 0 || col >= matrix.columns()</tt>
     *             .
     */
    public FloatMatrix2D subMatrix(FloatMatrix2D A, int rowFrom, int rowTo, int[] columnIndexes) {
        if (rowTo - rowFrom >= A.rows())
            throw new IndexOutOfBoundsException("Too many rows");
        int height = rowTo - rowFrom + 1;
        int columns = A.columns();
        A = A.viewPart(rowFrom, 0, height, columns);
        FloatMatrix2D sub = A.like(height, columnIndexes.length);

        for (int c = columnIndexes.length; --c >= 0;) {
            int column = columnIndexes[c];
            if (column < 0 || column >= columns)
                throw new IndexOutOfBoundsException("Illegal Index");
            sub.viewColumn(c).assign(A.viewColumn(column));
        }
        return sub;
    }

    /**
     * Constructs and returns a new <i>sub-range view</i> which is the sub
     * matrix <tt>A[fromRow..toRow,fromColumn..toColumn]</tt>. The returned
     * matrix is backed by this matrix, so changes in the returned matrix are
     * reflected in this matrix, and vice-versa. Use idioms like
     * <tt>result = subMatrix(...).copy()</tt> to generate an independent sub
     * matrix.
     * 
     * @param A
     *            the source matrix.
     * @param fromRow
     *            The index of the first row (inclusive).
     * @param toRow
     *            The index of the last row (inclusive).
     * @param fromColumn
     *            The index of the first column (inclusive).
     * @param toColumn
     *            The index of the last column (inclusive).
     * @return a new sub-range view.
     * @throws IndexOutOfBoundsException
     *             if
     * 
     *             <tt>fromColumn<0 || toColumn-fromColumn+1<0 || toColumn>=A.columns() || fromRow<0 || toRow-fromRow+1<0 || toRow>=A.rows()</tt>
     */
    public FloatMatrix2D subMatrix(FloatMatrix2D A, int fromRow, int toRow, int fromColumn, int toColumn) {
        return A.viewPart(fromRow, fromColumn, toRow - fromRow + 1, toColumn - fromColumn + 1);
    }

    /**
     * Constructs and returns the SingularValue-decomposition of the given
     * matrix.
     */
    public DenseFloatSingularValueDecomposition svd(FloatMatrix2D matrix) {
        return new DenseFloatSingularValueDecomposition(matrix, true, true);
    }

    /**
     * Returns a String with (propertyName, propertyValue) pairs. Useful for
     * debugging or to quickly get the rough picture. For example,
     * 
     * <pre>
     * 	 cond          : 14.073264490042144
     * 	 det           : Illegal operation or error: Matrix must be square.
     * 	 norm1         : 0.9620244354009628
     * 	 norm2         : 3.0
     * 	 normF         : 1.304841791648992
     * 	 normInfinity  : 1.5406551198102534
     * 	 rank          : 3
     * 	 trace         : 0
     * 
     * </pre>
     */
    public String toString(FloatMatrix2D matrix) {
        final cern.colt.list.tobject.ObjectArrayList names = new cern.colt.list.tobject.ObjectArrayList();
        final cern.colt.list.tobject.ObjectArrayList values = new cern.colt.list.tobject.ObjectArrayList();
        String unknown = "Illegal operation or error: ";

        // determine properties
        names.add("cond");
        try {
            values.add(String.valueOf(cond(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("det");
        try {
            values.add(String.valueOf(det(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("norm1");
        try {
            values.add(String.valueOf(norm1(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("norm2");
        try {
            values.add(String.valueOf(norm2(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("normF");
        try {
            values.add(String.valueOf(normF(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("normInfinity");
        try {
            values.add(String.valueOf(normInfinity(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("rank");
        try {
            values.add(String.valueOf(rank(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        names.add("trace");
        try {
            values.add(String.valueOf(trace(matrix)));
        } catch (IllegalArgumentException exc) {
            values.add(unknown + exc.getMessage());
        }

        // sort ascending by property name
        cern.colt.function.tint.IntComparator comp = new cern.colt.function.tint.IntComparator() {
            public int compare(int a, int b) {
                return FloatProperty.get(names, a).compareTo(FloatProperty.get(names, b));
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
            buf.append(FloatProperty.blanks(maxLength - name.length()));
            buf.append(" : ");
            buf.append(values.get(i));
            if (i < names.size() - 1)
                buf.append('\n');
        }

        return buf.toString();
    }

    /**
     * Returns the results of <tt>toString(A)</tt> and additionally the results
     * of all sorts of decompositions applied to the given matrix. Useful for
     * debugging or to quickly get the rough picture. For example,
     * 
     * <pre>
     * 	 A = 3 x 3 matrix
     * 	 249  66  68
     * 	 104 214 108
     * 	 144 146 293
     * 
     * 	 cond         : 3.931600417472078
     * 	 det          : 9638870.0
     * 	 norm1        : 497.0
     * 	 norm2        : 473.34508217011404
     * 	 normF        : 516.873292016525
     * 	 normInfinity : 583.0
     * 	 rank         : 3
     * 	 trace        : 756.0
     * 
     * 	 density                      : 1.0
     * 	 isDiagonal                   : false
     * 	 isDiagonallyDominantByColumn : true
     * 	 isDiagonallyDominantByRow    : true
     * 	 isIdentity                   : false
     * 	 isLowerBidiagonal            : false
     * 	 isLowerTriangular            : false
     * 	 isNonNegative                : true
     * 	 isOrthogonal                 : false
     * 	 isPositive                   : true
     * 	 isSingular                   : false
     * 	 isSkewSymmetric              : false
     * 	 isSquare                     : true
     * 	 isStrictlyLowerTriangular    : false
     * 	 isStrictlyTriangular         : false
     * 	 isStrictlyUpperTriangular    : false
     * 	 isSymmetric                  : false
     * 	 isTriangular                 : false
     * 	 isTridiagonal                : false
     * 	 isUnitTriangular             : false
     * 	 isUpperBidiagonal            : false
     * 	 isUpperTriangular            : false
     * 	 isZero                       : false
     * 	 lowerBandwidth               : 2
     * 	 semiBandwidth                : 3
     * 	 upperBandwidth               : 2
     * 
     * 	 -----------------------------------------------------------------------------
     * 	 LUDecompositionQuick(A) --&gt; isNonSingular(A), det(A), pivot, L, U, inverse(A)
     * 	 -----------------------------------------------------------------------------
     * 	 isNonSingular = true
     * 	 det = 9638870.0
     * 	 pivot = [0, 1, 2]
     * 
     * 	 L = 3 x 3 matrix
     * 	 1        0       0
     * 	 0.417671 1       0
     * 	 0.578313 0.57839 1
     * 
     * 	 U = 3 x 3 matrix
     * 	 249  66         68       
     * 	 0 186.433735  79.598394
     * 	 0   0        207.635819
     * 
     * 	 inverse(A) = 3 x 3 matrix
     * 	 0.004869 -0.000976 -0.00077 
     * 	 -0.001548  0.006553 -0.002056
     * 	 -0.001622 -0.002786  0.004816
     * 
     * 	 -----------------------------------------------------------------
     * 	 QRDecomposition(A) --&gt; hasFullRank(A), H, Q, R, pseudo inverse(A)
     * 	 -----------------------------------------------------------------
     * 	 hasFullRank = true
     * 
     * 	 H = 3 x 3 matrix
     * 	 1.814086 0        0
     * 	 0.34002  1.903675 0
     * 	 0.470797 0.428218 2
     * 
     * 	 Q = 3 x 3 matrix
     * 	 -0.814086  0.508871  0.279845
     * 	 -0.34002  -0.808296  0.48067 
     * 	 -0.470797 -0.296154 -0.831049
     * 
     * 	 R = 3 x 3 matrix
     * 	 -305.864349 -195.230337 -230.023539
     * 	 0        -182.628353  467.703164
     * 	 0           0        -309.13388 
     * 
     * 	 pseudo inverse(A) = 3 x 3 matrix
     * 	 0.006601  0.001998 -0.005912
     * 	 -0.005105  0.000444  0.008506
     * 	 -0.000905 -0.001555  0.002688
     * 
     * 	 --------------------------------------------------------------------------
     * 	 CholeskyDecomposition(A) --&gt; isSymmetricPositiveDefinite(A), L, inverse(A)
     * 	 --------------------------------------------------------------------------
     * 	 isSymmetricPositiveDefinite = false
     * 
     * 	 L = 3 x 3 matrix
     * 	 15.779734  0         0       
     * 	 6.590732 13.059948  0       
     * 	 9.125629  6.573948 12.903724
     * 
     * 	 inverse(A) = Illegal operation or error: Matrix is not symmetric positive definite.
     * 
     * 	 ---------------------------------------------------------------------
     * 	 EigenvalueDecomposition(A) --&gt; D, V, realEigenvalues, imagEigenvalues
     * 	 ---------------------------------------------------------------------
     * 	 realEigenvalues = 1 x 3 matrix
     * 	 462.796507 172.382058 120.821435
     * 	 imagEigenvalues = 1 x 3 matrix
     * 	 0 0 0
     * 
     * 	 D = 3 x 3 matrix
     * 	 462.796507   0          0       
     * 	 0        172.382058   0       
     * 	 0          0        120.821435
     * 
     * 	 V = 3 x 3 matrix
     * 	 -0.398877 -0.778282  0.094294
     * 	 -0.500327  0.217793 -0.806319
     * 	 -0.768485  0.66553   0.604862
     * 
     * 	 ---------------------------------------------------------------------
     * 	 SingularValueDecomposition(A) --&gt; cond(A), rank(A), norm2(A), U, S, V
     * 	 ---------------------------------------------------------------------
     * 	 cond = 3.931600417472078
     * 	 rank = 3
     * 	 norm2 = 473.34508217011404
     * 
     * 	 U = 3 x 3 matrix
     * 	 0.46657  -0.877519  0.110777
     * 	 0.50486   0.161382 -0.847982
     * 	 0.726243  0.45157   0.51832 
     * 
     * 	 S = 3 x 3 matrix
     * 	 473.345082   0          0       
     * 	 0        169.137441   0       
     * 	 0          0        120.395013
     * 
     * 	 V = 3 x 3 matrix
     * 	 0.577296 -0.808174  0.116546
     * 	 0.517308  0.251562 -0.817991
     * 	 0.631761  0.532513  0.563301
     * 
     * </pre>
     */
    public String toVerboseString(FloatMatrix2D matrix) {

        String constructionException = "Illegal operation or error upon construction of ";
        StringBuffer buf = new StringBuffer();

        buf.append("A = ");
        buf.append(matrix);

        buf.append("\n\n" + toString(matrix));
        buf.append("\n\n" + FloatProperty.DEFAULT.toString(matrix));

        DenseFloatLUDecomposition lu = null;
        try {
            lu = new DenseFloatLUDecomposition(matrix);
        } catch (IllegalArgumentException exc) {
            buf.append("\n\n" + constructionException + " LUDecomposition: " + exc.getMessage());
        }
        if (lu != null)
            buf.append("\n\n" + lu.toString());

        DenseFloatQRDecomposition qr = null;
        try {
            qr = new DenseFloatQRDecomposition(matrix);
        } catch (IllegalArgumentException exc) {
            buf.append("\n\n" + constructionException + " QRDecomposition: " + exc.getMessage());
        }
        if (qr != null)
            buf.append("\n\n" + qr.toString());

        DenseFloatCholeskyDecomposition chol = null;
        try {
            chol = new DenseFloatCholeskyDecomposition(matrix);
        } catch (IllegalArgumentException exc) {
            buf.append("\n\n" + constructionException + " CholeskyDecomposition: " + exc.getMessage());
        }
        if (chol != null)
            buf.append("\n\n" + chol.toString());

        DenseFloatEigenvalueDecomposition eig = null;
        try {
            eig = new DenseFloatEigenvalueDecomposition(matrix);
        } catch (IllegalArgumentException exc) {
            buf.append("\n\n" + constructionException + " EigenvalueDecomposition: " + exc.getMessage());
        }
        if (eig != null)
            buf.append("\n\n" + eig.toString());

        DenseFloatSingularValueDecomposition svd = null;
        try {
            svd = new DenseFloatSingularValueDecomposition(matrix, true, true);
        } catch (IllegalArgumentException exc) {
            buf.append("\n\n" + constructionException + " SingularValueDecomposition: " + exc.getMessage());
        }
        if (svd != null)
            buf.append("\n\n" + svd.toString());

        return buf.toString();
    }

    /**
     * Returns the sum of the diagonal elements of matrix <tt>A</tt>;
     * <tt>Sum(A[i,i])</tt>.
     */
    public float trace(FloatMatrix2D A) {
        float sum = 0;
        for (int i = Math.min(A.rows(), A.columns()); --i >= 0;) {
            sum += A.getQuick(i, i);
        }
        return sum;
    }

    /**
     * Constructs and returns a new view which is the transposition of the given
     * matrix <tt>A</tt>. Equivalent to {@link FloatMatrix2D#viewDice
     * A.viewDice()}. This is a zero-copy transposition, taking O(1), i.e.
     * constant time. The returned view is backed by this matrix, so changes in
     * the returned view are reflected in this matrix, and vice-versa. Use
     * idioms like <tt>result = transpose(A).copy()</tt> to generate an
     * independent matrix.
     * <p>
     * <b>Example:</b>
     * <table border="0">
     * <tr nowrap>
     * <td valign="top">2 x 3 matrix: <br>
     * 1, 2, 3<br>
     * 4, 5, 6</td>
     * <td>transpose ==></td>
     * <td valign="top">3 x 2 matrix:<br>
     * 1, 4 <br>
     * 2, 5 <br>
     * 3, 6</td>
     * <td>transpose ==></td>
     * <td valign="top">2 x 3 matrix: <br>
     * 1, 2, 3<br>
     * 4, 5, 6</td>
     * </tr>
     * </table>
     * 
     * @return a new transposed view.
     */
    public FloatMatrix2D transpose(FloatMatrix2D A) {
        return A.viewDice();
    }

    /**
     * Modifies the matrix to be a lower trapezoidal matrix.
     * 
     * @return <tt>A</tt> (for convenience only).
     * 
     */
    public FloatMatrix2D trapezoidalLower(FloatMatrix2D A) {
        int rows = A.rows();
        int columns = A.columns();
        for (int r = rows; --r >= 0;) {
            for (int c = columns; --c >= 0;) {
                if (r < c)
                    A.setQuick(r, c, 0);
            }
        }
        return A;
    }

    /**
     * Outer product of two vectors; Returns a matrix with
     * <tt>A[i,j] = x[i] * y[j]</tt>.
     * 
     * @param x
     *            the first source vector.
     * @param y
     *            the second source vector.
     * @return the outer product </tt>A</tt>.
     */
    public FloatMatrix2D xmultOuter(FloatMatrix1D x, FloatMatrix1D y) {
        FloatMatrix2D A = x.like2D((int) x.size(), (int) y.size());
        multOuter(x, y, A);
        return A;
    }

    /**
     * Linear algebraic matrix power;
     * <tt>B = A<sup>k</sup> <==> B = A*A*...*A</tt>.
     * 
     * @param A
     *            the source matrix; must be square.
     * @param k
     *            the exponent, can be any number.
     * @return a new result matrix.
     * 
     * @throws IllegalArgumentException
     *             if <tt>!Testing.isSquare(A)</tt>.
     */
    public FloatMatrix2D xpowSlow(FloatMatrix2D A, int k) {
        // cern.colt.Timer timer = new cern.colt.Timer().start();
        FloatMatrix2D result = A.copy();
        for (int i = 0; i < k - 1; i++) {
            result = mult(result, A);
        }
        // timer.stop().display();
        return result;
    }
}
