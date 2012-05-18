/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.algo;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.decomposition.SparseFloatCholeskyDecomposition;
import cern.colt.matrix.tfloat.algo.decomposition.SparseFloatLUDecomposition;
import cern.colt.matrix.tfloat.algo.decomposition.SparseFloatQRDecomposition;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import cern.colt.matrix.tfloat.impl.SparseCCFloatMatrix2D;
import cern.colt.matrix.tfloat.impl.SparseRCFloatMatrix2D;
import edu.emory.mathcs.csparsej.tfloat.Scs_norm;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scs;

/**
 * Linear algebraic matrix operations operating on sparse matrices.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class SparseFloatAlgebra {

    /**
     * A default Algebra object; has {@link FloatProperty#DEFAULT} attached for
     * tolerance. Allows ommiting to construct an Algebra object time and again.
     * 
     * Note that this Algebra object is immutable. Any attempt to assign a new
     * Property object to it (via method <tt>setProperty</tt>), or to alter the
     * tolerance of its property object (via
     * <tt>property().setTolerance(...)</tt>) will throw an exception.
     */
    public static final SparseFloatAlgebra DEFAULT;

    /**
     * A default Algebra object; has {@link FloatProperty#ZERO} attached for
     * tolerance. Allows ommiting to construct an Algebra object time and again.
     * 
     * Note that this Algebra object is immutable. Any attempt to assign a new
     * Property object to it (via method <tt>setProperty</tt>), or to alter the
     * tolerance of its property object (via
     * <tt>property().setTolerance(...)</tt>) will throw an exception.
     */
    public static final SparseFloatAlgebra ZERO;

    static {
        // don't use new Algebra(Property.DEFAULT.tolerance()), because then
        // property object would be mutable.
        DEFAULT = new SparseFloatAlgebra();
        DEFAULT.property = FloatProperty.DEFAULT; // immutable property object

        ZERO = new SparseFloatAlgebra();
        ZERO.property = FloatProperty.ZERO; // immutable property object
    }

    private static float normInfinityRC(SparseRCFloatMatrix2D A) {
        int p, j, n, Ap[];
        float Ax[], norm = 0, s;
        n = A.rows();
        Ap = A.getRowPointers();
        Ax = A.getValues();
        for (j = 0; j < n; j++) {
            for (s = 0, p = Ap[j]; p < Ap[j + 1]; p++)
                s += Math.abs(Ax[p]);
            norm = Math.max(norm, s);
        }
        return (norm);
    }

    /**
     * The property object attached to this instance.
     */
    protected FloatProperty property;

    /**
     * Constructs a new instance with an equality tolerance given by
     * <tt>Property.DEFAULT.tolerance()</tt>.
     */
    public SparseFloatAlgebra() {
        this(FloatProperty.DEFAULT.tolerance());
    }

    /**
     * Constructs a new instance with the given equality tolerance.
     * 
     * @param tolerance
     *            the tolerance to be used for equality operations.
     */
    public SparseFloatAlgebra(float tolerance) {
        setProperty(new FloatProperty(tolerance));
    }

    /**
     * Constructs and returns the Cholesky-decomposition of the given matrix.
     * 
     * @param matrix
     *            sparse matrix
     * @param order
     *            ordering option (0 or 1); 0: natural ordering, 1: amd(A+A')
     * @return Cholesky-decomposition of the given matrix
     */
    public SparseFloatCholeskyDecomposition chol(FloatMatrix2D matrix, int order) {
        return new SparseFloatCholeskyDecomposition(matrix, order);
    }

    /**
     * Returns a copy of the receiver. The attached property object is also
     * copied. Hence, the property object of the copy is mutable.
     * 
     * @return a copy of the receiver.
     */

    public Object clone() {
        return new SparseFloatAlgebra(property.tolerance());
    }

    /**
     * Returns the determinant of matrix <tt>A</tt>.
     * 
     * @param A
     *            sparse matrix
     * @return the determinant of matrix <tt>A</tt>
     */
    public float det(FloatMatrix2D A) {
        return lu(A, 0).det();
    }

    /**
     * Constructs and returns the LU-decomposition of the given matrix.
     * 
     * @param matrix
     *            sparse matrix
     * @param order
     *            ordering option (0 to 3); 0: natural ordering, 1: amd(A+A'),
     *            2: amd(S'*S), 3: amd(A'*A)
     * @return the LU-decomposition of the given matrix
     */
    public SparseFloatLUDecomposition lu(FloatMatrix2D matrix, int order) {
        return new SparseFloatLUDecomposition(matrix, order, true);
    }

    /**
     * Returns the 1-norm of matrix <tt>A</tt>, which is the maximum absolute
     * column sum.
     */
    public float norm1(FloatMatrix2D A) {
        FloatProperty.DEFAULT.checkSparse(A);
        float norm;
        if (A instanceof SparseCCFloatMatrix2D) {
            norm = Scs_norm.cs_norm((Scs) A.elements());
        } else {
            norm = Scs_norm.cs_norm(((SparseRCFloatMatrix2D) A).getColumnCompressed().elements());
        }
        return norm;
    }

    /**
     * Returns the infinity norm of matrix <tt>A</tt>, which is the maximum
     * absolute row sum.
     */
    public float normInfinity(FloatMatrix2D A) {
        FloatProperty.DEFAULT.checkSparse(A);
        float norm;
        if (A instanceof SparseRCFloatMatrix2D) {
            norm = normInfinityRC((SparseRCFloatMatrix2D) A);
        } else {
            norm = normInfinityRC(((SparseCCFloatMatrix2D) A).getRowCompressed());
        }
        return norm;
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
     * 
     * @param matrix
     *            sparse matrix
     * @param order
     *            ordering option (0 to 3); 0: natural ordering, 1: amd(A+A'),
     *            2: amd(S'*S), 3: amd(A'*A)
     * @return the QR-decomposition of the given matrix
     */
    public SparseFloatQRDecomposition qr(FloatMatrix2D matrix, int order) {
        return new SparseFloatQRDecomposition(matrix, order);
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
     * Solves A*x = b.
     * 
     * @param A
     *            sparse matrix
     * @param b
     *            right hand side
     * @return x; a new independent matrix; solution if A is square, least
     *         squares solution if A.rows() > A.columns(), underdetermined
     *         system solution if A.rows() < A.columns().
     */
    public FloatMatrix1D solve(FloatMatrix2D A, FloatMatrix1D b) {
        FloatMatrix1D x = new DenseFloatMatrix1D(Math.max(A.rows(), A.columns()));
        x.viewPart(0, (int) b.size()).assign(b);
        if (A.rows() == A.columns()) {
            lu(A, 0).solve(x);
            return x;
        } else {
            qr(A, 0).solve(x);
            return x.viewPart(0, A.columns()).copy();
        }
    }
}
