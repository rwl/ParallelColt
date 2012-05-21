package cern.colt.matrix.tdouble.algo.decomposition;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcss;

/**
 * For a square matrix <tt>A</tt>, the LU decomposition is an unit lower
 * triangular matrix <tt>L</tt>, an upper triangular matrix <tt>U</tt>, and a
 * permutation vector <tt>piv</tt> so that <tt>A(piv,:) = L*U</tt>
 * <P>
 * The LU decomposition with pivoting always exists, even if the matrix is
 * singular. The primary use of the LU decomposition is in the solution of
 * square systems of simultaneous linear equations. This will fail if
 * <tt>isNonsingular()</tt> returns false.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public interface SparseDoubleLUDecomposition {

	/**
	 * Returns the determinant, <tt>det(A)</tt>.
	 * 
	 */
	public abstract double det();

	/**
	 * Returns the lower triangular factor, <tt>L</tt>.
	 * 
	 * @return <tt>L</tt>
	 */
	public abstract DoubleMatrix2D getL();

	/**
	 * Returns a copy of the pivot permutation vector.
	 * 
	 * @return piv
	 */
	public abstract int[] getPivot();

	/**
	 * Returns the upper triangular factor, <tt>U</tt>.
	 * 
	 * @return <tt>U</tt>
	 */
	public abstract DoubleMatrix2D getU();

	/**
	 * Returns a copy of the symbolic LU analysis object
	 * 
	 * @return symbolic LU analysis
	 */
	public abstract Dcss getSymbolicAnalysis();

	/**
	 * Returns whether the matrix is nonsingular (has an inverse).
	 * 
	 * @return true if <tt>U</tt>, and hence <tt>A</tt>, is nonsingular; false
	 *         otherwise.
	 */
	public abstract boolean isNonsingular();

	/**
	 * Solves <tt>A*x = b</tt>(in-place). Upon return <tt>b</tt> is overridden
	 * with the result <tt>x</tt>.
	 * 
	 * @param b
	 *            A vector with of size A.rows();
	 * @exception IllegalArgumentException
	 *                if <tt>b.size() != A.rows()</tt> or if A is singular.
	 */
	public abstract void solve(DoubleMatrix1D b);

}