package cern.colt.matrix.tdouble.algo.decomposition;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.SparseCCDoubleMatrix2D;

public class SparseDoubleDecompositionTest {

	private static int N = 5 ;
	private static int [ ] Ap = {0, 2, 5, 9, 10, 12} ;
	private static int [ ] Ai = {0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4} ;
	private static double [ ] Ax = {2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.} ;
	private static double [ ] b = {8., 45., -3., 3., 19.} ;

	/**
	 * solution is x = (1,2,3,4,5)
	 */
	public static void main(String[] args) {

	        DoubleMatrix2D A = new SparseCCDoubleMatrix2D(N, N, Ai, Ap, Ax);
	        DoubleMatrix1D Bclu = new DenseDoubleMatrix1D(b);
	        DoubleMatrix1D Bklu = new DenseDoubleMatrix1D(b);
	        DoubleMatrix1D Bqr = new DenseDoubleMatrix1D(b);

	        SparseDoubleLUDecomposition clu = new CSparseDoubleLUDecomposition(A, 0, true);
	        clu.solve(Bclu);
	        System.out.println(Bclu.toString());

	        SparseDoubleLUDecomposition klu = new SparseDoubleKLUDecomposition(A, 0, true);
	        klu.solve(Bklu);
	        System.out.println(Bklu.toString());
	        
	        SparseDoubleQRDecomposition qr = new SparseDoubleQRDecomposition(A, 0);
	        qr.solve(Bqr);
	        System.out.println(Bqr);
	}

}
