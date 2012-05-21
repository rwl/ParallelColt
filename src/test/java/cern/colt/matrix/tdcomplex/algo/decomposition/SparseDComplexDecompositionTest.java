package cern.colt.matrix.tdcomplex.algo.decomposition;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.SparseCCDComplexMatrix2D;

public class SparseDComplexDecompositionTest {

	private static int N = 5 ;
	private static int [ ] Ap = {0, 2, 5, 9, 10, 12} ;
	private static int [ ] Ai = {0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4} ;
	private static double [ ] Ax = {2, 0, 3, 0, 3, 0, -1, 0, 4, 0, 4, 0, -3, 0, 1, 0, 2, 0, 2, 0, 6, 0, 1, 0} ;
	private static double [ ] b = {8, 0, 45, 0, -3, 0, 3, 0, 19, 0} ;

	/**
	 * solution is x = (1,2,3,4,5)
	 */
	public static void main(String[] args) {

	        DComplexMatrix2D A = new SparseCCDComplexMatrix2D(N, N, Ai, Ap, Ax);
	        DComplexMatrix1D Blu = new DenseDComplexMatrix1D(b);
	        DComplexMatrix1D Bqr = new DenseDComplexMatrix1D(b);

	        SparseDComplexLUDecomposition lu = new SparseDComplexLUDecomposition(A, 0, true);
	        lu.solve(Blu);
	        System.out.println(Blu.toString());
	        
	        SparseDComplexQRDecomposition qr = new SparseDComplexQRDecomposition(A, 0);
	        qr.solve(Bqr);
	        System.out.println(Bqr);
	}

}
