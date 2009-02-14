package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatDiagonal;

/**
 * Test of FloatChebyshev with diagonal preconditioner
 */
public class FloatChebyshevDiagonalTest extends FloatChebyshevTest {

    public FloatChebyshevDiagonalTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatDiagonal(A.rows());
    }

}
