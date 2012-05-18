package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatDiagonal;

/**
 * Test of FloatCG with diagonal preconditioner
 */
public class FloatCGDiagonalTest extends FloatCGTest {

    public FloatCGDiagonalTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatDiagonal(A.rows());
    }

}
