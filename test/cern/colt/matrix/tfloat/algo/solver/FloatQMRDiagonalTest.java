package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatDiagonal;

/**
 * Test FloatQMR with diagonal preconditioner
 */
public class FloatQMRDiagonalTest extends FloatQMRTest {

    public FloatQMRDiagonalTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatDiagonal(A.rows());
    }

}
