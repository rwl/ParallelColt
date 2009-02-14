package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatDiagonal;


/**
 * Test of FloatIR with diagonal preconditioner
 */
public class FloatIRDiagonalTest extends FloatGMRESTest {

    public FloatIRDiagonalTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatDiagonal(A.rows());
    }

}
