package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleDiagonal;

/**
 * Test DoubleQMR with diagonal preconditioner
 */
public class DoubleQMRDiagonalTest extends DoubleQMRTest {

    public DoubleQMRDiagonalTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleDiagonal(A.rows());
    }

}
