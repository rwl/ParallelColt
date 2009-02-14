package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleDiagonal;


/**
 * Test of DoubleCG with diagonal preconditioner
 */
public class DoubleCGDiagonalTest extends DoubleCGTest {

    public DoubleCGDiagonalTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleDiagonal(A.rows());
    }

}
