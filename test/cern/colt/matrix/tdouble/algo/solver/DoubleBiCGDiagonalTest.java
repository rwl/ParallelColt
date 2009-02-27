package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleDiagonal;

/**
 * Test of DoubleBiCG with diagonal preconditioner
 */
public class DoubleBiCGDiagonalTest extends DoubleBiCGTest {

    public DoubleBiCGDiagonalTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleDiagonal(A.rows());
    }

}
