package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleDiagonal;

/**
 * Test of DoubleGMRES with diagonal preconditioner
 */
public class DoubleGMRESDiagonalTest extends DoubleGMRESTest {

    public DoubleGMRESDiagonalTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleDiagonal(A.rows());
    }

}
