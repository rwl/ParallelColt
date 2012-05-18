package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleICC;

/**
 * Test of DoubleQMR with ICC
 */
public class DoubleQMRICCTest extends DoubleQMRTest {

    public DoubleQMRICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleICC(A.rows());
    }

}
