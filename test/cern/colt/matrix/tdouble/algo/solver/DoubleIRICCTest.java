package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleICC;

/**
 * Test of DoubleIR with ICC
 */
public class DoubleIRICCTest extends DoubleIRTest {

    public DoubleIRICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleICC(A.rows());
    }

}
