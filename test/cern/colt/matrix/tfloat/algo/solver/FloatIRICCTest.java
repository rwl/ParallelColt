package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatICC;

/**
 * Test of FloatIR with ICC
 */
public class FloatIRICCTest extends FloatIRTest {

    public FloatIRICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatICC(A.rows());
    }

}
