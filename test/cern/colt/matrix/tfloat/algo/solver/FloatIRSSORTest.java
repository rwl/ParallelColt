package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatSSOR;

/**
 * Test of FloatIR with SSOR
 */
public class FloatIRSSORTest extends FloatIRTest {

    public FloatIRSSORTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        float omega = (float) Math.random() + 1;
        M = new FloatSSOR(A.rows(), true, omega, omega);
    }

}
