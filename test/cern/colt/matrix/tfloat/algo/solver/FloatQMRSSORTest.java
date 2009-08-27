package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatSSOR;

/**
 * Test of FloatQMR with SSOR
 */
public class FloatQMRSSORTest extends FloatQMRTest {

    public FloatQMRSSORTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        float omega = (float) Math.random() + 1;
        M = new FloatSSOR(A.rows(), true, omega, omega);
    }

}
