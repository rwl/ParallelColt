package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatSSOR;

/**
 * Test of FloatBiCG with SSOR
 */
public class FloatBiCGSSORTest extends FloatBiCGTest {

    public FloatBiCGSSORTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        float omega = (float) Math.random() + 1;
        M = new FloatSSOR(A.rows(), true, omega, omega);
    }

}
