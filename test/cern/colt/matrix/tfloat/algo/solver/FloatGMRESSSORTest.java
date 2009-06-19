package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatSSOR;

/**
 * Test of FloatGMRES with SSOR
 */
public class FloatGMRESSSORTest extends FloatGMRESTest {

    public FloatGMRESSSORTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        float omega = (float) Math.random() + 1;
        M = new FloatSSOR(A.rows(), true, omega, omega);
    }

}
