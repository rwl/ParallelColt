package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatICC;

/**
 * Test of FloatBiCG with ICC
 */
public class FloatBiCGICCTest extends FloatBiCGTest {

    public FloatBiCGICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatICC(A.rows());
    }

}
