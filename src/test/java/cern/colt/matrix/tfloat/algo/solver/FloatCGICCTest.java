package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatICC;

/**
 * Test of FloatCG with ICC
 */
public class FloatCGICCTest extends FloatCGTest {

    public FloatCGICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatICC(A.rows());
    }

}
