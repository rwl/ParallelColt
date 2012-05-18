package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatICC;

/**
 * Test of FloatBiCGstab with ICC
 */
public class FloatBiCGstabICCTest extends FloatBiCGstabTest {

    public FloatBiCGstabICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatICC(A.rows());
    }

}
