package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILUT;

/**
 * Test of FloatBiCGstab with ILUT
 */
public class FloatBiCGstabILUTTest extends FloatBiCGstabTest {

    public FloatBiCGstabILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILUT(A.rows());
    }

}
