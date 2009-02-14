package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;


/**
 * Test of FloatBiCGstab with AMG
 */
public class FloatBiCGstabAMGTest extends FloatBiCGstabTest {

    public FloatBiCGstabAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
