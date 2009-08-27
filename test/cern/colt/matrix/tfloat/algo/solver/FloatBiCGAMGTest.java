package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;

/**
 * Test of FloatBiCG wit AMG
 * 
 */
public class FloatBiCGAMGTest extends FloatBiCGTest {

    public FloatBiCGAMGTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
