package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;


/**
 * Test of FloatGMRES with AMG
 */
public class FloatGMRESAMGTest extends FloatGMRESTest {

    public FloatGMRESAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
