package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;


/**
 * Test of FloatCG with AMG
 */
public class FloatCGAMGTest extends FloatCGTest {

    public FloatCGAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
