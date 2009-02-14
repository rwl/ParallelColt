package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;


/**
 * Test of FloatCGS with AMG
 */
public class FloatCGSAMGTest extends FloatCGSTest {

    public FloatCGSAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
