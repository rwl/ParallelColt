package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;

/**
 * Test of DoubleBiCG wit AMG
 * 
 */
public class DoubleBiCGAMGTest extends DoubleBiCGTest {

    public DoubleBiCGAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
