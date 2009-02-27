package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;

/**
 * Test of DoubleBiCGstab with AMG
 */
public class DoubleBiCGstabAMGTest extends DoubleBiCGstabTest {

    public DoubleBiCGstabAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
