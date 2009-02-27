package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;

/**
 * Test of DoubleCG with AMG
 */
public class DoubleCGAMGTest extends DoubleCGTest {

    public DoubleCGAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
