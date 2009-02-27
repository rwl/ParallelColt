package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;

/**
 * Test of DoubleCGS with AMG
 */
public class DoubleCGSAMGTest extends DoubleCGSTest {

    public DoubleCGSAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
