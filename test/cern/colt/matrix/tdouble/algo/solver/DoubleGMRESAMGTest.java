package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;


/**
 * Test of DoubleGMRES with AMG
 */
public class DoubleGMRESAMGTest extends DoubleGMRESTest {

    public DoubleGMRESAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
