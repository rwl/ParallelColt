package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;


/**
 * Test of DoubleQMR with AMG
 */
public class DoubleQMRAMGTest extends DoubleQMRTest {

    public DoubleQMRAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
