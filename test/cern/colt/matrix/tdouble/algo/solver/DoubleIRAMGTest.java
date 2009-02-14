package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;


/**
 * Test of DoubleIR with AMG
 */
public class DoubleIRAMGTest extends DoubleIRTest {

    public DoubleIRAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
