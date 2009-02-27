package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;

/**
 * Test of FloatIR with AMG
 */
public class FloatIRAMGTest extends FloatIRTest {

    public FloatIRAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
