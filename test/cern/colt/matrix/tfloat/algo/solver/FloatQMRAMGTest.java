package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;


/**
 * Test of FloatQMR with AMG
 */
public class FloatQMRAMGTest extends FloatQMRTest {

    public FloatQMRAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
