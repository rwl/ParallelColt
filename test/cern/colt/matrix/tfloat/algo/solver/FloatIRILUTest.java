package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILU;

/**
 * Test of FloatIR with ILU
 */
public class FloatIRILUTest extends FloatIRTest {

    public FloatIRILUTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILU(A.rows());
    }

}
