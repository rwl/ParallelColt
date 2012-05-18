package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILUT;

/**
 * Test of FloatIR with ILUT
 */
public class FloatIRILUTTest extends FloatIRTest {

    public FloatIRILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILUT(A.rows());
    }

}
