package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILUT;

/**
 * Test of FloatQMR with ILUT
 */
public class FloatQMRILUTTest extends FloatQMRTest {

    public FloatQMRILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILUT(A.rows());
    }

}
