package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILUT;

/**
 * Test of FloatGMRES with ILUT
 */
public class FloatGMRESILUTTest extends FloatGMRESTest {

    public FloatGMRESILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILUT(A.rows());
    }

}
