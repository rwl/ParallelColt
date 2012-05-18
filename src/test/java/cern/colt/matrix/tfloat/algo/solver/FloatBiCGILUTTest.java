package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILUT;

/**
 * Test of FloatBiCG with ILUT
 */
public class FloatBiCGILUTTest extends FloatBiCGTest {

    public FloatBiCGILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILUT(A.rows());
    }

}
