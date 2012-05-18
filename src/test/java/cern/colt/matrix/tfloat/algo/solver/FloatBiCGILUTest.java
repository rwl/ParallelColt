package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILU;

/**
 * Test of FloatBiCG with ILU
 */
public class FloatBiCGILUTest extends FloatBiCGTest {

    public FloatBiCGILUTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILU(A.rows());
    }

}
