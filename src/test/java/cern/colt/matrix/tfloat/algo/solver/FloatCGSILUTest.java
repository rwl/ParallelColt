package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILU;

/**
 * Test of FloatCGS with ILU
 */
public class FloatCGSILUTest extends FloatCGSTest {

    public FloatCGSILUTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILU(A.rows());
    }

}
