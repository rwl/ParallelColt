package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILU;

/**
 * Test of FloatChebyshev with ILU
 */
public class FloatChebyshevILUTest extends FloatChebyshevTest {

    public FloatChebyshevILUTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILU(A.rows());
    }

}
