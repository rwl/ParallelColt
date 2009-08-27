package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatICC;

/**
 * Test of FloatChebyshev with ICC
 */
public class FloatChebyshevICCTest extends FloatChebyshevTest {

    public FloatChebyshevICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatICC(A.rows());
    }

}
