package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatAMG;

/**
 * Test of FloatChebyshev with AMG
 */
public class FloatChebyshevAMGTest extends FloatChebyshevTest {

    public FloatChebyshevAMGTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatAMG();
    }

}
