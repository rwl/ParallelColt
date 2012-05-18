package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleICC;

/**
 * Test of DoubleChebyshev with ICC
 */
public class DoubleChebyshevICCTest extends DoubleChebyshevTest {

    public DoubleChebyshevICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleICC(A.rows());
    }

}
