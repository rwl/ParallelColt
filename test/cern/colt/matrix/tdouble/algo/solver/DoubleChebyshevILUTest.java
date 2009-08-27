package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILU;

/**
 * Test of DoubleChebyshev with ILU
 */
public class DoubleChebyshevILUTest extends DoubleChebyshevTest {

    public DoubleChebyshevILUTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILU(A.rows());
    }

}
