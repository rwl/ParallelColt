package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILUT;

/**
 * Test of DoubleChebyshev with ILUT
 */
public class DoubleChebyshevILUTTest extends DoubleChebyshevTest {

    public DoubleChebyshevILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILUT(A.rows());
    }

}
