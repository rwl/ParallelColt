package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;


/**
 * Test of DoubleChebyshev with AMG
 */
public class DoubleChebyshevAMGTest extends DoubleChebyshevTest {

    public DoubleChebyshevAMGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
