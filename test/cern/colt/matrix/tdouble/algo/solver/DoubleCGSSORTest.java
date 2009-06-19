package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleSSOR;

/**
 * Test of DoubleCG with SSOR
 */
public class DoubleCGSSORTest extends DoubleCGTest {

    public DoubleCGSSORTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        double omega = Math.random() + 1;
        M = new DoubleSSOR(A.rows(), true, omega, omega);
    }

}
