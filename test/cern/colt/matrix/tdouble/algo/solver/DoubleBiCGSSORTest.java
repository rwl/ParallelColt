package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleSSOR;

/**
 * Test of DoubleBiCG with SSOR
 */
public class DoubleBiCGSSORTest extends DoubleBiCGTest {

    public DoubleBiCGSSORTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        double omega = Math.random() + 1;
        M = new DoubleSSOR(A.rows(), true, omega, omega);
    }

}
