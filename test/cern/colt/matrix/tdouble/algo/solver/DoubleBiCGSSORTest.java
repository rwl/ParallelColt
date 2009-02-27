package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleSSOR;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2D;

/**
 * Test of DoubleBiCG with SSOR
 */
public class DoubleBiCGSSORTest extends DoubleBiCGTest {

    public DoubleBiCGSSORTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        double omega = Math.random() + 1;
        M = new DoubleSSOR((RCDoubleMatrix2D) new RCDoubleMatrix2D(A.rows(), A.columns()).assign(A), true, omega, omega);
    }

}
