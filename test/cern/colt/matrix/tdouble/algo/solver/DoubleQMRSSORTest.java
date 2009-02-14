package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleSSOR;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2D;


/**
 * Test of DoubleQMR with SSOR
 */
public class DoubleQMRSSORTest extends DoubleQMRTest {

    public DoubleQMRSSORTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        double omega = Math.random() + 1;
        M = new DoubleSSOR((RCDoubleMatrix2D)A.copy(), true, omega, omega);
    }

}
