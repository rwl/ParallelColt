package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleICC;

/**
 * Test of DoubleBiCGstab with ICC
 */
public class DoubleBiCGstabICCTest extends DoubleBiCGstabTest {

    public DoubleBiCGstabICCTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleICC(A.rows());
    }

}
