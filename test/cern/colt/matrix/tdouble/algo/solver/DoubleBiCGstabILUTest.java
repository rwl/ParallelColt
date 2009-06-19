package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILU;

/**
 * Test of DoubleBiCGstab with ILU
 */
public class DoubleBiCGstabILUTest extends DoubleBiCGstabTest {

    public DoubleBiCGstabILUTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILU(A.rows());
    }

}
