package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILUT;

/**
 * Test of DoubleBiCGstab with ILUT
 */
public class DoubleBiCGstabILUTTest extends DoubleBiCGstabTest {

    public DoubleBiCGstabILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILUT(A.rows());
    }

}
