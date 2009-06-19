package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILUT;

/**
 * Test of DoubleBiCG with ILUT
 */
public class DoubleBiCGILUTTest extends DoubleBiCGTest {

    public DoubleBiCGILUTTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILUT(A.rows());
    }

}
