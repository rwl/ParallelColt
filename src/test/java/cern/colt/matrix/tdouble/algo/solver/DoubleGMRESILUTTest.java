package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILUT;

/**
 * Test of DoubleGMRES with ILUT
 */
public class DoubleGMRESILUTTest extends DoubleGMRESTest {

    public DoubleGMRESILUTTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILUT(A.rows());
    }

}
