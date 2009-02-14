package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILU;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2D;


/**
 * Test of DoubleBiCG with ILU
 */
public class DoubleBiCGILUTest extends DoubleBiCGTest {

    public DoubleBiCGILUTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILU((RCDoubleMatrix2D)new RCDoubleMatrix2D(A.rows(), A.columns()).assign(A));
    }

}
