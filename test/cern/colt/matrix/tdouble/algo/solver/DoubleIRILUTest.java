package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILU;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2D;


/**
 * Test of DoubleIR with ILU
 */
public class DoubleIRILUTest extends DoubleIRTest {

    public DoubleIRILUTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILU((RCDoubleMatrix2D)new RCDoubleMatrix2D(A.rows(), A.columns()).assign(A));
    }

}
