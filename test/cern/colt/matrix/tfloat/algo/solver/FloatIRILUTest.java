package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILU;
import cern.colt.matrix.tfloat.impl.RCFloatMatrix2D;

/**
 * Test of FloatIR with ILU
 */
public class FloatIRILUTest extends FloatIRTest {

    public FloatIRILUTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILU((RCFloatMatrix2D) new RCFloatMatrix2D(A.rows(), A.columns()).assign(A));
    }

}
