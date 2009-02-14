package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatILUT;
import cern.colt.matrix.tfloat.impl.RCMFloatMatrix2D;

/**
 * Test of FloatBiCG with ILUT
 */
public class FloatBiCGILUTTest extends FloatBiCGTest {

    public FloatBiCGILUTTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatILUT((RCMFloatMatrix2D)(new RCMFloatMatrix2D(A.rows(), A.columns()).assign(A)));
    }

}
