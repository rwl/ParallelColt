package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatICC;
import cern.colt.matrix.tfloat.impl.RCFloatMatrix2D;

/**
 * Test of FloatBiCGstab with ICC
 */
public class FloatBiCGstabICCTest extends FloatBiCGstabTest {

    public FloatBiCGstabICCTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatICC((RCFloatMatrix2D) new RCFloatMatrix2D(A.rows(), A.columns()).assign(A));
    }

}
