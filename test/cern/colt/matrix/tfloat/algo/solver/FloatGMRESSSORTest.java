package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatSSOR;
import cern.colt.matrix.tfloat.impl.RCFloatMatrix2D;


/**
 * Test of FloatGMRES with SSOR
 */
public class FloatGMRESSSORTest extends FloatGMRESTest {

    public FloatGMRESSSORTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        float omega = (float)Math.random() + 1;
        M = new FloatSSOR((RCFloatMatrix2D)new RCFloatMatrix2D(A.rows(), A.columns()).assign(A), true, omega, omega);
    }

}
