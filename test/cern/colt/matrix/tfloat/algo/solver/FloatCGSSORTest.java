package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatSSOR;
import cern.colt.matrix.tfloat.impl.RCFloatMatrix2D;


/**
 * Test of FloatCG with SSOR
 */
public class FloatCGSSORTest extends FloatCGTest {

    public FloatCGSSORTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        float omega = (float)Math.random() + 1;
        M = new FloatSSOR((RCFloatMatrix2D)A, true, omega, omega);
    }

}
