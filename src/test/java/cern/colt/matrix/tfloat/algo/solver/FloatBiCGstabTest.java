package cern.colt.matrix.tfloat.algo.solver;

/**
 * Test of FloatBiCGstab
 */
public class FloatBiCGstabTest extends FloatIterativeSolverTest {

    public FloatBiCGstabTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        solver = new FloatBiCGstab(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
