package cern.colt.matrix.tfloat.algo.solver;

/**
 * Test of FloatBiCG
 */
public class FloatBiCGTest extends FloatIterativeSolverTest {

    public FloatBiCGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new FloatBiCG(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
