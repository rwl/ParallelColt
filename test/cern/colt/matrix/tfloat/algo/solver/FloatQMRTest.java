package cern.colt.matrix.tfloat.algo.solver;

/**
 * Test of FloatQMR
 */
public class FloatQMRTest extends FloatIterativeSolverTest {

    public FloatQMRTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new FloatQMR(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
