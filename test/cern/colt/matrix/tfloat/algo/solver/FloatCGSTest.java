package cern.colt.matrix.tfloat.algo.solver;

/**
 * Test of FloatCGS
 */
public class FloatCGSTest extends FloatIterativeSolverTest {

    public FloatCGSTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new FloatCGS(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
