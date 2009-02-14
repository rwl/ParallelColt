package cern.colt.matrix.tfloat.algo.solver;


/**
 * Test of FloatCG
 */
public class FloatCGTest extends FloatIterativeSolverTest {

    public FloatCGTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new FloatCG(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
