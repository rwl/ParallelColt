package cern.colt.matrix.tfloat.algo.solver;


/**
 * Test of FloatHyBR
 */
public class FloatHyBRTest extends FloatIterativeSolverTest {

    public FloatHyBRTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new FloatHyBR();
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
