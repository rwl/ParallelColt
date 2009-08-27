package cern.colt.matrix.tfloat.algo.solver;

/**
 * Test of FloatIR
 */
public class FloatIRTest extends FloatIterativeSolverTest {

    public FloatIRTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        solver = new FloatIR(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
