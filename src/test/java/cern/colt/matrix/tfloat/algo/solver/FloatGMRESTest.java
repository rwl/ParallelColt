package cern.colt.matrix.tfloat.algo.solver;

/**
 * Test of FloatGMRES
 */
public class FloatGMRESTest extends FloatIterativeSolverTest {

    public FloatGMRESTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        solver = new FloatGMRES(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
