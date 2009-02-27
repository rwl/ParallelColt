package cern.colt.matrix.tdouble.algo.solver;

/**
 * Test of DoubleHyBR
 */
public class DoubleHyBRTest extends DoubleIterativeSolverTest {

    public DoubleHyBRTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleHyBR();
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
