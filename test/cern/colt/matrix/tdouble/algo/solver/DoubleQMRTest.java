package cern.colt.matrix.tdouble.algo.solver;


/**
 * Test of DoubleQMR
 */
public class DoubleQMRTest extends DoubleIterativeSolverTest {

    public DoubleQMRTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleQMR(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
