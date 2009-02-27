package cern.colt.matrix.tdouble.algo.solver;

/**
 * Test of DoubleGMRES
 */
public class DoubleGMRESTest extends DoubleIterativeSolverTest {

    public DoubleGMRESTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleGMRES(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
