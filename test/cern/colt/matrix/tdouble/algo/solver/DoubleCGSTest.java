package cern.colt.matrix.tdouble.algo.solver;

/**
 * Test of DoubleCGS
 */
public class DoubleCGSTest extends DoubleIterativeSolverTest {

    public DoubleCGSTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleCGS(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
