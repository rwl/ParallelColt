package cern.colt.matrix.tdouble.algo.solver;

/**
 * Test of DoubleCG
 */
public class DoubleCGTest extends DoubleIterativeSolverTest {

    public DoubleCGTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        solver = new DoubleCG(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
