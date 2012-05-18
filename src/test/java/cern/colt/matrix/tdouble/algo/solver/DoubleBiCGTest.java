package cern.colt.matrix.tdouble.algo.solver;

/**
 * Test of DoubleBiCG
 */
public class DoubleBiCGTest extends DoubleIterativeSolverTest {

    public DoubleBiCGTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        solver = new DoubleBiCG(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
