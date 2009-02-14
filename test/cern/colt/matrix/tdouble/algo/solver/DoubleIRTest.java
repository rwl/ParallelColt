package cern.colt.matrix.tdouble.algo.solver;


/**
 * Test of DoubleIR
 */
public class DoubleIRTest extends DoubleIterativeSolverTest {

    public DoubleIRTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleIR(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
