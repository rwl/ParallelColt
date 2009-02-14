package cern.colt.matrix.tdouble.algo.solver;


/**
 * Test of DoubleBiCGstab
 */
public class DoubleBiCGstabTest extends DoubleIterativeSolverTest {

    public DoubleBiCGstabTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleBiCGstab(x);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
