package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleEigenvalueDecomposition;

/**
 * Test of DoubleChebyshev
 */
public class DoubleChebyshevTest extends DoubleIterativeSolverTest {

    public DoubleChebyshevTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        // Get the extremal eigenvalues
        DenseDoubleEigenvalueDecomposition evd = DenseDoubleAlgebra.DEFAULT.eig(A);
        double[] eigs = (double[]) evd.getRealEigenvalues().elements();

        double eigmin = 1, eigmax = 1;
        if (eigs.length > 0) {
            eigmin = eigs[0];
            eigmax = eigs[eigs.length - 1];
        }

        solver = new DoubleChebyshev(x, eigmin, eigmax);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
