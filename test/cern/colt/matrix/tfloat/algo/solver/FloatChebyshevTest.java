package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.FloatAlgebra;
import cern.colt.matrix.tfloat.algo.decomposition.FloatEigenvalueDecomposition;

/**
 * Test of FloatChebyshev
 */
public class FloatChebyshevTest extends FloatIterativeSolverTest {

    public FloatChebyshevTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        // Get the extremal eigenvalues
        FloatEigenvalueDecomposition evd = FloatAlgebra.DEFAULT.eig(A);
        float[] eigs = (float[]) evd.getRealEigenvalues().elements();

        float eigmin = 1, eigmax = 1;
        if (eigs.length > 0) {
            eigmin = eigs[0];
            eigmax = eigs[eigs.length - 1];
        }

        solver = new FloatChebyshev(x, eigmin, eigmax);
        M = solver.getPreconditioner(); //identity preconditioner
    }

}
