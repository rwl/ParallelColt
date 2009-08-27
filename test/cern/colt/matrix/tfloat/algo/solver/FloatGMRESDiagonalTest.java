package cern.colt.matrix.tfloat.algo.solver;

import cern.colt.matrix.tfloat.algo.solver.preconditioner.FloatDiagonal;

/**
 * Test of FloatGMRES with diagonal preconditioner
 */
public class FloatGMRESDiagonalTest extends FloatGMRESTest {

    public FloatGMRESDiagonalTest(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new FloatDiagonal(A.rows());
    }

}
