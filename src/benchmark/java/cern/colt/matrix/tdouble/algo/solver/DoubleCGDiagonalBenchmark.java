package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleDiagonal;

/**
 * Benchmark of DoubleCG with diagonal preconditioning
 */
public class DoubleCGDiagonalBenchmark extends DoubleCGBenchmark {

    public DoubleCGDiagonalBenchmark(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleDiagonal(A.rows());
    }

}
