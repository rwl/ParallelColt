package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleAMG;

/**
 * Benchmark of DoubleCG with AMG
 */
public class DoubleCGAMGBenchmark extends DoubleCGBenchmark {

    public DoubleCGAMGBenchmark(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleAMG();
    }

}
