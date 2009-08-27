package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleICC;

/**
 * Benchmark of DoubleCG with ICC
 */
public class DoubleCGICCBenchmark extends DoubleCGBenchmark {

    public DoubleCGICCBenchmark(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleICC(A.rows());
    }

}
