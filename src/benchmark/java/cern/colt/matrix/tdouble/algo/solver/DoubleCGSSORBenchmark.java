package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleSSOR;

/**
 * Benchmark of DoubleCG with SOR
 */
public class DoubleCGSSORBenchmark extends DoubleCGBenchmark {

    public DoubleCGSSORBenchmark(String arg0) {
        super(arg0);
    }

    protected void createSolver() throws Exception {
        super.createSolver();
        double omega = 1.1;
        M = new DoubleSSOR(A.rows(), true, omega, omega);
    }

}
