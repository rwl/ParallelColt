package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleICC;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2D;


/**
 * Benchmark of DoubleCG with ICC
 */
public class DoubleCGICCBenchmark extends DoubleCGBenchmark {

    public DoubleCGICCBenchmark(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleICC((RCDoubleMatrix2D)new RCDoubleMatrix2D(A.rows(), A.columns()).assign(A));
    }

}
