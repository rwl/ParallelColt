package cern.colt.matrix.tdouble.algo.solver;

import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoubleILU;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2D;


/**
 * Benchmark of DoubleCG with ILU
 */
public class DoubleCGILUBenchmark extends DoubleCGBenchmark {

    public DoubleCGILUBenchmark(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        super.createSolver();
        M = new DoubleILU((RCDoubleMatrix2D)new RCDoubleMatrix2D(A.rows(), A.columns()).assign(A));
    }

}
