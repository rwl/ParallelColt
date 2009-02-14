package cern.colt.matrix.tdouble.algo.solver;



/**
 * Benchmark of DoubleCG
 */
public class DoubleCGBenchmark extends DoubleIterativeSolverBenchmark {

    public DoubleCGBenchmark(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleCG(x);
        M = solver.getPreconditioner();
    }

}
