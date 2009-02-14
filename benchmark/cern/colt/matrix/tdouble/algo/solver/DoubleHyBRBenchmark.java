package cern.colt.matrix.tdouble.algo.solver;


/**
 * Benchmark of DoubleHyBR
 */
public class DoubleHyBRBenchmark extends DoubleIterativeSolverBenchmark {

    public DoubleHyBRBenchmark(String arg0) {
        super(arg0);
    }

    @Override
    protected void createSolver() throws Exception {
        solver = new DoubleHyBR();
        M = solver.getPreconditioner();
    }

}
