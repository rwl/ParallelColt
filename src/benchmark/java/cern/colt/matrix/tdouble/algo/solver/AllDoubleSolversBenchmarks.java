package cern.colt.matrix.tdouble.algo.solver;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Benchmarks of double precision iterative solvers
 */
public class AllDoubleSolversBenchmarks {

    public static Test suite() {
        TestSuite suite = new TestSuite("Benchmarks of iterative solvers");
        suite.addTestSuite(DoubleHyBRBenchmark.class);
        suite.addTestSuite(DoubleCGBenchmark.class);
        suite.addTestSuite(DoubleCGDiagonalBenchmark.class);
        suite.addTestSuite(DoubleCGICCBenchmark.class);
        suite.addTestSuite(DoubleCGILUBenchmark.class);
        suite.addTestSuite(DoubleCGILUTBenchmark.class);
        suite.addTestSuite(DoubleCGSSORBenchmark.class);
        suite.addTestSuite(DoubleCGAMGBenchmark.class);
        return suite;
    }
}
