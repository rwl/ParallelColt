package cern.colt;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tdcomplex.AllDComplexMatrixTests;
import cern.colt.matrix.tdouble.AllDoubleMatrixTests;
import cern.colt.matrix.tfcomplex.AllFComplexMatrixTests;
import cern.colt.matrix.tfloat.AllFloatMatrixTests;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class AllTests {

    public static int NTHREADS = 2;

    public static Test suite() {
        ConcurrencyUtils.setNumberOfThreads(NTHREADS);
        System.out.println("Running Parallel Colt tests using " + ConcurrencyUtils.getNumberOfThreads() + " threads.");
        TestSuite suite = new TestSuite("Parallel Colt tests");
        suite.addTest(AllDoubleMatrixTests.suite());
        suite.addTest(AllDComplexMatrixTests.suite());
        suite.addTest(AllFloatMatrixTests.suite());
        suite.addTest(AllFComplexMatrixTests.suite());
        return suite;
    }

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main(AllTests.class.getName().toString());
    }
}
