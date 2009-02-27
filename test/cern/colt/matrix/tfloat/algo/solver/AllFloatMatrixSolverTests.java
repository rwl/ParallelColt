package cern.colt.matrix.tfloat.algo.solver;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Test of all float precision iterative solvers
 */
public class AllFloatMatrixSolverTests {

    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tfloat.algo.solvers tests");
        suite.addTestSuite(FloatHyBRTest.class);

        suite.addTestSuite(FloatCGTest.class);
        suite.addTestSuite(FloatCGDiagonalTest.class);
        suite.addTestSuite(FloatCGSSORTest.class);
        suite.addTestSuite(FloatCGILUTest.class);
        suite.addTestSuite(FloatCGICCTest.class);
        //                suite.addTestSuite(CGAMGTest.class);
        suite.addTestSuite(FloatCGILUTTest.class);

        suite.addTestSuite(FloatCGSTest.class);
        suite.addTestSuite(FloatCGSDiagonalTest.class);
        suite.addTestSuite(FloatCGSSSORTest.class);
        suite.addTestSuite(FloatCGSILUTest.class);
        suite.addTestSuite(FloatCGSICCTest.class);
        //        suite.addTestSuite(CGSAMGTest.class);
        suite.addTestSuite(FloatCGSILUTTest.class);

        suite.addTestSuite(FloatQMRTest.class);
        suite.addTestSuite(FloatQMRDiagonalTest.class);
        suite.addTestSuite(FloatQMRSSORTest.class);
        suite.addTestSuite(FloatQMRILUTest.class);
        suite.addTestSuite(FloatQMRICCTest.class);
        //        suite.addTestSuite(QMRAMGTest.class);
        suite.addTestSuite(FloatQMRILUTTest.class);

        suite.addTestSuite(FloatBiCGTest.class);
        suite.addTestSuite(FloatBiCGDiagonalTest.class);
        suite.addTestSuite(FloatBiCGSSORTest.class);
        suite.addTestSuite(FloatBiCGILUTest.class);
        suite.addTestSuite(FloatBiCGICCTest.class);
        suite.addTestSuite(FloatBiCGAMGTest.class);
        suite.addTestSuite(FloatBiCGILUTTest.class);

        suite.addTestSuite(FloatBiCGstabTest.class);
        suite.addTestSuite(FloatBiCGstabDiagonalTest.class);
        suite.addTestSuite(FloatBiCGstabSSORTest.class);
        suite.addTestSuite(FloatBiCGstabILUTest.class);
        suite.addTestSuite(FloatBiCGstabICCTest.class);
        suite.addTestSuite(FloatBiCGstabAMGTest.class);
        suite.addTestSuite(FloatBiCGstabILUTTest.class);

        suite.addTestSuite(FloatGMRESTest.class);
        suite.addTestSuite(FloatGMRESDiagonalTest.class);
        suite.addTestSuite(FloatGMRESSSORTest.class);
        suite.addTestSuite(FloatGMRESILUTest.class);
        suite.addTestSuite(FloatGMRESICCTest.class);
        suite.addTestSuite(FloatGMRESAMGTest.class);
        suite.addTestSuite(FloatGMRESILUTTest.class);

        suite.addTestSuite(FloatChebyshevTest.class);
        suite.addTestSuite(FloatChebyshevDiagonalTest.class);
        suite.addTestSuite(FloatChebyshevSSORTest.class);
        suite.addTestSuite(FloatChebyshevILUTest.class);
        suite.addTestSuite(FloatChebyshevICCTest.class);
        suite.addTestSuite(FloatChebyshevAMGTest.class);
        suite.addTestSuite(FloatChebyshevILUTTest.class);

        //                suite.addTestSuite(IRTest.class);
        suite.addTestSuite(FloatIRDiagonalTest.class);
        suite.addTestSuite(FloatIRSSORTest.class);
        suite.addTestSuite(FloatIRILUTest.class);
        suite.addTestSuite(FloatIRICCTest.class);
        suite.addTestSuite(FloatIRAMGTest.class);
        suite.addTestSuite(FloatIRILUTTest.class);

        return suite;
    }
}
