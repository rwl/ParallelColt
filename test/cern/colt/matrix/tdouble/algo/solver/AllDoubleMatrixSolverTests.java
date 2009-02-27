package cern.colt.matrix.tdouble.algo.solver;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Test of all double precision iterative solvers
 */
public class AllDoubleMatrixSolverTests {

    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tdouble.algo.solvers tests");
        suite.addTestSuite(DoubleHyBRTest.class);

        suite.addTestSuite(DoubleCGTest.class);
        suite.addTestSuite(DoubleCGDiagonalTest.class);
        suite.addTestSuite(DoubleCGSSORTest.class);
        suite.addTestSuite(DoubleCGILUTest.class);
        suite.addTestSuite(DoubleCGICCTest.class);
        //                suite.addTestSuite(CGAMGTest.class);
        suite.addTestSuite(DoubleCGILUTTest.class);

        suite.addTestSuite(DoubleCGSTest.class);
        suite.addTestSuite(DoubleCGSDiagonalTest.class);
        suite.addTestSuite(DoubleCGSSSORTest.class);
        suite.addTestSuite(DoubleCGSILUTest.class);
        suite.addTestSuite(DoubleCGSICCTest.class);
        //        suite.addTestSuite(CGSAMGTest.class);
        suite.addTestSuite(DoubleCGSILUTTest.class);

        suite.addTestSuite(DoubleQMRTest.class);
        suite.addTestSuite(DoubleQMRDiagonalTest.class);
        suite.addTestSuite(DoubleQMRSSORTest.class);
        suite.addTestSuite(DoubleQMRILUTest.class);
        suite.addTestSuite(DoubleQMRICCTest.class);
        //        suite.addTestSuite(QMRAMGTest.class);
        suite.addTestSuite(DoubleQMRILUTTest.class);

        suite.addTestSuite(DoubleBiCGTest.class);
        suite.addTestSuite(DoubleBiCGDiagonalTest.class);
        suite.addTestSuite(DoubleBiCGSSORTest.class);
        suite.addTestSuite(DoubleBiCGILUTest.class);
        suite.addTestSuite(DoubleBiCGICCTest.class);
        suite.addTestSuite(DoubleBiCGAMGTest.class);
        suite.addTestSuite(DoubleBiCGILUTTest.class);

        suite.addTestSuite(DoubleBiCGstabTest.class);
        suite.addTestSuite(DoubleBiCGstabDiagonalTest.class);
        suite.addTestSuite(DoubleBiCGstabSSORTest.class);
        suite.addTestSuite(DoubleBiCGstabILUTest.class);
        suite.addTestSuite(DoubleBiCGstabICCTest.class);
        suite.addTestSuite(DoubleBiCGstabAMGTest.class);
        suite.addTestSuite(DoubleBiCGstabILUTTest.class);

        suite.addTestSuite(DoubleGMRESTest.class);
        suite.addTestSuite(DoubleGMRESDiagonalTest.class);
        suite.addTestSuite(DoubleGMRESSSORTest.class);
        suite.addTestSuite(DoubleGMRESILUTest.class);
        suite.addTestSuite(DoubleGMRESICCTest.class);
        suite.addTestSuite(DoubleGMRESAMGTest.class);
        suite.addTestSuite(DoubleGMRESILUTTest.class);

        suite.addTestSuite(DoubleChebyshevTest.class);
        suite.addTestSuite(DoubleChebyshevDiagonalTest.class);
        suite.addTestSuite(DoubleChebyshevSSORTest.class);
        suite.addTestSuite(DoubleChebyshevILUTest.class);
        suite.addTestSuite(DoubleChebyshevICCTest.class);
        suite.addTestSuite(DoubleChebyshevAMGTest.class);
        suite.addTestSuite(DoubleChebyshevILUTTest.class);

        //                suite.addTestSuite(IRTest.class);
        suite.addTestSuite(DoubleIRDiagonalTest.class);
        suite.addTestSuite(DoubleIRSSORTest.class);
        suite.addTestSuite(DoubleIRILUTest.class);
        suite.addTestSuite(DoubleIRICCTest.class);
        suite.addTestSuite(DoubleIRAMGTest.class);
        suite.addTestSuite(DoubleIRILUTTest.class);

        return suite;
    }
}
