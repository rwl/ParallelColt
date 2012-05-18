package cern.colt.matrix.tdouble;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tdouble.algo.solver.AllDoubleMatrixSolverTests;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1DTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1DViewTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3DTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3DViewTest;
import cern.colt.matrix.tdouble.impl.DenseLargeDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.DenseLargeDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.DenseLargeDoubleMatrix3DTest;
import cern.colt.matrix.tdouble.impl.DenseLargeDoubleMatrix3DViewTest;
import cern.colt.matrix.tdouble.impl.DiagonalDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.DiagonalDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.SparseCCDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.SparseCCDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.SparseCCMDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.SparseCCMDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix1DTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix1DViewTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix3DTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix3DViewTest;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.SparseRCMDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.SparseRCMDoubleMatrix2DViewTest;

public class AllDoubleMatrixTests {

    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tdouble tests");
        suite.addTestSuite(DenseDoubleMatrix1DTest.class);
        suite.addTestSuite(DenseDoubleMatrix1DViewTest.class);
        suite.addTestSuite(SparseDoubleMatrix1DTest.class);
        suite.addTestSuite(SparseDoubleMatrix1DViewTest.class);

        suite.addTestSuite(DenseDoubleMatrix2DTest.class);
        suite.addTestSuite(DenseDoubleMatrix2DViewTest.class);
        suite.addTestSuite(DenseColumnDoubleMatrix2DTest.class);
        suite.addTestSuite(DenseColumnDoubleMatrix2DViewTest.class);
        suite.addTestSuite(DenseLargeDoubleMatrix2DTest.class);
        suite.addTestSuite(DenseLargeDoubleMatrix2DViewTest.class);

        suite.addTestSuite(SparseDoubleMatrix2DTest.class);
        suite.addTestSuite(SparseDoubleMatrix2DViewTest.class);
        suite.addTestSuite(DiagonalDoubleMatrix2DTest.class);
        suite.addTestSuite(DiagonalDoubleMatrix2DViewTest.class);

        suite.addTestSuite(SparseRCDoubleMatrix2DTest.class);
        suite.addTestSuite(SparseRCDoubleMatrix2DViewTest.class);
        suite.addTestSuite(SparseRCMDoubleMatrix2DTest.class);
        suite.addTestSuite(SparseRCMDoubleMatrix2DViewTest.class);

        suite.addTestSuite(SparseCCDoubleMatrix2DTest.class);
        suite.addTestSuite(SparseCCDoubleMatrix2DViewTest.class);
        suite.addTestSuite(SparseCCMDoubleMatrix2DTest.class);
        suite.addTestSuite(SparseCCMDoubleMatrix2DViewTest.class);

        suite.addTestSuite(DenseDoubleMatrix3DTest.class);
        suite.addTestSuite(DenseDoubleMatrix3DViewTest.class);
        suite.addTestSuite(SparseDoubleMatrix3DTest.class);
        suite.addTestSuite(SparseDoubleMatrix3DViewTest.class);
        suite.addTestSuite(DenseLargeDoubleMatrix3DTest.class);
        suite.addTestSuite(DenseLargeDoubleMatrix3DViewTest.class);

        suite.addTest(AllDoubleMatrixSolverTests.suite());

        return suite;
    }
}
