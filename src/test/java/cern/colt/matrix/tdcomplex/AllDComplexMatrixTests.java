package cern.colt.matrix.tdcomplex;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tdcomplex.impl.DenseColumnDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.DenseColumnDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1DTest;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1DViewTest;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix3DTest;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix3DViewTest;
import cern.colt.matrix.tdcomplex.impl.DiagonalDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.DiagonalDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.LargeDenseDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.LargeDenseDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.LargeDenseDComplexMatrix3DTest;
import cern.colt.matrix.tdcomplex.impl.LargeDenseDComplexMatrix3DViewTest;
import cern.colt.matrix.tdcomplex.impl.SparseCCDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.SparseCCDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.SparseCCMDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.SparseCCMDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix1DTest;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix1DViewTest;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix3DTest;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix3DViewTest;
import cern.colt.matrix.tdcomplex.impl.SparseRCDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.SparseRCDComplexMatrix2DViewTest;
import cern.colt.matrix.tdcomplex.impl.SparseRCMDComplexMatrix2DTest;
import cern.colt.matrix.tdcomplex.impl.SparseRCMDComplexMatrix2DViewTest;

public class AllDComplexMatrixTests {
    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tdcomplex tests");
        suite.addTestSuite(DenseDComplexMatrix1DTest.class);
        suite.addTestSuite(DenseDComplexMatrix1DViewTest.class);
        suite.addTestSuite(SparseDComplexMatrix1DTest.class);
        suite.addTestSuite(SparseDComplexMatrix1DViewTest.class);

        suite.addTestSuite(DenseDComplexMatrix2DTest.class);
        suite.addTestSuite(DenseDComplexMatrix2DViewTest.class);
        suite.addTestSuite(DenseColumnDComplexMatrix2DTest.class);
        suite.addTestSuite(DenseColumnDComplexMatrix2DViewTest.class);
        suite.addTestSuite(LargeDenseDComplexMatrix2DTest.class);
        suite.addTestSuite(LargeDenseDComplexMatrix2DViewTest.class);

        suite.addTestSuite(SparseDComplexMatrix2DTest.class);
        suite.addTestSuite(SparseDComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseCCDComplexMatrix2DTest.class);
        suite.addTestSuite(SparseCCDComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseCCMDComplexMatrix2DTest.class);
        suite.addTestSuite(SparseCCMDComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseRCDComplexMatrix2DTest.class);
        suite.addTestSuite(SparseRCDComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseRCMDComplexMatrix2DTest.class);
        suite.addTestSuite(SparseRCMDComplexMatrix2DViewTest.class);
        suite.addTestSuite(DiagonalDComplexMatrix2DTest.class);
        suite.addTestSuite(DiagonalDComplexMatrix2DViewTest.class);

        suite.addTestSuite(DenseDComplexMatrix3DTest.class);
        suite.addTestSuite(DenseDComplexMatrix3DViewTest.class);
        suite.addTestSuite(SparseDComplexMatrix3DTest.class);
        suite.addTestSuite(SparseDComplexMatrix3DViewTest.class);
        suite.addTestSuite(LargeDenseDComplexMatrix3DTest.class);
        suite.addTestSuite(LargeDenseDComplexMatrix3DViewTest.class);

        return suite;
    }

}
