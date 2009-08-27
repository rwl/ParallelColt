package cern.colt.matrix.tfcomplex;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tfcomplex.impl.DenseColumnFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.DenseColumnFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1DTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1DViewTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix3DTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix3DViewTest;
import cern.colt.matrix.tfcomplex.impl.DiagonalFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.DiagonalFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.LargeDenseFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.LargeDenseFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.LargeDenseFComplexMatrix3DTest;
import cern.colt.matrix.tfcomplex.impl.LargeDenseFComplexMatrix3DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseCCFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.SparseCCFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseCCMFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.SparseCCMFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix1DTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix1DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix3DTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix3DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseRCFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.SparseRCFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseRCMFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.SparseRCMFComplexMatrix2DViewTest;

public class AllFComplexMatrixTests {
    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tfcomplex tests");
        suite.addTestSuite(DenseFComplexMatrix1DTest.class);
        suite.addTestSuite(DenseFComplexMatrix1DViewTest.class);
        suite.addTestSuite(SparseFComplexMatrix1DTest.class);
        suite.addTestSuite(SparseFComplexMatrix1DViewTest.class);

        suite.addTestSuite(DenseFComplexMatrix2DTest.class);
        suite.addTestSuite(DenseFComplexMatrix2DViewTest.class);
        suite.addTestSuite(DenseColumnFComplexMatrix2DTest.class);
        suite.addTestSuite(DenseColumnFComplexMatrix2DViewTest.class);
        suite.addTestSuite(LargeDenseFComplexMatrix2DTest.class);
        suite.addTestSuite(LargeDenseFComplexMatrix2DViewTest.class);

        suite.addTestSuite(SparseCCFComplexMatrix2DTest.class);
        suite.addTestSuite(SparseCCFComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseRCFComplexMatrix2DTest.class);
        suite.addTestSuite(SparseRCFComplexMatrix2DViewTest.class);

        suite.addTestSuite(SparseCCMFComplexMatrix2DTest.class);
        suite.addTestSuite(SparseCCMFComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseRCMFComplexMatrix2DTest.class);
        suite.addTestSuite(SparseRCMFComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseFComplexMatrix2DTest.class);
        suite.addTestSuite(SparseFComplexMatrix2DViewTest.class);
        suite.addTestSuite(DiagonalFComplexMatrix2DTest.class);
        suite.addTestSuite(DiagonalFComplexMatrix2DViewTest.class);

        suite.addTestSuite(DenseFComplexMatrix3DTest.class);
        suite.addTestSuite(DenseFComplexMatrix3DViewTest.class);
        suite.addTestSuite(SparseFComplexMatrix3DTest.class);
        suite.addTestSuite(SparseFComplexMatrix3DViewTest.class);
        suite.addTestSuite(LargeDenseFComplexMatrix3DTest.class);
        suite.addTestSuite(LargeDenseFComplexMatrix3DViewTest.class);

        return suite;
    }

}
