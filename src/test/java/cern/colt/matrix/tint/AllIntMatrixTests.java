package cern.colt.matrix.tint;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tint.impl.DenseColumnIntMatrix2DTest;
import cern.colt.matrix.tint.impl.DenseColumnIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.DenseIntMatrix1DTest;
import cern.colt.matrix.tint.impl.DenseIntMatrix1DViewTest;
import cern.colt.matrix.tint.impl.DenseIntMatrix2DTest;
import cern.colt.matrix.tint.impl.DenseIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.DenseIntMatrix3DTest;
import cern.colt.matrix.tint.impl.DenseIntMatrix3DViewTest;
import cern.colt.matrix.tint.impl.DenseLargeIntMatrix2DTest;
import cern.colt.matrix.tint.impl.DenseLargeIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.DenseLargeIntMatrix3DTest;
import cern.colt.matrix.tint.impl.DenseLargeIntMatrix3DViewTest;
import cern.colt.matrix.tint.impl.DiagonalIntMatrix2DTest;
import cern.colt.matrix.tint.impl.DiagonalIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.SparseCCIntMatrix2DTest;
import cern.colt.matrix.tint.impl.SparseCCIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.SparseCCMIntMatrix2DTest;
import cern.colt.matrix.tint.impl.SparseCCMIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.SparseIntMatrix1DTest;
import cern.colt.matrix.tint.impl.SparseIntMatrix1DViewTest;
import cern.colt.matrix.tint.impl.SparseIntMatrix2DTest;
import cern.colt.matrix.tint.impl.SparseIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.SparseIntMatrix3DTest;
import cern.colt.matrix.tint.impl.SparseIntMatrix3DViewTest;
import cern.colt.matrix.tint.impl.SparseRCIntMatrix2DTest;
import cern.colt.matrix.tint.impl.SparseRCIntMatrix2DViewTest;
import cern.colt.matrix.tint.impl.SparseRCMIntMatrix2DTest;
import cern.colt.matrix.tint.impl.SparseRCMIntMatrix2DViewTest;

public class AllIntMatrixTests {

    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tint tests");
        suite.addTestSuite(DenseIntMatrix1DTest.class);
        suite.addTestSuite(DenseIntMatrix1DViewTest.class);
        suite.addTestSuite(SparseIntMatrix1DTest.class);
        suite.addTestSuite(SparseIntMatrix1DViewTest.class);

        suite.addTestSuite(DenseIntMatrix2DTest.class);
        suite.addTestSuite(DenseIntMatrix2DViewTest.class);
        suite.addTestSuite(DenseColumnIntMatrix2DTest.class);
        suite.addTestSuite(DenseColumnIntMatrix2DViewTest.class);
        suite.addTestSuite(DenseLargeIntMatrix2DTest.class);
        suite.addTestSuite(DenseLargeIntMatrix2DViewTest.class);

        suite.addTestSuite(SparseIntMatrix2DTest.class);
        suite.addTestSuite(SparseIntMatrix2DViewTest.class);
        suite.addTestSuite(DiagonalIntMatrix2DTest.class);
        suite.addTestSuite(DiagonalIntMatrix2DViewTest.class);

        suite.addTestSuite(SparseRCIntMatrix2DTest.class);
        suite.addTestSuite(SparseRCIntMatrix2DViewTest.class);
        suite.addTestSuite(SparseRCMIntMatrix2DTest.class);
        suite.addTestSuite(SparseRCMIntMatrix2DViewTest.class);

        suite.addTestSuite(SparseCCIntMatrix2DTest.class);
        suite.addTestSuite(SparseCCIntMatrix2DViewTest.class);
        suite.addTestSuite(SparseCCMIntMatrix2DTest.class);
        suite.addTestSuite(SparseCCMIntMatrix2DViewTest.class);

        suite.addTestSuite(DenseIntMatrix3DTest.class);
        suite.addTestSuite(DenseIntMatrix3DViewTest.class);
        suite.addTestSuite(SparseIntMatrix3DTest.class);
        suite.addTestSuite(SparseIntMatrix3DViewTest.class);
        suite.addTestSuite(DenseLargeIntMatrix3DTest.class);
        suite.addTestSuite(DenseLargeIntMatrix3DViewTest.class);

        return suite;
    }
}
