package cern.colt.matrix.tlong;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tlong.impl.DenseColumnLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.DenseColumnLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.DenseLargeLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.DenseLargeLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.DenseLargeLongMatrix3DTest;
import cern.colt.matrix.tlong.impl.DenseLargeLongMatrix3DViewTest;
import cern.colt.matrix.tlong.impl.DenseLongMatrix1DTest;
import cern.colt.matrix.tlong.impl.DenseLongMatrix1DViewTest;
import cern.colt.matrix.tlong.impl.DenseLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.DenseLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.DenseLongMatrix3DTest;
import cern.colt.matrix.tlong.impl.DenseLongMatrix3DViewTest;
import cern.colt.matrix.tlong.impl.DiagonalLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.DiagonalLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.SparseCCLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.SparseCCLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.SparseCCMLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.SparseCCMLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.SparseLongMatrix1DTest;
import cern.colt.matrix.tlong.impl.SparseLongMatrix1DViewTest;
import cern.colt.matrix.tlong.impl.SparseLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.SparseLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.SparseLongMatrix3DTest;
import cern.colt.matrix.tlong.impl.SparseLongMatrix3DViewTest;
import cern.colt.matrix.tlong.impl.SparseRCLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.SparseRCLongMatrix2DViewTest;
import cern.colt.matrix.tlong.impl.SparseRCMLongMatrix2DTest;
import cern.colt.matrix.tlong.impl.SparseRCMLongMatrix2DViewTest;

public class AllLongMatrixTests {

    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tlong tests");
        suite.addTestSuite(DenseLongMatrix1DTest.class);
        suite.addTestSuite(DenseLongMatrix1DViewTest.class);
        suite.addTestSuite(SparseLongMatrix1DTest.class);
        suite.addTestSuite(SparseLongMatrix1DViewTest.class);

        suite.addTestSuite(DenseLongMatrix2DTest.class);
        suite.addTestSuite(DenseLongMatrix2DViewTest.class);
        suite.addTestSuite(DenseColumnLongMatrix2DTest.class);
        suite.addTestSuite(DenseColumnLongMatrix2DViewTest.class);
        suite.addTestSuite(DenseLargeLongMatrix2DTest.class);
        suite.addTestSuite(DenseLargeLongMatrix2DViewTest.class);

        suite.addTestSuite(SparseLongMatrix2DTest.class);
        suite.addTestSuite(SparseLongMatrix2DViewTest.class);
        suite.addTestSuite(DiagonalLongMatrix2DTest.class);
        suite.addTestSuite(DiagonalLongMatrix2DViewTest.class);

        suite.addTestSuite(SparseRCLongMatrix2DTest.class);
        suite.addTestSuite(SparseRCLongMatrix2DViewTest.class);
        suite.addTestSuite(SparseRCMLongMatrix2DTest.class);
        suite.addTestSuite(SparseRCMLongMatrix2DViewTest.class);

        suite.addTestSuite(SparseCCLongMatrix2DTest.class);
        suite.addTestSuite(SparseCCLongMatrix2DViewTest.class);
        suite.addTestSuite(SparseCCMLongMatrix2DTest.class);
        suite.addTestSuite(SparseCCMLongMatrix2DViewTest.class);

        suite.addTestSuite(DenseLongMatrix3DTest.class);
        suite.addTestSuite(DenseLongMatrix3DViewTest.class);
        suite.addTestSuite(SparseLongMatrix3DTest.class);
        suite.addTestSuite(SparseLongMatrix3DViewTest.class);
        suite.addTestSuite(DenseLargeLongMatrix3DTest.class);
        suite.addTestSuite(DenseLargeLongMatrix3DViewTest.class);

        return suite;
    }
}
