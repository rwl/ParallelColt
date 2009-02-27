package cern.colt.matrix.tfloat;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tfloat.algo.solver.AllFloatMatrixSolverTests;
import cern.colt.matrix.tfloat.impl.CCFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.CCFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.CCMFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.CCMFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.DenseColFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.DenseColFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1DTest;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1DViewTest;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix3DTest;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix3DViewTest;
import cern.colt.matrix.tfloat.impl.DiagonalFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.DiagonalFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.RCFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.RCFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.RCMFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.RCMFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix1DTest;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix1DViewTest;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix2DTest;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix2DViewTest;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix3DTest;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix3DViewTest;

public class AllFloatMatrixTests {

    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tfloat tests");
        suite.addTestSuite(DenseFloatMatrix1DTest.class);
        suite.addTestSuite(DenseFloatMatrix1DViewTest.class);
        suite.addTestSuite(SparseFloatMatrix1DTest.class);
        suite.addTestSuite(SparseFloatMatrix1DViewTest.class);

        suite.addTestSuite(DenseFloatMatrix2DTest.class);
        suite.addTestSuite(DenseFloatMatrix2DViewTest.class);
        suite.addTestSuite(DenseColFloatMatrix2DTest.class);
        suite.addTestSuite(DenseColFloatMatrix2DViewTest.class);
        suite.addTestSuite(SparseFloatMatrix2DTest.class);
        suite.addTestSuite(SparseFloatMatrix2DViewTest.class);
        suite.addTestSuite(DiagonalFloatMatrix2DTest.class);
        suite.addTestSuite(DiagonalFloatMatrix2DViewTest.class);

        suite.addTestSuite(RCFloatMatrix2DTest.class);
        suite.addTestSuite(RCFloatMatrix2DViewTest.class);
        suite.addTestSuite(RCMFloatMatrix2DTest.class);
        suite.addTestSuite(RCMFloatMatrix2DViewTest.class);

        suite.addTestSuite(CCFloatMatrix2DTest.class);
        suite.addTestSuite(CCFloatMatrix2DViewTest.class);
        suite.addTestSuite(CCMFloatMatrix2DTest.class);
        suite.addTestSuite(CCMFloatMatrix2DViewTest.class);

        suite.addTestSuite(DenseFloatMatrix3DTest.class);
        suite.addTestSuite(DenseFloatMatrix3DViewTest.class);
        suite.addTestSuite(SparseFloatMatrix3DTest.class);
        suite.addTestSuite(SparseFloatMatrix3DViewTest.class);

        suite.addTest(AllFloatMatrixSolverTests.suite());

        return suite;
    }
}
