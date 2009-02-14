package cern.colt.matrix.tdouble;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tdouble.algo.solver.AllDoubleMatrixSolverTests;
import cern.colt.matrix.tdouble.impl.CCDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.CCDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.CCMDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.CCMDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.DenseColDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.DenseColDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1DTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1DViewTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3DTest;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3DViewTest;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.RCDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.RCMDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.RCMDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix1DTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix1DViewTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2DTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2DViewTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix3DTest;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix3DViewTest;

public class AllDoubleMatrixTests {
    
    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tdouble tests");
        suite.addTestSuite(DenseDoubleMatrix1DTest.class);
        suite.addTestSuite(DenseDoubleMatrix1DViewTest.class);
        suite.addTestSuite(SparseDoubleMatrix1DTest.class);
        suite.addTestSuite(SparseDoubleMatrix1DViewTest.class);
        
        suite.addTestSuite(DenseDoubleMatrix2DTest.class);
        suite.addTestSuite(DenseDoubleMatrix2DViewTest.class);   
        suite.addTestSuite(DenseColDoubleMatrix2DTest.class);
        suite.addTestSuite(DenseColDoubleMatrix2DViewTest.class);
        suite.addTestSuite(SparseDoubleMatrix2DTest.class);
        suite.addTestSuite(SparseDoubleMatrix2DViewTest.class);
               
        suite.addTestSuite(RCDoubleMatrix2DTest.class);
        suite.addTestSuite(RCDoubleMatrix2DViewTest.class);        
        suite.addTestSuite(RCMDoubleMatrix2DTest.class);
        suite.addTestSuite(RCMDoubleMatrix2DViewTest.class);   
        
        suite.addTestSuite(CCDoubleMatrix2DTest.class);
        suite.addTestSuite(CCDoubleMatrix2DViewTest.class);
        suite.addTestSuite(CCMDoubleMatrix2DTest.class);
        suite.addTestSuite(CCMDoubleMatrix2DViewTest.class);   
              
        suite.addTestSuite(DenseDoubleMatrix3DTest.class);   
        suite.addTestSuite(DenseDoubleMatrix3DViewTest.class);   
        suite.addTestSuite(SparseDoubleMatrix3DTest.class);   
        suite.addTestSuite(SparseDoubleMatrix3DViewTest.class);   
        
        suite.addTest(AllDoubleMatrixSolverTests.suite());

        return suite;
    }
}
