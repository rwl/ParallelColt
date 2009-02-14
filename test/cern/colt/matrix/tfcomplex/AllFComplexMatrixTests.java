package cern.colt.matrix.tfcomplex;

import junit.framework.Test;
import junit.framework.TestSuite;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1DTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1DViewTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix3DTest;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix3DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix1DTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix1DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix2DTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix2DViewTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix3DTest;
import cern.colt.matrix.tfcomplex.impl.SparseFComplexMatrix3DViewTest;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class AllFComplexMatrixTests {
    public static Test suite() {
        TestSuite suite = new TestSuite("cern.colt.matrix.tfcomplex tests");
        suite.addTestSuite(DenseFComplexMatrix1DTest.class);
        suite.addTestSuite(DenseFComplexMatrix1DViewTest.class);
        suite.addTestSuite(SparseFComplexMatrix1DTest.class);
        suite.addTestSuite(SparseFComplexMatrix1DViewTest.class);
        
        suite.addTestSuite(DenseFComplexMatrix2DTest.class);
        suite.addTestSuite(DenseFComplexMatrix2DViewTest.class);
        suite.addTestSuite(SparseFComplexMatrix2DTest.class);
        suite.addTestSuite(SparseFComplexMatrix2DViewTest.class);
        
        suite.addTestSuite(DenseFComplexMatrix3DTest.class);
        suite.addTestSuite(DenseFComplexMatrix3DViewTest.class);
        suite.addTestSuite(SparseFComplexMatrix3DTest.class);
        suite.addTestSuite(SparseFComplexMatrix3DViewTest.class);
        
        return suite;
    }

}
