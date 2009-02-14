package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix1DTest;

public class SparseDComplexMatrix1DTest extends DComplexMatrix1DTest {
    
    public SparseDComplexMatrix1DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseDComplexMatrix1D(SIZE);
        B = new SparseDComplexMatrix1D(SIZE);
    }
    
}
