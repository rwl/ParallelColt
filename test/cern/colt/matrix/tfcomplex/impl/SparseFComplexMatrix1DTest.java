package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix1DTest;

public class SparseFComplexMatrix1DTest extends FComplexMatrix1DTest {

    public SparseFComplexMatrix1DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseFComplexMatrix1D(SIZE);
        B = new SparseFComplexMatrix1D(SIZE);
    }

}
