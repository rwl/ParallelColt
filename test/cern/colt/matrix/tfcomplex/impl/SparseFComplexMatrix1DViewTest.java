package cern.colt.matrix.tfcomplex.impl;

public class SparseFComplexMatrix1DViewTest extends SparseFComplexMatrix1DTest {
    public SparseFComplexMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseFComplexMatrix1D(SIZE).viewFlip();
        B = new SparseFComplexMatrix1D(SIZE).viewFlip();
    }
}
