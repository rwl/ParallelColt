package cern.colt.matrix.tfcomplex.impl;

public class DenseFComplexMatrix1DViewTest extends DenseFComplexMatrix1DTest {
    public DenseFComplexMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseFComplexMatrix1D(SIZE).viewFlip();
        B = new DenseFComplexMatrix1D(SIZE).viewFlip();
    }
}
