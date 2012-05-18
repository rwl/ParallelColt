package cern.colt.matrix.tdcomplex.impl;

public class DenseDComplexMatrix1DViewTest extends DenseDComplexMatrix1DTest {
    public DenseDComplexMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseDComplexMatrix1D(SIZE).viewFlip();
        B = new DenseDComplexMatrix1D(SIZE).viewFlip();
    }
}
