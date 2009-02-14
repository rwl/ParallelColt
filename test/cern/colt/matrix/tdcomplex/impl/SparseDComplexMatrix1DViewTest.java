package cern.colt.matrix.tdcomplex.impl;

public class SparseDComplexMatrix1DViewTest extends SparseDComplexMatrix1DTest {
    public SparseDComplexMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseDComplexMatrix1D(SIZE).viewFlip();
        B = new SparseDComplexMatrix1D(SIZE).viewFlip();
    }
}
