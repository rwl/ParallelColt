package cern.colt.matrix.tfloat.impl;

public class SparseFloatMatrix2DViewTest extends SparseFloatMatrix2DTest {

    public SparseFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}