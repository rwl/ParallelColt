package cern.colt.matrix.tfloat.impl;

public class SparseRCMFloatMatrix2DViewTest extends SparseRCMFloatMatrix2DTest {

    public SparseRCMFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseRCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCMFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
