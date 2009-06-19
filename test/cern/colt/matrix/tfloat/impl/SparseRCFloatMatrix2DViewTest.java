package cern.colt.matrix.tfloat.impl;

public class SparseRCFloatMatrix2DViewTest extends SparseRCFloatMatrix2DTest {

    public SparseRCFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseRCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
