package cern.colt.matrix.tfloat.impl;

public class SparseCCMFloatMatrix2DViewTest extends SparseCCMFloatMatrix2DTest {

    public SparseCCMFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseCCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCMFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
