package cern.colt.matrix.tfloat.impl;


public class DenseColFloatMatrix2DViewTest extends DenseColFloatMatrix2DTest {

    public DenseColFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseColFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseColFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseColFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
