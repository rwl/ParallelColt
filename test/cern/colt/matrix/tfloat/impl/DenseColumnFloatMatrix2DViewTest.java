package cern.colt.matrix.tfloat.impl;

public class DenseColumnFloatMatrix2DViewTest extends DenseColumnFloatMatrix2DTest {

    public DenseColumnFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseColumnFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseColumnFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseColumnFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
