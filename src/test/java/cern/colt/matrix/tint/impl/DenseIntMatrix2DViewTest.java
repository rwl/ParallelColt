package cern.colt.matrix.tint.impl;

public class DenseIntMatrix2DViewTest extends DenseIntMatrix2DTest {

    public DenseIntMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseIntMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
