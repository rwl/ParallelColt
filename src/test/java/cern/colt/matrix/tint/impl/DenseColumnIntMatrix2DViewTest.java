package cern.colt.matrix.tint.impl;

public class DenseColumnIntMatrix2DViewTest extends DenseColumnIntMatrix2DTest {

    public DenseColumnIntMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseColumnIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseColumnIntMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
