package cern.colt.matrix.tlong.impl;

public class DenseColumnLongMatrix2DViewTest extends DenseColumnLongMatrix2DTest {

    public DenseColumnLongMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseColumnLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseColumnLongMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
