package cern.colt.matrix.tint.impl;

public class DenseLargeIntMatrix2DViewTest extends DenseLargeIntMatrix2DTest {

    public DenseLargeIntMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseLargeIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseLargeIntMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}
