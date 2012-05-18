package cern.colt.matrix.tlong.impl;

public class DenseLargeLongMatrix2DViewTest extends DenseLargeLongMatrix2DTest {

    public DenseLargeLongMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseLargeLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseLargeLongMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}
