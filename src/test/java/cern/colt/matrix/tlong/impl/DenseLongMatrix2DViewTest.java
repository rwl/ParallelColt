package cern.colt.matrix.tlong.impl;

public class DenseLongMatrix2DViewTest extends DenseLongMatrix2DTest {

    public DenseLongMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseLongMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
