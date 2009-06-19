package cern.colt.matrix.tdouble.impl;

public class DenseLargeDoubleMatrix2DViewTest extends DenseLargeDoubleMatrix2DTest {

    public DenseLargeDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseLargeDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseLargeDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseLargeDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}
