package cern.colt.matrix.tdcomplex.impl;

public class DenseDComplexMatrix2DViewTest extends DenseDComplexMatrix2DTest {
    public DenseDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
