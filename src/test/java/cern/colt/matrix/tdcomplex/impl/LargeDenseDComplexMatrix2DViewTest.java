package cern.colt.matrix.tdcomplex.impl;

public class LargeDenseDComplexMatrix2DViewTest extends LargeDenseDComplexMatrix2DTest {
    public LargeDenseDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseLargeDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseLargeDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
