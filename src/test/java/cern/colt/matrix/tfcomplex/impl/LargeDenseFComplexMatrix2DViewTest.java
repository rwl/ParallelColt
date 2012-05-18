package cern.colt.matrix.tfcomplex.impl;

public class LargeDenseFComplexMatrix2DViewTest extends LargeDenseFComplexMatrix2DTest {
    public LargeDenseFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseLargeFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseLargeFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
