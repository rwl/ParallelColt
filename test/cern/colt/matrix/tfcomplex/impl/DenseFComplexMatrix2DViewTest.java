package cern.colt.matrix.tfcomplex.impl;

public class DenseFComplexMatrix2DViewTest extends DenseFComplexMatrix2DTest {
    public DenseFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
