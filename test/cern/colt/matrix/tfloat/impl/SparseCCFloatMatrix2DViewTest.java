package cern.colt.matrix.tfloat.impl;

public class SparseCCFloatMatrix2DViewTest extends SparseCCFloatMatrix2DTest {

    public SparseCCFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseCCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
