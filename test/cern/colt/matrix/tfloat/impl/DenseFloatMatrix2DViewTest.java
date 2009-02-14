package cern.colt.matrix.tfloat.impl;


public class DenseFloatMatrix2DViewTest extends DenseFloatMatrix2DTest {

    public DenseFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
