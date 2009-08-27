package cern.colt.matrix.tfloat.impl;

public class DenseFloatMatrix1DViewTest extends DenseFloatMatrix1DTest {

    public DenseFloatMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseFloatMatrix1D(SIZE).viewFlip();
        B = new DenseFloatMatrix1D(SIZE).viewFlip();
    }
}
