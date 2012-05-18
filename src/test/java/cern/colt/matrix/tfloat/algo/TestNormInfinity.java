package cern.colt.matrix.tfloat.algo;

import cern.colt.matrix.tfloat.FloatFactory1D;
import cern.colt.matrix.tfloat.FloatMatrix1D;

class TestNormInfinity {

    public static void main(String[] args) {
        FloatMatrix1D x1 = FloatFactory1D.dense.make(new float[] { 1.0f, 2.0f });
        FloatMatrix1D x2 = FloatFactory1D.dense.make(new float[] { 1.0f, -2.0f });
        FloatMatrix1D x3 = FloatFactory1D.dense.make(new float[] { -1.0f, -2.0f });

        System.out.println(DenseFloatAlgebra.DEFAULT.normInfinity(x1));
        System.out.println(DenseFloatAlgebra.DEFAULT.normInfinity(x2));
        System.out.println(DenseFloatAlgebra.DEFAULT.normInfinity(x3));
    }
}