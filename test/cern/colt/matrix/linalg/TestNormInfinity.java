package cern.colt.matrix.linalg;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

class TestNormInfinity {

    public static void main(String[] args) {
        DoubleMatrix1D x1 = DoubleFactory1D.dense.make(new double[] { 1.0, 2.0 });
        DoubleMatrix1D x2 = DoubleFactory1D.dense.make(new double[] { 1.0, -2.0 });
        DoubleMatrix1D x3 = DoubleFactory1D.dense.make(new double[] { -1.0, -2.0 });

        System.out.println(DoubleAlgebra.DEFAULT.normInfinity(x1));
        System.out.println(DoubleAlgebra.DEFAULT.normInfinity(x2));
        System.out.println(DoubleAlgebra.DEFAULT.normInfinity(x3));
    }
}