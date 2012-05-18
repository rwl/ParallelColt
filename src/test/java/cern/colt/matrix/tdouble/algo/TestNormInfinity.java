package cern.colt.matrix.tdouble.algo;

import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;

class TestNormInfinity {

    public static void main(String[] args) {
        DoubleMatrix1D x1 = DoubleFactory1D.dense.make(new double[] { 1.0, 2.0 });
        DoubleMatrix1D x2 = DoubleFactory1D.dense.make(new double[] { 1.0, -2.0 });
        DoubleMatrix1D x3 = DoubleFactory1D.dense.make(new double[] { -1.0, -2.0 });

        System.out.println(DenseDoubleAlgebra.DEFAULT.normInfinity(x1));
        System.out.println(DenseDoubleAlgebra.DEFAULT.normInfinity(x2));
        System.out.println(DenseDoubleAlgebra.DEFAULT.normInfinity(x3));
    }
}