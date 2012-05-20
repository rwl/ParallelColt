package cern.colt.matrix.tdcomplex.algo;

import cern.colt.matrix.tdcomplex.DComplexFactory1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;

class TestNormInfinity {

    public static void main(String[] args) {
        DComplexMatrix1D x1 = DComplexFactory1D.dense.make(new double[] { 3.0, 4.0,  2.0,  0.0 });
        DComplexMatrix1D x2 = DComplexFactory1D.dense.make(new double[] { 1.0, 0.0, -3.0,  4.0 });
        DComplexMatrix1D x3 = DComplexFactory1D.dense.make(new double[] {-1.0, 0.0, -3.0, -4.0 });

        System.out.println(DenseDComplexAlgebra.DEFAULT.normInfinity(x1));
        System.out.println(DenseDComplexAlgebra.DEFAULT.normInfinity(x2));
        System.out.println(DenseDComplexAlgebra.DEFAULT.normInfinity(x3));
    }
}