package cern.colt.matrix.tdouble.algo.decomposition;

import java.util.Random;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.algo.SparseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2D;

public class TestSparseDoubleLUDecomposition {

    public static void main(String[] args) {
        int N = 180;
        Random r = new Random(0);

        DoubleMatrix2D A1 = new SparseRCDoubleMatrix2D(N, N);
        DoubleMatrix2D A2 = new SparseRCDoubleMatrix2D(N, N);
        DoubleMatrix1D B1 = new DenseDoubleMatrix1D(N);
        DoubleMatrix1D B2 = new DenseDoubleMatrix1D(N);

        /* Initialize A1 and A2*/
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A1.setQuick(i, j, 0.5 - r.nextDouble());
                A2.setQuick(i, j, A1.getQuick(i, j));
            }
        }

        /* Initialize B1 and B2 */
        for (int i = 0; i < N; i++) {
            B1.setQuick(i, 0.5 - r.nextDouble());
            B2.setQuick(i, B1.getQuick(i));
        }

        testLU(A1, A2, B1, B2);

    }

    private static void testLU(DoubleMatrix2D A1, DoubleMatrix2D A2, DoubleMatrix1D B1, DoubleMatrix1D B2) {
        int M = A1.rows();
        int N = A1.columns();
        double eps = 1e-10;

        SparseDoubleLUDecomposition lu = new SparseDoubleLUDecomposition(A2, 0, true);
        //        DoubleMatrix2D L = lu.getL();
        //        DoubleMatrix2D U = lu.getU();
        DoubleMatrix1D X = B2.copy();
        lu.solve(X);

        System.out.print("\n");
        System.out.print("------ SparseDoubleLUFactorization tests-------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", M, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the factorization and the solution */
        int info_solution = checkSolution(A1, B1, X, eps);

        if ((info_solution == 0)) {
            System.out.print("***************************************************\n");
            System.out.print(" ---- SparseDoubleLUFactorization tests... PASSED !\n");
            System.out.print("***************************************************\n");
        } else {
            System.out.print("***************************************************\n");
            System.out.print(" ---- SparseDoubleLUFactorization tests... FAILED !\n");
            System.out.print("***************************************************\n");
        }
    }

    /*------------------------------------------------------------------------
     *  Check the accuracy of the solution of the linear system 
     */

    private static int checkSolution(DoubleMatrix2D A1, DoubleMatrix1D B1, DoubleMatrix1D B2, double eps) {
        DoubleProperty.DEFAULT.checkSparse(A1);
        DoubleProperty.DEFAULT.checkDense(B1);
        DoubleProperty.DEFAULT.checkDense(B2);
        int N = A1.rows();
        int info_solution;
        double Rnorm, Anorm, Xnorm, Bnorm;
        double alpha, beta;

        alpha = 1.0;
        beta = -1.0;

        Xnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(B2);
        Anorm = SparseDoubleAlgebra.DEFAULT.normInfinity(A1);
        Bnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(B1);

        //B1 = A1*B2 - B1;
        A1.zMult(B2, B1, alpha, beta, false);
        Rnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(B1);

        System.out.print("============\n");
        System.out.print("Checking the Residual of the solution \n");
        System.out.print(String.format("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", Rnorm
                / ((Anorm * Xnorm + Bnorm) * N * eps)));

        if (Rnorm / ((Anorm * Xnorm + Bnorm) * N * eps) > 10.0) {
            System.out.print("-- The solution is suspicious ! \n");
            info_solution = 1;
        } else {
            System.out.print("-- The solution is CORRECT ! \n");
            info_solution = 0;
        }

        return info_solution;
    }

}
