package cern.colt.matrix.tdouble.algo.decomposition;

import java.util.Random;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

public class TestDenseDoubleSVD {
    public static void main(String[] args) {
        int M = 60;
        int N = 40;
        int NRHS = 1;
        Random r = new Random(0);

        DoubleMatrix2D A1 = new DenseDoubleMatrix2D(M, N);
        DoubleMatrix2D A2 = new DenseDoubleMatrix2D(M, N);
        DoubleMatrix2D B1 = new DenseDoubleMatrix2D(M, NRHS);
        DoubleMatrix2D B2 = new DenseDoubleMatrix2D(M, NRHS);

        /* Initialize A1 and A2*/
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                A1.setQuick(i, j, 0.5 - r.nextDouble());
                A2.setQuick(i, j, A1.getQuick(i, j));
            }
        }

        /* Initialize B1 and B2 */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < NRHS; j++) {
                B1.setQuick(i, j, 0.5 - r.nextDouble());
                B2.setQuick(i, j, B1.getQuick(i, j));
            }
        }

        testSVD(A1, A2, B1, B2);
    }

    private static void testSVD(DoubleMatrix2D A1, DoubleMatrix2D A2, DoubleMatrix2D B1, DoubleMatrix2D B2) {
        int M = A1.rows();
        int N = A1.columns();
        double eps = 1e-10;

        DenseDoubleSingularValueDecomposition svd = new DenseDoubleSingularValueDecomposition(A2, true, false);
        DoubleMatrix2D S = svd.getS();
        DoubleMatrix2D V = svd.getV();
        DoubleMatrix2D U = svd.getU();
        System.out.println(svd.toString());

        System.out.print("\n");
        System.out.print("------ DenseDoubleSingularValueDecomposition tests-------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", M, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the factorization */
        int info_factorization = checkFactorization(A1, U, S, V, eps);

        if (info_factorization == 0) {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DenseDoubleSingularValueDecomposition .... PASSED !\n");
            System.out.print("************************************************\n");
        } else {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DenseDoubleSingularValueDecomposition .... FAILED !\n");
            System.out.print("************************************************\n");
        }

    }

    private static int checkFactorization(DoubleMatrix2D A1, DoubleMatrix2D U, DoubleMatrix2D S, DoubleMatrix2D V,
            double eps) {
        DoubleProperty.DEFAULT.checkDense(A1);
        DoubleProperty.DEFAULT.checkDense(U);
        DoubleProperty.DEFAULT.checkDense(V);
        int M = A1.rows();
        int N = A1.columns();
        int info_factorization;
        double Anorm, Rnorm;
        double alpha;
        DoubleMatrix2D Residual;
        DoubleMatrix2D A2 = U.copy();

        alpha = 1.0;

        Residual = A1.copy();

        A2 = A2.zMult(S, null, alpha, 0, false, false); // A2 = US

        A2 = A2.zMult(V, null, alpha, 0, false, true); // A2 = USV'

        /* Compute the Residual ||USV'-A|| */
        Residual.assign(A2, DoubleFunctions.plusMultFirst(-1));

        Rnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(Residual);
        Anorm = DenseDoubleAlgebra.DEFAULT.normInfinity(A1);

        System.out.print("============\n");
        System.out.print("Checking the SVD Factorization \n");
        System.out.print(String.format("-- ||USV'-A||_oo/(||A||_oo.N.eps) = %e \n", Rnorm / (Anorm * N * eps)));

        if (Rnorm / (Anorm * N * eps) > 10.0) {
            System.out.print("-- Factorization is suspicious ! \n");
            info_factorization = 1;
        } else {
            System.out.print("-- Factorization is CORRECT ! \n");
            info_factorization = 0;
        }
        return info_factorization;
    }

}
