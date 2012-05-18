package cern.colt.matrix.tdouble.algo.decomposition;

import java.util.Random;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import edu.emory.mathcs.jplasma.tdouble.Dplasma;

public class TestDenseDoubleCholeskyDecomposition {

    public static void main(String[] args) {

        for (int k = 0; k < 20; k++) {

            int N = 600;
            int NRHS = 5;
            Random r = new Random(0);

            DoubleMatrix2D A1 = new DenseColumnDoubleMatrix2D(N, N);
            DoubleMatrix2D A2 = new DenseColumnDoubleMatrix2D(N, N);
            DoubleMatrix2D B1 = new DenseColumnDoubleMatrix2D(N, NRHS);
            DoubleMatrix2D B2 = new DenseColumnDoubleMatrix2D(N, NRHS);

            /* Initialize A1 and A2 for Symmetric Positive Matrix */
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A1.setQuick(i, j, 0.5 - r.nextDouble());
                    A2.setQuick(i, j, A1.getQuick(i, j));
                }
            }

            for (int i = 0; i < N; i++) {
                A1.setQuick(i, i, A1.getQuick(i, i) + N);
                A2.setQuick(i, i, A1.getQuick(i, i));
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A1.setQuick(i, j, A1.getQuick(j, i));
                    A2.setQuick(i, j, A1.getQuick(j, i));
                }
            }
            /* Initialize B1 and B2 */
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < NRHS; j++) {
                    B1.setQuick(i, j, 0.5 - r.nextDouble());
                    B2.setQuick(i, j, B1.getQuick(i, j));
                }
            }

            testCholesky(A1, A2, B1, B2);

            A1 = new DenseDoubleMatrix2D(N, N);
            A2 = new DenseDoubleMatrix2D(N, N);
            B1 = new DenseDoubleMatrix2D(N, NRHS);
            B2 = new DenseDoubleMatrix2D(N, NRHS);
            r = new Random(0);

            /* Initialize A1 and A2 for Symmetric Positive Matrix */
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A1.setQuick(i, j, 0.5 - r.nextDouble());
                    A2.setQuick(i, j, A1.getQuick(i, j));
                }
            }

            for (int i = 0; i < N; i++) {
                A1.setQuick(i, i, A1.getQuick(i, i) + N);
                A2.setQuick(i, i, A1.getQuick(i, i));
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A1.setQuick(i, j, A1.getQuick(j, i));
                    A2.setQuick(i, j, A1.getQuick(j, i));
                }
            }
            /* Initialize B1 and B2 */
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < NRHS; j++) {
                    B1.setQuick(i, j, 0.5 - r.nextDouble());
                    B2.setQuick(i, j, B1.getQuick(i, j));
                }
            }

            testCholesky(A1, A2, B1, B2);
        }
        System.out.println("All finished");       
        System.exit(0);

    }

    private static void testCholesky(DoubleMatrix2D A1, DoubleMatrix2D A2, DoubleMatrix2D B1, DoubleMatrix2D B2) {
        int N = A1.rows();
        double eps = 1e-10;

        DenseDoubleCholeskyDecomposition cf = new DenseDoubleCholeskyDecomposition(A2);
        DoubleMatrix2D Lt = cf.getLtranspose();
        DoubleMatrix2D X = B2.copy();
        cf.solve(X);
        System.out.print("\n");
        System.out.print("------ DoubleCholeskyFactorization tests-------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", N, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the factorization and the solution */
        int info_factorization = checkFactorization(A1, Lt, eps);
        int info_solution = checkSolution(A1, B1, X, eps);

        if ((info_solution == 0) & (info_factorization == 0)) {
            System.out.print("***************************************************\n");
            System.out.print(" ---- DoubleCholeskyFactorization tests... PASSED !\n");
            System.out.print("***************************************************\n");
        } else {
            System.err.print("***************************************************\n");
            System.err.print(" ---- DoubleCholeskyFactorization tests... FAILED !\n");
            System.err.print("***************************************************\n");
        }
    }

    private static int checkFactorization(DoubleMatrix2D A1, DoubleMatrix2D A2, double eps) {
        DoubleProperty.DEFAULT.checkDense(A1);
        DoubleProperty.DEFAULT.checkDense(A2);
        int N = A1.rows();
        int LDA = N;
        int uplo = Dplasma.PlasmaUpper;
        double Anorm, Rnorm;
        double alpha;
        String norm = "I";
        int info_factorization;
        int i, j;
        double[] A1elems;
        double[] A2elems;

        if (A1 instanceof DenseDoubleMatrix2D) {
            A1elems = (double[]) A1.viewDice().copy().elements();
        } else {
            A1elems = (double[]) A1.copy().elements();
        }

        if (A2 instanceof DenseDoubleMatrix2D) {
            A2elems = (double[]) A2.viewDice().copy().elements();
        } else {
            A2elems = (double[]) A2.copy().elements();
        }

        double[] Residual = new double[N * N];
        double[] L1 = new double[N * N];
        double[] L2 = new double[N * N];
        double[] work = new double[N];

        alpha = 1.0;

        org.netlib.lapack.Dlacpy.dlacpy("ALL", N, N, A1elems, 0, LDA, Residual, 0, N);

        /* Dealing with L'L or U'U  */
        org.netlib.lapack.Dlacpy.dlacpy(Dplasma.lapack_const(Dplasma.PlasmaUpper), N, N, A2elems, 0, LDA, L1, 0, N);
        org.netlib.lapack.Dlacpy.dlacpy(Dplasma.lapack_const(Dplasma.PlasmaUpper), N, N, A2elems, 0, LDA, L2, 0, N);
        org.netlib.blas.Dtrmm.dtrmm("L", "U", "T", "N", N, N, alpha, L1, 0, N, L2, 0, N);

        /* Compute the Residual || A -L'L|| */
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                Residual[j * N + i] = L2[j * N + i] - Residual[j * N + i];

        Rnorm = org.netlib.lapack.Dlange.dlange(norm, N, N, Residual, 0, N, work, 0);
        Anorm = org.netlib.lapack.Dlange.dlange(norm, N, N, A1elems, 0, LDA, work, 0);

        System.out.print("============\n");
        System.out.print("Checking the Cholesky Factorization \n");
        System.out.print(String.format("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", Rnorm / (Anorm * N * eps)));

        if (Rnorm / (Anorm * N * eps) > 10.0) {
            System.out.print("-- Factorization is suspicious ! \n");
            info_factorization = 1;
        } else {
            System.out.print("-- Factorization is CORRECT ! \n");
            info_factorization = 0;
        }
        return info_factorization;
    }

    /*------------------------------------------------------------------------
     *  Check the accuracy of the solution of the linear system 
     */

    private static int checkSolution(DoubleMatrix2D A1, DoubleMatrix2D B1, DoubleMatrix2D B2, double eps) {
        DoubleProperty.DEFAULT.checkDense(A1);
        DoubleProperty.DEFAULT.checkDense(B1);
        DoubleProperty.DEFAULT.checkDense(B2);
        int N = A1.rows();
        int LDA = N;
        int LDB = N;
        int NRHS = B1.columns();
        int info_solution;
        double Rnorm, Anorm, Xnorm, Bnorm;
        String norm = "I";
        double alpha, beta;
        double[] work = new double[N];
        double[] A1elems;
        double[] B1elems;
        double[] B2elems;

        if (A1 instanceof DenseDoubleMatrix2D) {
            A1elems = (double[]) A1.viewDice().copy().elements();
        } else {
            A1elems = (double[]) A1.copy().elements();
        }

        if (B1 instanceof DenseDoubleMatrix2D) {
            B1elems = (double[]) B1.viewDice().copy().elements();
        } else {
            B1elems = (double[]) B1.copy().elements();
        }

        if (B2 instanceof DenseDoubleMatrix2D) {
            B2elems = (double[]) B2.viewDice().copy().elements();
        } else {
            B2elems = (double[]) B2.copy().elements();
        }

        alpha = 1.0;
        beta = -1.0;

        Xnorm = org.netlib.lapack.Dlange.dlange(norm, N, NRHS, B2elems, 0, LDB, work, 0);
        Anorm = org.netlib.lapack.Dlange.dlange(norm, N, N, A1elems, 0, LDA, work, 0);
        Bnorm = org.netlib.lapack.Dlange.dlange(norm, N, NRHS, B1elems, 0, LDB, work, 0);

        org.netlib.blas.Dgemm.dgemm("N", "N", N, NRHS, N, alpha, A1elems, 0, LDA, B2elems, 0, LDB, beta, B1elems, 0,
                LDB);
        Rnorm = org.netlib.lapack.Dlange.dlange(norm, N, NRHS, B1elems, 0, LDB, work, 0);

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
