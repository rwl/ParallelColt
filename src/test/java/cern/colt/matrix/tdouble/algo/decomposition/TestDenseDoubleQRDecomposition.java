package cern.colt.matrix.tdouble.algo.decomposition;

import java.util.Random;

import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

public class TestDenseDoubleQRDecomposition {

    public static void main(String[] args) {

        for (int k = 0; k < 20; k++) {
            int M = 600;
            int N = 400;
            int NRHS = 5;
            Random r = new Random(0);

            DoubleMatrix2D A1 = new DenseColumnDoubleMatrix2D(M, N);
            DoubleMatrix2D A2 = new DenseColumnDoubleMatrix2D(M, N);
            DoubleMatrix2D B1 = new DenseColumnDoubleMatrix2D(M, NRHS);
            DoubleMatrix2D B2 = new DenseColumnDoubleMatrix2D(M, NRHS);

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
            
            testQR(A1, A2, B1, B2);
        }
        System.out.println("All finished");       
        System.exit(0);
    }

    private static void testQR(DoubleMatrix2D A1, DoubleMatrix2D A2, DoubleMatrix2D B1, DoubleMatrix2D B2) {
        int M = A1.rows();
        int N = A1.columns();
        double eps = 1e-10;

        DenseDoubleQRDecomposition qr = new DenseDoubleQRDecomposition(A2);
        DoubleMatrix2D Q = qr.getQ(false);
        DoubleMatrix2D R = qr.getR(false);
        DoubleMatrix2D X = B2.copy();
        qr.solve(X);

        System.out.print("\n");
        System.out.print("------ DenseDoubleQRFactorization tests-------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", M, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the orthogonality, factorization and the solution */
        int info_ortho = check_orthogonality(M, Q, eps);
        int info_factorization = checkFactorization(A1, Q, R, eps);
        int info_solution = checkSolution(A1, B1, X.viewPart(0, 0, A1.columns(), X.columns()).copy(), eps);

        if ((info_solution == 0) & (info_factorization == 0) & (info_ortho == 0)) {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGEQRF + DORMQR + DTRSM .... PASSED !\n");
            System.out.print("************************************************\n");
        } else {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGEQRF + DORMQR + DTRSM .... FAILED !\n");
            System.out.print("************************************************\n");
        }
    }

    /*-------------------------------------------------------------------
     * Check the orthogonality of Q
     */

    private static int check_orthogonality(int M, DoubleMatrix2D Q, double eps) {
        double alpha, beta;
        double normQ;
        int info_ortho;

        alpha = 1.0;
        beta = -1.0;

        /* Build the idendity matrix */
        DoubleMatrix2D Id = DoubleFactory2D.dense.identity(Q.columns());

        /* Perform Id - Q'*Q */
        Id = Q.zMult(Q, Id, alpha, beta, true, false);
        normQ = DenseDoubleAlgebra.DEFAULT.normInfinity(Id);

        System.out.print("============\n");
        System.out.print("Checking the orthogonality of Q \n");
        System.out.print(String.format("||Id-Q'*Q||_oo / (N*eps) = %e\n", normQ / (M * eps)));

        if (normQ / (M * eps) > 10.0) {
            System.out.print("-- Orthogonality is suspicious ! \n");
            info_ortho = 1;
        } else {
            System.out.print("-- Orthogonality is CORRECT ! \n");
            info_ortho = 0;
        }
        return info_ortho;
    }

    private static int checkFactorization(DoubleMatrix2D A1, DoubleMatrix2D Q, DoubleMatrix2D R, double eps) {
        DoubleProperty.DEFAULT.checkDense(A1);
        DoubleProperty.DEFAULT.checkDense(Q);
        DoubleProperty.DEFAULT.checkDense(R);
        int M = A1.rows();
        int N = A1.columns();
        int info_factorization;
        double Anorm, Rnorm;
        double alpha;
        DoubleMatrix2D Residual;
        DoubleMatrix2D A2 = Q.copy();

        alpha = 1.0;

        Residual = A1.copy();

        A2 = A2.zMult(R, null, alpha, 0, false, false); // A2 = QR

        /* Compute the Residual ||QR-A|| */
        Residual.assign(A2, DoubleFunctions.plusMultFirst(-1));

        Rnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(Residual);
        Anorm = DenseDoubleAlgebra.DEFAULT.normInfinity(A1);

        System.out.print("============\n");
        System.out.print("Checking the QR Factorization \n");
        System.out.print(String.format("-- ||QR-A||_oo/(||A||_oo.N.eps) = %e \n", Rnorm / (Anorm * N * eps)));

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
        int M = A1.rows();
        int N = A1.columns();
        int info_solution;
        double Rnorm, Anorm, Xnorm, Bnorm;
        double alpha, beta;

        alpha = 1.0;
        beta = -1.0;

        Xnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(B2);
        Anorm = DenseDoubleAlgebra.DEFAULT.normInfinity(A1);
        Bnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(B1);

        A1.zMult(B2, B1, alpha, beta, false, false);

        DoubleMatrix2D Residual = A1.zMult(B1.viewPart(0, 0, A1.rows(), B1.columns()).copy(), null, alpha, beta, true,
                false);

        Rnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(Residual);

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
