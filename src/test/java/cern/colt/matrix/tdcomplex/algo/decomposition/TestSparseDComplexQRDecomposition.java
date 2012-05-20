package cern.colt.matrix.tdcomplex.algo.decomposition;

import java.util.Random;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.algo.DenseDComplexAlgebra;
import cern.colt.matrix.tdcomplex.algo.DComplexProperty;
import cern.colt.matrix.tdcomplex.algo.SparseDComplexAlgebra;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.SparseRCDComplexMatrix2D;

import static edu.emory.mathcs.utils.Utils.CONE;
import static edu.emory.mathcs.utils.Utils.CNEG_ONE;

public class TestSparseDComplexQRDecomposition {
    public static void main(String[] args) {
        int M = 280;
        int N = 180;
        int MN = Math.max(M, N);
        Random r = new Random(0);

        DComplexMatrix2D A1 = new SparseRCDComplexMatrix2D(M, N);
        DComplexMatrix2D A2 = new SparseRCDComplexMatrix2D(M, N);
        DComplexMatrix1D B1 = new DenseDComplexMatrix1D(MN);
        DComplexMatrix1D B2 = new DenseDComplexMatrix1D(MN);

        /* Initialize A1 and A2*/
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                A1.setQuick(i, j, 0.5 - r.nextDouble(), 0.5 - r.nextDouble());
                A2.setQuick(i, j, A1.getQuick(i, j));
            }
        }

        /* Initialize B1 and B2 */
        for (int i = 0; i < MN; i++) {
            B1.setQuick(i, 0.5 - r.nextDouble(), 0.5 - r.nextDouble());
            B2.setQuick(i, B1.getQuick(i));
        }

        testQR(A1, A2, B1, B2);
    }

    private static void testQR(DComplexMatrix2D A1, DComplexMatrix2D A2, DComplexMatrix1D B1, DComplexMatrix1D B2) {
        int M = A1.rows();
        int N = A1.columns();
        double eps = 1e-10;

        SparseDComplexQRDecomposition qr = new SparseDComplexQRDecomposition(A2, 0);
        DComplexMatrix1D X = B2.copy();
        qr.solve(X);

        System.out.print("\n");
        System.out.print("------ SparseDComplexQRFactorization tests-------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", M, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the factorization and the solution */
        int info_solution = checkSolution(A1, B1, X.viewPart(0, A1.columns()).copy(), eps);

        if ((info_solution == 0)) {
            System.out.print("***************************************************\n");
            System.out.print(" ---- SparseDComplexQRFactorization tests... PASSED !\n");
            System.out.print("***************************************************\n");
        } else {
            System.out.print("***************************************************\n");
            System.out.print(" ---- SparseDComplexQRFactorization tests... FAILED !\n");
            System.out.print("***************************************************\n");
        }
    }

    private static int checkSolution(DComplexMatrix2D A1, DComplexMatrix1D B1, DComplexMatrix1D B2, double eps) {
        DComplexProperty.DEFAULT.checkSparse(A1);
        DComplexProperty.DEFAULT.checkDense(B1);
        DComplexProperty.DEFAULT.checkDense(B2);
        int M = A1.rows();
        int N = A1.columns();
        int info_solution;
        double Rnorm, Anorm, Xnorm, Bnorm;
        double[] alpha, beta;

        alpha = CONE;
        beta = CNEG_ONE;

        Xnorm = DenseDComplexAlgebra.DEFAULT.normInfinity(B2);
        Anorm = SparseDComplexAlgebra.DEFAULT.normInfinity(A1);
        Bnorm = DenseDComplexAlgebra.DEFAULT.normInfinity(B1);

        //B1 = A1*B2 - B1;
        A1.zMult(B2, B1, alpha, beta, false);

        DComplexMatrix1D Residual = A1.zMult(B1.viewPart(0, A1.rows()).copy(), null, alpha, beta, true);

        Rnorm = DenseDComplexAlgebra.DEFAULT.normInfinity(Residual);

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
