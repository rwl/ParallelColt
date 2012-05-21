package cern.colt.matrix.tdcomplex.algo.decomposition;

import java.util.Random;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.algo.DComplexProperty;
import cern.colt.matrix.tdcomplex.algo.SparseDComplexAlgebra;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.SparseCCDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.SparseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplexFunctions;

import static edu.emory.mathcs.utils.Utils.CZERO;
import static edu.emory.mathcs.utils.Utils.CONE;
import static edu.emory.mathcs.utils.Utils.CNEG_ONE;

public class TestSparseDComplexCholeskyDecomposition {
    public static void main(String[] args) {
        int N = 200;
        Random r = new Random(0);

        DComplexMatrix2D A1 = new SparseCCDComplexMatrix2D(N, N);
        DComplexMatrix2D A2 = new SparseCCDComplexMatrix2D(N, N);
        DComplexMatrix1D B1 = new DenseDComplexMatrix1D(N);
        DComplexMatrix1D B2 = new DenseDComplexMatrix1D(N);

        /* Initialize A1 and A2 for Symmetric Positive Matrix */
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A1.setQuick(i, j, 0.5 - r.nextDouble(), 0.0);  // TODO: test complex part
                A2.setQuick(i, j, A1.getQuick(i, j));
            }
        }

        for (int i = 0; i < N; i++) {
            A1.setQuick(i, i, A1.getQuick(i, i)[0] + N, 0.0);  // TODO: test complex part
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
            B1.setQuick(i, 0.5 - r.nextDouble(), 0.0);  // TODO: test complex part
            B2.setQuick(i, B1.getQuick(i));
        }

        testCholesky(A1, A2, B1, B2);

        System.exit(0);

    }

    private static void testCholesky(DComplexMatrix2D A1, DComplexMatrix2D A2, DComplexMatrix1D B1, DComplexMatrix1D B2) {
        int N = A1.rows();
        double eps = 1e-10;

        SparseDComplexCholeskyDecomposition cf = new SparseDComplexCholeskyDecomposition(A2, 0);
        DComplexMatrix2D L = cf.getL();
        DComplexMatrix1D X = B2.copy();
        cf.solve(X);

        System.out.print("\n");
        System.out.print("------ SparseDComplexCholeskyFactorization tests-------  \n");
        System.out.printf("            Size of the Matrix %d by %d\n", N, N);
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.printf(" The relative machine precision (eps) is to be %e \n", eps);
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the factorization and the solution */
        int info_factorization = checkFactorization(A1, L, eps);
        int info_solution = checkSolution(A1, B1, X, eps);

        if ((info_solution == 0) & (info_factorization == 0)) {
            System.out.print("***************************************************\n");
            System.out.print(" ---- SparseDComplexCholeskyFactorization tests... PASSED !\n");
            System.out.print("***************************************************\n");
        } else {
            System.err.print("***************************************************\n");
            System.err.print(" ---- SparseDComplexCholeskyFactorization tests... FAILED !\n");
            System.err.print("***************************************************\n");
        }
    }

    private static int checkFactorization(DComplexMatrix2D A1, DComplexMatrix2D L, double eps) {
        DComplexProperty.DEFAULT.checkSparse(A1);
        DComplexProperty.DEFAULT.checkSparse(L);
        int N = A1.rows();
        int info_factorization;
        double Anorm, Rnorm;
        double[] alpha;
        DComplexMatrix2D Residual;
        DComplexMatrix2D L1 = L.copy();
        DComplexMatrix2D L2 = L.copy();

        alpha = CONE;

        Residual = A1.copy();

        L2 = L1.zMult(L2, null, alpha, CZERO, false, true); // L2 = LL'

        /* Compute the Residual || A -LL'|| */
        Residual.assign(L2, DComplexFunctions.plusMultFirst(CNEG_ONE));

        Rnorm = SparseDComplexAlgebra.DEFAULT.normInfinity(Residual);
        Anorm = SparseDComplexAlgebra.DEFAULT.normInfinity(A1);

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

    private static int checkSolution(DComplexMatrix2D A1, DComplexMatrix1D B1, DComplexMatrix1D B2, double eps) {
        DComplexProperty.DEFAULT.checkSparse(A1);
        DComplexProperty.DEFAULT.checkDense(B1);
        DComplexProperty.DEFAULT.checkDense(B2);
        int N = A1.rows();
        int info_solution;
        double Rnorm, Anorm, Xnorm, Bnorm;
        double alpha, beta;

        alpha = 1; 
        beta = -1;

        // TODO: test complex part
        DoubleMatrix2D AA1 = new SparseRCDoubleMatrix2D(A1.getRealPart().toArray());
        DoubleMatrix1D BB1 = B1.getRealPart();
        DoubleMatrix1D BB2 = B2.getRealPart();

        Xnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(BB2);
        Anorm = SparseDoubleAlgebra.DEFAULT.normInfinity(AA1);
        Bnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(BB1);

        //B1 = A1*B2 - B1;
        AA1.zMult(BB2, BB1, alpha, beta, false);
        Rnorm = DenseDoubleAlgebra.DEFAULT.normInfinity(BB1);

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
