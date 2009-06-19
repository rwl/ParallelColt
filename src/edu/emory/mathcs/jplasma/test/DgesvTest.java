/* ***** BEGIN LICENSE BLOCK *****
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright 2008
 *
 * Redistribution  and  use  in  source and binary forms, with or without
 * modification,  are  permitted  provided  that the following conditions
 * are met:
 *
 * * Redistributions  of  source  code  must  retain  the above copyright
 *   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * * Redistributions  in  binary  form must reproduce the above copyright
 *   notice,  this list of conditions and the following disclaimer in the
 *   documentation  and/or other materials provided with the distribution.
 * * Neither  the  name of the University of Tennessee, Knoxville nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jplasma.test;

import edu.emory.mathcs.jplasma.tdouble.Dplasma;

/**
 * Test of plasma_DGESV
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DgesvTest {

    public static void main(String[] args) {

        /* Check for valid arguments*/
        if (args.length != 4) {
            System.out
                    .print(" Proper Usage is : java edu.emory.mathcs.jplasma.test.DgesvTest N LDA NRHS LDB with \n - N : the size of the matrix \n - LDA : leading dimension of the matrix A \n - NRHS : number of RHS \n - LDB : leading dimension of the matrix B \n");
            System.exit(1);
        }

        int N = Integer.parseInt(args[0]);
        int LDA = Integer.parseInt(args[1]);
        int NRHS = Integer.parseInt(args[2]);
        int LDB = Integer.parseInt(args[3]);
        double eps;
        int info_solution;
        int i, j;

        double[] A1 = new double[LDA * N];
        double[] A2 = new double[LDA * N];
        double[] B1 = new double[LDB * NRHS];
        double[] B2 = new double[LDB * NRHS];

        /*----------------------------------------------------------
        *  TESTING DGESV
        */

        /*Plasma Initialize*/
        Dplasma.plasma_Init(N, N, NRHS);

        /* Initialize A1 and A2 Matrix */
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                A1[LDA * j + i] = A2[LDA * j + i] = 0.5 - Math.random();
        ;

        /* Initialize B1 and B2 */
        for (i = 0; i < N; i++)
            for (j = 0; j < NRHS; j++)
                B1[LDB * j + i] = B2[LDB * j + i] = 0.5 - Math.random();

        /* PLASAM DGESV */
        double[] L = Dplasma.plasma_Allocate_L(N, N);
        int[] IPIV = Dplasma.plasma_Allocate_IPIV(N, N);
        Dplasma.plasma_DGESV(N, NRHS, A2, 0, LDA, L, 0, IPIV, 0, B2, 0, LDB);
        Dplasma.plasma_Finalize();

        eps = 1e-10;
        System.out.print("\n");
        System.out.print("------ TESTS FOR PLASMA DGESV ROUTINE -------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", N, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the factorization and the solution */
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if ((info_solution == 0)) {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGESV ... PASSED !\n");
            System.out.print("************************************************\n");
        } else {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGESV ... FAILED !\n");
            System.out.print("************************************************\n");
        }

        /*-------------------------------------------------------------
        *  TESTING DGETRF + DGETRS
        */

        /* Initialize A1 and A2  */
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                A1[LDA * i + j] = A2[LDA * i + j] = 0.5 - Math.random();

        /* Initialize B1 and B2 */
        for (i = 0; i < N; i++)
            for (j = 0; j < NRHS; j++)
                B1[LDB * j + i] = B2[LDB * j + i] = 0.5 - Math.random();

        /* Plasma routines */
        Dplasma.plasma_Init(N, N, NRHS);
        L = Dplasma.plasma_Allocate_L(N, N);
        IPIV = Dplasma.plasma_Allocate_IPIV(N, N);
        Dplasma.plasma_DGETRF(N, N, A2, 0, LDA, L, 0, IPIV, 0);
        Dplasma.plasma_Finalize();
        Dplasma.plasma_Init(N, N, NRHS);
        Dplasma.plasma_DGETRS(N, NRHS, N, A2, 0, LDA, L, 0, IPIV, 0, B2, 0, LDB);
        Dplasma.plasma_Finalize();

        System.out.print("\n");
        System.out.print("------ TESTS FOR PLASMA DGETRF + DGETRS ROUTINE -------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", N, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the solution */
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if ((info_solution == 0)) {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGETRF + DGETRS ... PASSED !\n");
            System.out.print("************************************************\n");
        } else {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGETRF + DGETRS ... FAILED !\n");
            System.out.print("************************************************\n");
        }

        /*-------------------------------------------------------------
        *  TESTING DGETRF + DTRSMPL + DTRSM
        */

        /* Initialize A1 and A2  */
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                A1[LDA * i + j] = A2[LDA * i + j] = 0.5 - Math.random();

        /* Initialize B1 and B2 */
        for (i = 0; i < N; i++)
            for (j = 0; j < NRHS; j++)
                B1[LDB * j + i] = B2[LDB * j + i] = 0.5 - Math.random();

        /* PLASMA routines */
        Dplasma.plasma_Init(N, N, NRHS);
        L = Dplasma.plasma_Allocate_L(N, N);
        IPIV = Dplasma.plasma_Allocate_IPIV(N, N);
        Dplasma.plasma_DGETRF(N, N, A2, 0, LDA, L, 0, IPIV, 0);
        Dplasma.plasma_Finalize();
        Dplasma.plasma_Init(N, N, NRHS);
        Dplasma.plasma_DTRSMPL(N, NRHS, N, A2, 0, LDA, L, 0, IPIV, 0, B2, 0, LDB);
        Dplasma.plasma_Finalize();
        Dplasma.plasma_Init(N, N, NRHS);
        Dplasma.plasma_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaUpper, Dplasma.PlasmaNoTrans, Dplasma.PlasmaNonUnit, N,
                NRHS, A2, 0, LDA, B2, 0, LDB);
        Dplasma.plasma_Finalize();

        System.out.print("\n");
        System.out.print("------ TESTS FOR PLASMA DGETRF + DTRSMPL + DTRSM  ROUTINE -------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", N, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the solution */
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if ((info_solution == 0)) {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGETRF + DTRSMPL + DTRSM ... PASSED !\n");
            System.out.print("************************************************\n");
        } else {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGETRF + DTRSMPL + DTRSM ... FAILED !\n");
            System.out.print("************************************************\n");
        }
        System.exit(0);
    }

    /*------------------------------------------------------------------------
     *  Check the accuracy of the solution of the linear system 
     */

    private static int check_solution(int N, int NRHS, double[] A1, int LDA, double[] B1, double[] B2, int LDB,
            double eps) {
        int info_solution;
        double Rnorm, Anorm, Xnorm, Bnorm;
        String norm = "I";
        double alpha, beta;
        double[] work = new double[N];

        alpha = 1.0;
        beta = -1.0;

        Xnorm = org.netlib.lapack.Dlange.dlange(norm, N, NRHS, B2, 0, LDB, work, 0);
        Anorm = org.netlib.lapack.Dlange.dlange(norm, N, N, A1, 0, LDA, work, 0);
        Bnorm = org.netlib.lapack.Dlange.dlange(norm, N, NRHS, B1, 0, LDB, work, 0);

        org.netlib.blas.Dgemm.dgemm("N", "N", N, NRHS, N, alpha, A1, 0, LDA, B2, 0, LDB, beta, B1, 0, LDB);
        Rnorm = org.netlib.lapack.Dlange.dlange(norm, N, NRHS, B1, 0, LDB, work, 0);

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
