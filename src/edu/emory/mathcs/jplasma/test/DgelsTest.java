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

import java.util.Random;

import edu.emory.mathcs.jplasma.tdouble.Dplasma;

/**
 * Test of plasma_DGELS
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DgelsTest {

    public static void main(String[] args) {

        /* Check for number of arguments*/
        if (args.length != 5) {
            System.out
                    .println(" Proper Usage is : java edu.emory.mathcs.jplasma.test.DgelsTest M N LDA NRHS LDB with \n - M : number of rows of the matrix A \n - N : number of columns of the matrix A \n - LDA : leading dimension of the matrix A \n - NRHS : number of RHS \n - LDB : leading dimension of the matrix B");
            System.exit(1);
        }

        int M = Integer.parseInt(args[0]);
        int N = Integer.parseInt(args[1]);
        int LDA = Integer.parseInt(args[2]);
        int NRHS = Integer.parseInt(args[3]);
        int LDB = Integer.parseInt(args[4]);
        double eps;
        int info_ortho, info_solution, info_factorization;
        int i, j;

        double[] A1 = new double[LDA * N];
        double[] A2 = new double[LDA * N];
        double[] Q = new double[M * M];
        double[] B1 = new double[LDB * NRHS];
        double[] B2 = new double[LDB * NRHS];
        Random r = new Random(0);

        /* Plasma Initialization */
        Dplasma.plasma_Init(M, N, NRHS);

        /*----------------------------------------------------------
        *  TESTING DGELS
        */

        /* Initialize A1 and A2 */
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                A1[LDA * j + i] = A2[LDA * j + i] = 0.5 - r.nextDouble();

        for (i = 0; i < M; i++)
            Q[M * i + i] = 1.0;

        /* Initialize B1 and B2 */
        for (i = 0; i < M; i++)
            for (j = 0; j < NRHS; j++)
                B1[LDB * j + i] = B2[LDB * j + i] = 0.5 - r.nextDouble();

        /* PLASMA DGELS */
        double[] T = Dplasma.plasma_Allocate_T(M, N);
        Dplasma.plasma_DGELS(Dplasma.PlasmaNoTrans, M, N, NRHS, A2, 0, LDA, T, 0, B2, 0, LDB);
        Dplasma.plasma_Finalize();
        Dplasma.plasma_Init(M, N, NRHS);
        Dplasma.plasma_DORMQR(Dplasma.PlasmaLeft, Dplasma.PlasmaNoTrans, M, M, N, A2, 0, LDA, T, 0, Q, 0, M);

        eps = 1e-10;
        System.out.print("\n");
        System.out.print("------ TESTS FOR PLASMA DGELS ROUTINE -------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", M, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the orthogonality, factorization and the solution */
        info_ortho = check_orthogonality(M, Q, eps);
        info_factorization = check_factorization(M, N, A1, A2, LDA, Q, eps);
        info_solution = check_solution(M, N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if ((info_solution == 0) & (info_factorization == 0) & (info_ortho == 0)) {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGELS .... PASSED !\n");
            System.out.print("************************************************\n");
        } else {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGELS .... FAILED !\n");
            System.out.print("************************************************\n");
        }

        /*----------------------------------------------------------
        *  TESTING DGEQRF + DORMQR + DTRSM
        */

        r = new Random(0);
        /* Initialize A1 and A2 */
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                A1[LDA * j + i] = A2[LDA * j + i] = 0.5 - r.nextDouble();

        Q = new double[M * M];
        for (i = 0; i < M; i++)
            Q[M * i + i] = 1.0;

        /* Initialize B1 and B2 */
        for (i = 0; i < M; i++)
            for (j = 0; j < NRHS; j++)
                B1[LDB * j + i] = B2[LDB * j + i] = 0.5 - r.nextDouble();

        /* PLASMA DGEQRF+ DORMQR + DTRSM */
        T = Dplasma.plasma_Allocate_T(M, N);
        Dplasma.plasma_DGEQRF(M, N, A2, 0, LDA, T, 0);
        Dplasma.plasma_Finalize();
        Dplasma.plasma_Init(M, N, NRHS);
        Dplasma.plasma_DORMQR(Dplasma.PlasmaLeft, Dplasma.PlasmaNoTrans, M, M, N, A2, 0, LDA, T, 0, Q, 0, M); /* To extract Q */
        Dplasma.plasma_Finalize();
        Dplasma.plasma_Init(M, N, NRHS);
        Dplasma.plasma_DORMQR(Dplasma.PlasmaLeft, Dplasma.PlasmaNoTrans, M, NRHS, N, A2, 0, LDA, T, 0, B2, 0, LDB);
        Dplasma.plasma_Finalize();
        Dplasma.plasma_Init(M, N, NRHS);
        Dplasma.plasma_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaUpper, Dplasma.PlasmaNoTrans, Dplasma.PlasmaNonUnit, N,
                NRHS, A2, 0, LDA, B2, 0, LDB);

        System.out.print("\n");
        System.out.print("------ TESTS FOR PLASMA DGEQRF + DORMQR + DTRSM  ROUTINE -------  \n");
        System.out.print(String.format("            Size of the Matrix %d by %d\n", M, N));
        System.out.print("\n");
        System.out.print(" The matrix A is randomly generated for each test.\n");
        System.out.print("============\n");
        System.out.print(String.format(" The relative machine precision (eps) is to be %e \n", eps));
        System.out.print(" Computational tests pass if scaled residuals are less than 10.\n");

        /* Check the orthogonality, factorization and the solution */
        info_ortho = check_orthogonality(M, Q, eps);
        info_factorization = check_factorization(M, N, A1, A2, LDA, Q, eps);
        info_solution = check_solution(M, N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if ((info_solution == 0) & (info_factorization == 0) & (info_ortho == 0)) {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGEQRF + DORMQR + DTRSM .... PASSED !\n");
            System.out.print("************************************************\n");
        } else {
            System.out.print("************************************************\n");
            System.out.print(" ---- TESTING DGEQRF + DORMQR + DTRSM .... FAILED !\n");
            System.out.print("************************************************\n");
        }
        Dplasma.plasma_Finalize();
        System.exit(0);
    }

    /*-------------------------------------------------------------------
     * Check the orthogonality of Q
     */

    private static int check_orthogonality(int M, double[] Q, double eps) {
        double alpha, beta;
        double normQ;
        String norm = "I";
        int info_ortho;
        int i;

        double[] work = new double[M];

        alpha = 1.0;
        beta = -1.0;

        /* Build the idendity matrix */
        double[] Id = new double[M * M];
        for (i = 0; i < M; i++)
            Id[i * M + i] = 1.0;

        /* Perform Id - Q'Q */
        org.netlib.blas.Dsyrk.dsyrk("U", "N", M, M, alpha, Q, 0, M, beta, Id, 0, M);
        normQ = org.netlib.lapack.Dlansy.dlansy(norm, Dplasma.lapack_const(Dplasma.PlasmaUpper), M, Id, 0, M, work, 0);

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

    /*------------------------------------------------------------
     *  Check the factorization QR
     */

    private static int check_factorization(int M, int N, double[] A1, double[] A2, int LDA, double[] Q, double eps) {
        double Anorm, Rnorm;
        double alpha, beta;
        String norm = "I";
        int info_factorization;
        int i, j;

        double[] Ql = new double[M * N];
        double[] Residual = new double[M * N];
        double[] work = new double[M];

        alpha = 1.0;
        beta = 0.0;

        /* Extract the R */
        double[] R = new double[M * N];
        org.netlib.lapack.Dlacpy.dlacpy("U", M, N, A2, 0, LDA, R, 0, M);

        /* Perform Ql=Q*R-Ql */
        Ql = new double[M * N];
        org.netlib.blas.Dgemm.dgemm("T", "N", M, N, M, alpha, Q, 0, M, R, 0, M, beta, Ql, 0, M);

        /* Compute the Residual */
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                Residual[j * M + i] = A1[j * LDA + i] - Ql[j * M + i];

        Rnorm = org.netlib.lapack.Dlange.dlange(norm, M, N, Residual, 0, M, work, 0);
        Anorm = org.netlib.lapack.Dlange.dlange(norm, M, N, A2, 0, LDA, work, 0);

        System.out.print("============\n");
        System.out.print("Checking the QR Factorization \n");
        System.out.print(String.format("-- ||A-QR||_oo/(||A||_oo.N.eps) = %e \n", Rnorm / (Anorm * N * eps)));

        if (Rnorm / (Anorm * N * eps) > 10.0) {
            System.out.print("-- Factorization is suspicious ! \n");
            info_factorization = 1;
        } else {
            System.out.print("-- Factorization is CORRECT ! \n");
            info_factorization = 0;
        }
        return info_factorization;
    }

    /*--------------------------------------------------------------
     * Check the solution 
     */

    private static int check_solution(int M, int N, int NRHS, double[] A1, int LDA, double[] B1, double[] B2, int LDB,
            double eps) {
        int info_solution;
        double Rnorm, Anorm, Xnorm, Bnorm;
        String norm = "I";
        double alpha, beta;

        double[] work = new double[M];

        alpha = 1.0;
        beta = -1.0;

        Anorm = org.netlib.lapack.Dlange.dlange(norm, M, N, A1, 0, LDA, work, 0);
        Xnorm = org.netlib.lapack.Dlange.dlange(norm, M, NRHS, B2, 0, LDB, work, 0);
        Bnorm = org.netlib.lapack.Dlange.dlange(norm, M, NRHS, B1, 0, LDB, work, 0);

        org.netlib.blas.Dgemm.dgemm("N", "N", M, NRHS, N, alpha, A1, 0, LDA, B2, 0, LDB, beta, B1, 0, LDB);

        double[] Residual = new double[M * NRHS];

        org.netlib.blas.Dgemm.dgemm("T", "N", N, NRHS, M, alpha, A1, 0, LDA, B1, 0, LDB, beta, Residual, 0, M);
        Rnorm = org.netlib.lapack.Dlange.dlange(norm, M, NRHS, Residual, 0, M, work, 0);

        System.out.print("============\n");
        System.out.print("Checking the Residual of the solution \n");
        System.out.print(String.format("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||)_oo.N.eps) = %e \n", Rnorm
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
