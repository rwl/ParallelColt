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

package edu.emory.mathcs.jplasma.example;

import edu.emory.mathcs.jplasma.tdouble.Dplasma;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Example for solving a system of linear equations using QR factorization.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DgelsExample {
    public static void main(String[] args) {
        int M = 15;
        int N = 10;
        int NRHS = 5;
        int INFO;
        int i, j;

        double[] A = new double[M * N];
        double[] B = new double[M * NRHS];

        /* Initialize A */
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                A[M * j + i] = 0.5 - Math.random();

        /* Initialize B */
        for (i = 0; i < M; i++)
            for (j = 0; j < NRHS; j++)
                B[M * j + i] = Math.random();

        /* (Optional step) Set the maximum number of threads. It has to be called before plasma_Init().
         * By default, the maximum number of threads is equal to the number of available processors. */
        ConcurrencyUtils.setNumberOfThreads(4);

        /* Plasma Initialize */
        Dplasma.plasma_Init(M, N, NRHS);

        /* (Optional step) Set the number of threads used in computations. It has to be called after plasma_Init().
         * By default, the number of threads used in computations is equal to the number of available processors. */
        Dplasma.plasma_set_int(Dplasma.PLASMA_CONCURRENCY, 2);

        /* Allocate T */
        double[] T = Dplasma.plasma_Allocate_T(M, N);

        /* Solve the problem */
        INFO = Dplasma.plasma_DGELS(Dplasma.PlasmaNoTrans, M, N, NRHS, A, 0, M, T, 0, B, 0, M);

        /* Plasma Finalize */
        Dplasma.plasma_Finalize();

        if (INFO < 0)
            System.err.println("-- Error in DgelsExample example !");
        else
            System.out.println("-- Run successfull !");

        System.exit(0);
    }
}
