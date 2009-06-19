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

package edu.emory.mathcs.jplasma.benchmark;

import org.netlib.util.intW;

import edu.emory.mathcs.jplasma.tdouble.Dplasma;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Benchmark of plasma_DGELS
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DgelsBenchmark {
    private static void dgelsBenchmarkJPlasma(int M, int N, int NRHS, int NITERS, int NTHREADS) {
        ConcurrencyUtils.setNumberOfThreads(NTHREADS);

        int LDA = M;
        int LDB = Math.max(M, N);
        double[] A = new double[M * N];
        double[] B = new double[LDB * NRHS];
        double[] T;
        double avtime = 0;
        Dplasma.plasma_Init(M, N, 1);

        for (int k = 0; k < NITERS + 2; k++) { //the first two iterations are just for warm-up

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    A[M * j + i] = 0.5 - Math.random();
                }
            }

            for (int i = 0; i < LDB; i++) {
                for (int j = 0; j < NRHS; j++) {
                    B[M * j + i] = Math.random();
                }
            }

            T = Dplasma.plasma_Allocate_T(M, N);

            double temp = System.nanoTime();
            int info = Dplasma.plasma_DGELS(Dplasma.PlasmaNoTrans, M, N, NRHS, A, 0, LDA, T, 0, B, 0, LDB);
            temp = System.nanoTime() - temp;
            if (k >= 2) {
                avtime += temp;
            }
            if (info != 0) {
                System.err.println("failure with error " + info);
            }
        }
        System.out.println("Average execution time of JPlasma DGELS (" + M + "x" + N + ", " + NRHS
                + " right-hand sides, " + NTHREADS + " threads): " + avtime / NITERS / 1000000.0 + " milliseconds");
        Dplasma.plasma_Finalize();

    }

    private static void dgelsBenchmarkJLAPACK(int M, int N, int NRHS, int NITERS) {
        int LDA = M;
        int LDB = Math.max(M, N);
        int MN = Math.min(M, N);
        int LWORK = Math.max(1, MN + Math.max(MN, NRHS));
        double[] A = new double[M * N];
        double[] B = new double[LDB * NRHS];
        double[] WORK;
        double avtime = 0;
        Dplasma.plasma_Init(M, N, 1);

        for (int k = 0; k < NITERS + 2; k++) { //the first two iterations are just for warm-up

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    A[M * j + i] = 0.5 - Math.random();
                }
            }

            for (int i = 0; i < LDB; i++) {
                for (int j = 0; j < NRHS; j++) {
                    B[M * j + i] = Math.random();
                }
            }

            WORK = new double[LWORK];
            intW info = new intW(0);
            double temp = System.nanoTime();
            org.netlib.lapack.Dgels.dgels("N", M, N, NRHS, A, 0, LDA, B, 0, LDB, WORK, 0, LWORK, info);
            temp = System.nanoTime() - temp;
            if (k >= 2) {
                avtime += temp;
            }
            if (info.val != 0) {
                System.err.println("failure with error " + info);
            }
        }
        System.out.println("Average execution time of JLAPACK DGELS (" + M + "x" + N + ", " + NRHS
                + " right-hand sides): " + avtime / NITERS / 1000000.0 + " milliseconds");
        Dplasma.plasma_Finalize();

    }

    public static void main(String[] args) {
        if (args.length != 5) {
            System.out
                    .println("Usage: java edu.emory.mathcs.jplasma.benchmark.DgelsBenchmark M N NRHS, NITERS NTHREADS");
            System.exit(1);
        }
        int M = Integer.parseInt(args[0]);
        int N = Integer.parseInt(args[1]);
        int NRHS = Integer.parseInt(args[2]);
        int NITERS = Integer.parseInt(args[3]);
        int NTHREADS = Integer.parseInt(args[4]);
        dgelsBenchmarkJPlasma(M, N, NRHS, NITERS, NTHREADS);
        //        dgelsBenchmarkJLAPACK(M, N, NRHS, NITERS);        
        System.exit(0);
    }
}
