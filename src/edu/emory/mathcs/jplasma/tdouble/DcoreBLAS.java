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

package edu.emory.mathcs.jplasma.tdouble;

import org.netlib.util.intW;

class DcoreBLAS {

    private DcoreBLAS() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DPOTRF(int uplo, int N, double[] A, int A_offset, int LDA, intW INFO) {
        edu.emory.mathcs.jplasma.tdouble.corelapack.Dpotrf
                .dpotrf(Dplasma.lapack_const(uplo), N, A, A_offset, LDA, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DSYRK(int uplo, int trans, int N, int K, double alpha, double[] A, int A_offset,
            int LDA, double beta, double[] C, int C_offset, int LDC) {
        org.netlib.blas.Dsyrk.dsyrk(Dplasma.lapack_const(uplo), Dplasma.lapack_const(trans), N, K, alpha, A, A_offset,
                LDA, beta, C, C_offset, LDC);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DTRSM(int side, int uplo, int transA, int diag, int M, int N, double alpha, double[] A,
            int A_offset, int LDA, double[] B, int B_offset, int LDB) {
        org.netlib.blas.Dtrsm.dtrsm(Dplasma.lapack_const(side), Dplasma.lapack_const(uplo), Dplasma
                .lapack_const(transA), Dplasma.lapack_const(diag), M, N, alpha, A, A_offset, LDA, B, B_offset, LDB);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DGEMM(int transA, int transB, int M, int N, int K, double alpha, double[] A,
            int A_offset, int LDA, double[] B, int B_offset, int LDB, double beta, double[] C, int C_offset, int LDC) {
        org.netlib.blas.Dgemm.dgemm(Dplasma.lapack_const(transA), Dplasma.lapack_const(transB), M, N, K, alpha, A,
                A_offset, LDA, B, B_offset, LDB, beta, C, C_offset, LDC);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DGEQRT(int M, int N, int IB, double[] A, int A_offset, int LDA, double[] T,
            int T_offset, int LDT, double[] TAU, int TAU_offset, double[] WORK, int WORK_offset) {
        intW INFO = new intW(0);
        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dgeqrt.core_dgeqrt(M, N, IB, A, A_offset, LDA, T, T_offset, LDT,
                TAU, TAU_offset, WORK, WORK_offset, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DTSQRT(int M, int N, int IB, double[] A1, int A1_offset, int LDA1, double[] A2,
            int A2_offset, int LDA2, double[] T, int T_offset, int LDT, double[] TAU, int TAU_offset, double[] WORK,
            int WORK_offset) {
        intW INFO = new intW(0);
        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dtsqrt.core_dtsqrt(M, N, IB, A1, A1_offset, LDA1, A2, A2_offset,
                LDA2, T, T_offset, LDT, TAU, TAU_offset, WORK, WORK_offset, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DLARFB(int side, int trans, int direct, int storev, int M, int N, int K, int IB,
            double[] V, int V_offset, int LDV, double[] T, int T_offset, int LDT, double[] C, int C_offset, int LDC,
            double[] WORK, int WORK_offset, int LDWORK) {
        intW INFO = new intW(0);

        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dlarfb.core_dlarfb(Dplasma.lapack_const(side), Dplasma
                .lapack_const(trans), Dplasma.lapack_const(direct), Dplasma.lapack_const(storev), M, N, K, IB, V,
                V_offset, LDV, T, T_offset, LDT, C, C_offset, LDC, WORK, WORK_offset, LDWORK, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DSSRFB(int side, int storev, int M1, int M2, int NN, int IB, int K, double[] A1,
            int A1_offset, int LDA1, double[] A2, int A2_offset, int LDA2, double[] V, int V_offset, int LDV,
            double[] T, int T_offset, int LDT, double[] WORK, int WORK_offset) {
        intW INFO = new intW(0);

        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dssrfb.core_dssrfb(Dplasma.lapack_const(side), Dplasma
                .lapack_const(storev), M1, M2, NN, IB, K, A1, A1_offset, LDA1, A2, A2_offset, LDA2, V, V_offset, LDV,
                T, T_offset, LDT, WORK, WORK_offset, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DGELQT(int M, int N, int IB, double[] A, int A_offset, int LDA, double[] T,
            int T_offset, int LDT, double[] TAU, int TAU_offset, double[] WORK, int WORK_offset) {
        intW INFO = new intW(0);

        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dgelqt.core_dgelqt(M, N, IB, A, A_offset, LDA, T, T_offset, LDT,
                TAU, TAU_offset, WORK, WORK_offset, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DTSLQT(int M, int N, int IB, double[] A1, int A1_offset, int LDA1, double[] A2,
            int A2_offset, int LDA2, double[] T, int T_offset, int LDT, double[] TAU, int TAU_offset, double[] WORK,
            int WORK_offset) {
        intW INFO = new intW(0);

        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dtslqt.core_dtslqt(M, N, IB, A1, A1_offset, LDA1, A2, A2_offset,
                LDA2, T, T_offset, LDT, TAU, TAU_offset, WORK, WORK_offset, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DGETRF(int M, int N, int IB, double[] A, int A_offset, int LDA, double[] L,
            int L_offset, int LDL, int[] IPIV, int IPIV_offset, intW INFO) {
        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dgetrf.core_dgetrf(M, N, IB, A, A_offset, LDA, L, L_offset, LDL,
                IPIV, IPIV_offset, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DTSTRF(int M, int N, int IB, int NB, double[] U, int U_offset, int LDU, double[] A,
            int A_offset, int LDA, double[] L, int L_offset, int LDL, int[] IPIV, int IPIV_offset, intW INFO) {
        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dtstrf.core_dtstrf(M, N, IB, NB, U, U_offset, LDU, A, A_offset,
                LDA, L, L_offset, LDL, IPIV, IPIV_offset, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DGESSM(int M, int N, int K, int IB, int[] IPIV, int IPIV_offset, double[] L,
            int L_offset, int LDL, double[] A, int A_offset, int LDA) {
        intW INFO = new intW(0);

        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dgessm.core_dgessm(M, N, K, IB, IPIV, IPIV_offset, L, L_offset,
                LDL, A, A_offset, LDA, INFO);
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     */
    protected static void core_DSSSSM(int M1, int M2, int NN, int IB, int K, int[] IPIV, int IPIV_offset, double[] L0,
            int L0_offset, int LDL0, double[] L1, int L1_offset, int LDL1, double[] A0, int A0_offset, int LDA0,
            double[] A1, int A1_offset, int LDA1) {
        intW INFO = new intW(0);

        edu.emory.mathcs.jplasma.tdouble.coreblas.Core_dssssm.core_dssssm(M1, M2, NN, IB, K, IPIV, IPIV_offset, L0,
                L0_offset, LDL0, L1, L1_offset, LDL1, A0, A0_offset, LDA0, A1, A1_offset, LDA1, INFO);
    }

}
