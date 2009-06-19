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

class Pdtrsm {

    private Pdtrsm() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Parallel triangular solve
     */
    protected static void plasma_pDTRSM(int side, int uplo, int transA, int diag, int N, int NRHS, double alpha,
            double[] A, int A_offset, int NB, int NBNBSIZE, int NT, int MT, double[] B, int B_offset, int MTB,
            int NTRHS, int cores_num, int my_core_id) {
        int[] progress = Dcommon.plasma_aux.progress;
        int k, m, n;
        int next_k;
        int next_m;
        int next_n;

        k = 0;
        m = my_core_id;
        while (m >= NT) {
            k++;
            m = m - NT + k;
        }
        n = 0;

        while (k < NT && m < NT) {
            next_n = n;
            next_m = m;
            next_k = k;

            next_n++;
            if (next_n >= NTRHS) {
                next_m += cores_num;
                while (next_m >= NT && next_k < NT) {
                    next_k++;
                    next_m = next_m - NT + next_k;
                }
                next_n = 0;
            }

            if (m == k) {
                while (progress[(m) + NT * (n)] != k - 1)
                    Dcommon.delay();
                if (uplo == Dplasma.PlasmaLower) {
                    if (transA == Dplasma.PlasmaNoTrans)
                        DcoreBLAS.core_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaLower, Dplasma.PlasmaNoTrans, diag,
                                k == NT - 1 ? N - k * NB : NB, n == NTRHS - 1 ? NRHS - n * NB : NB, 1.0, A, A_offset
                                        + NBNBSIZE * (k) + NBNBSIZE * MT * (k), NB, B, B_offset + NBNBSIZE * (k)
                                        + NBNBSIZE * MTB * (n), NB);
                    else
                        DcoreBLAS.core_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaLower, Dplasma.PlasmaTrans, diag,
                                k == 0 ? N - (NT - 1) * NB : NB, n == NTRHS - 1 ? NRHS - n * NB : NB, 1.0, A, A_offset
                                        + NBNBSIZE * (NT - 1 - k) + NBNBSIZE * MT * (NT - 1 - k), NB, B, B_offset
                                        + NBNBSIZE * (NT - 1 - k) + NBNBSIZE * MTB * (n), NB);
                } else {
                    if (transA == Dplasma.PlasmaNoTrans)
                        DcoreBLAS.core_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaUpper, Dplasma.PlasmaNoTrans, diag,
                                k == 0 ? N - (NT - 1) * NB : NB, n == NTRHS - 1 ? NRHS - n * NB : NB, 1.0, A, A_offset
                                        + NBNBSIZE * (NT - 1 - k) + NBNBSIZE * MT * (NT - 1 - k), NB, B, B_offset
                                        + NBNBSIZE * (NT - 1 - k) + NBNBSIZE * MTB * (n), NB);
                    else
                        DcoreBLAS.core_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaUpper, Dplasma.PlasmaTrans, diag,
                                k == NT - 1 ? N - k * NB : NB, n == NTRHS - 1 ? NRHS - n * NB : NB, 1.0, A, A_offset
                                        + NBNBSIZE * (k) + NBNBSIZE * MT * (k), NB, B, B_offset + NBNBSIZE * (k)
                                        + NBNBSIZE * MTB * (n), NB);
                }
                progress[(k) + NT * (n)] = k;
            } else {
                while (progress[(k) + NT * (n)] != k)
                    Dcommon.delay();
                while (progress[(m) + NT * (n)] != k - 1)
                    Dcommon.delay();
                if (uplo == Dplasma.PlasmaLower) {
                    if (transA == Dplasma.PlasmaNoTrans)
                        DcoreBLAS.core_DGEMM(Dplasma.PlasmaNoTrans, Dplasma.PlasmaNoTrans, m == NT - 1 ? N - m * NB
                                : NB, n == NTRHS - 1 ? NRHS - n * NB : NB, NB, -1.0, A, A_offset + NBNBSIZE * (m)
                                + NBNBSIZE * MT * (k), NB, B, B_offset + NBNBSIZE * (k) + NBNBSIZE * MTB * (n), NB,
                                1.0, B, B_offset + NBNBSIZE * (m) + NBNBSIZE * MTB * (n), NB);
                    else
                        DcoreBLAS.core_DGEMM(Dplasma.PlasmaTrans, Dplasma.PlasmaNoTrans, NB, n == NTRHS - 1 ? NRHS - n
                                * NB : NB, k == 0 ? N - (NT - 1) * NB : NB, -1.0, A, A_offset + NBNBSIZE * (NT - 1 - k)
                                + NBNBSIZE * MT * (NT - 1 - m), NB, B, B_offset + NBNBSIZE * (NT - 1 - k) + NBNBSIZE
                                * MTB * (n), NB, 1.0, B, B_offset + NBNBSIZE * (NT - 1 - m) + NBNBSIZE * MTB * (n), NB);
                } else {
                    if (transA == Dplasma.PlasmaNoTrans)
                        DcoreBLAS.core_DGEMM(Dplasma.PlasmaNoTrans, Dplasma.PlasmaNoTrans, NB, n == NTRHS - 1 ? NRHS
                                - n * NB : NB, k == 0 ? N - (NT - 1) * NB : NB, -1.0, A, A_offset + NBNBSIZE
                                * (NT - 1 - m) + NBNBSIZE * MT * (NT - 1 - k), NB, B, B_offset + NBNBSIZE
                                * (NT - 1 - k) + NBNBSIZE * MTB * (n), NB, 1.0, B, B_offset + NBNBSIZE * (NT - 1 - m)
                                + NBNBSIZE * MTB * (n), NB);
                    else
                        DcoreBLAS.core_DGEMM(Dplasma.PlasmaTrans, Dplasma.PlasmaNoTrans, m == NT - 1 ? N - m * NB : NB,
                                n == NTRHS - 1 ? NRHS - n * NB : NB, NB, -1.0, A, A_offset + NBNBSIZE * (k) + NBNBSIZE
                                        * MT * (m), NB, B, B_offset + NBNBSIZE * (k) + NBNBSIZE * MTB * (n), NB, 1.0,
                                B, B_offset + NBNBSIZE * (m) + NBNBSIZE * MTB * (n), NB);
                }
                progress[(m) + NT * (n)] = k;
            }
            n = next_n;
            m = next_m;
            k = next_k;
        }
    }

}
