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

class Pdpotrf {

    private Pdpotrf() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Parallel Cholesky factorization
     */
    protected static void plasma_pDPOTRF(int uplo, int N, double[] A, int A_offset, int NB, int NBNBSIZE, int NT,
            intW INFO, int cores_num, int my_core_id) {
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
            if (next_n > next_k) {
                next_m += cores_num;
                while (next_m >= NT && next_k < NT) {
                    next_k++;
                    next_m = next_m - NT + next_k;
                }
                next_n = 0;
            }

            if (m == k) {
                if (n == k) {
                    if (uplo == Dplasma.PlasmaLower)
                        DcoreBLAS.core_DPOTRF(Dplasma.PlasmaLower, k == NT - 1 ? N - k * NB : NB, A, A_offset
                                + NBNBSIZE * (k) + NBNBSIZE * NT * (k), NB, INFO);
                    else
                        DcoreBLAS.core_DPOTRF(Dplasma.PlasmaUpper, k == NT - 1 ? N - k * NB : NB, A, A_offset
                                + NBNBSIZE * (k) + NBNBSIZE * NT * (k), NB, INFO);
                    if (INFO.val != 0)
                        INFO.val += NB * k;
                    progress[(k) + NT * (k)] = 1;
                } else {
                    while (progress[(k) + NT * (n)] != 1 && INFO.val == 0)
                        Dcommon.delay();
                    if (uplo == Dplasma.PlasmaLower)
                        DcoreBLAS.core_DSYRK(Dplasma.PlasmaLower, Dplasma.PlasmaNoTrans, k == NT - 1 ? N - k * NB : NB,
                                NB, -1.0, A, A_offset + NBNBSIZE * (k) + NBNBSIZE * NT * (n), NB, 1.0, A, A_offset
                                        + NBNBSIZE * (k) + NBNBSIZE * NT * (k), NB);
                    else
                        DcoreBLAS.core_DSYRK(Dplasma.PlasmaUpper, Dplasma.PlasmaTrans, k == NT - 1 ? N - k * NB : NB,
                                NB, -1.0, A, A_offset + NBNBSIZE * (n) + NBNBSIZE * NT * (k), NB, 1.0, A, A_offset
                                        + NBNBSIZE * (k) + NBNBSIZE * NT * (k), NB);
                }
            } else {
                if (n == k) {
                    while (progress[(k) + NT * (k)] != 1 && INFO.val == 0)
                        Dcommon.delay();
                    if (uplo == Dplasma.PlasmaLower)
                        DcoreBLAS.core_DTRSM(Dplasma.PlasmaRight, Dplasma.PlasmaLower, Dplasma.PlasmaTrans,
                                Dplasma.PlasmaNonUnit, m == NT - 1 ? N - m * NB : NB, NB, 1.0, A, A_offset + NBNBSIZE
                                        * (k) + NBNBSIZE * NT * (k), NB, A, A_offset + NBNBSIZE * (m) + NBNBSIZE * NT
                                        * (k), NB);
                    else
                        DcoreBLAS.core_DTRSM(Dplasma.PlasmaLeft, Dplasma.PlasmaUpper, Dplasma.PlasmaTrans,
                                Dplasma.PlasmaNonUnit, NB, m == NT - 1 ? N - m * NB : NB, 1.0, A, A_offset + NBNBSIZE
                                        * (k) + NBNBSIZE * NT * (k), NB, A, A_offset + NBNBSIZE * (k) + NBNBSIZE * NT
                                        * (m), NB);
                    progress[(m) + NT * (k)] = 1;
                } else {
                    while (progress[(k) + NT * (n)] != 1 && INFO.val == 0)
                        Dcommon.delay();
                    while (progress[(m) + NT * (n)] != 1 && INFO.val == 0)
                        Dcommon.delay();
                    if (uplo == Dplasma.PlasmaLower)
                        DcoreBLAS.core_DGEMM(Dplasma.PlasmaNoTrans, Dplasma.PlasmaTrans, m == NT - 1 ? N - m * NB : NB,
                                NB, NB, -1.0, A, A_offset + NBNBSIZE * (m) + NBNBSIZE * NT * (n), NB, A, A_offset
                                        + NBNBSIZE * (k) + NBNBSIZE * NT * (n), NB, 1.0, A, A_offset + NBNBSIZE * (m)
                                        + NBNBSIZE * NT * (k), NB);
                    else
                        DcoreBLAS.core_DGEMM(Dplasma.PlasmaTrans, Dplasma.PlasmaNoTrans, NB, m == NT - 1 ? N - m * NB
                                : NB, NB, -1.0, A, A_offset + NBNBSIZE * (n) + NBNBSIZE * NT * (k), NB, A, A_offset
                                + NBNBSIZE * (n) + NBNBSIZE * NT * (m), NB, 1.0, A, A_offset + NBNBSIZE * (k)
                                + NBNBSIZE * NT * (m), NB);
                }
            }
            if (INFO.val != 0)
                return;

            n = next_n;
            m = next_m;
            k = next_k;
        }
    }
}
