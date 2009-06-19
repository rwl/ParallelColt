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

class Pdormlq {

    private Pdormlq() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Parallel application of Q using tile V - LQ factorization
     */
    protected void plasma_pDORMLQ(int M, int NRHS, int N, double[] A, int A_offset, int NB, int NBNBSIZE, int IBNBSIZE,
            int IB, int MT, int NTRHS, int NT, double[] T, int T_offset, double[] B, int B_offset, intW INFO,
            int cores_num, int my_core_id) {
        double[] WORK = Dcommon.plasma_aux.WORK[my_core_id];
        int[] progress = Dcommon.plasma_aux.progress;
        int k, m, n;
        int next_k;
        int next_m;
        int next_n;

        k = 0;
        n = my_core_id;
        while (n >= NTRHS) {
            k++;
            n = n - NTRHS;
        }
        m = k;

        while (k < Math.min(MT, NT) && n < NTRHS) {
            next_n = n;
            next_m = m;
            next_k = k;

            next_m++;
            if (next_m == MT) {
                next_n += cores_num;
                while (next_n >= NTRHS && next_k < Math.min(MT, NT)) {
                    next_k++;
                    next_n = next_n - NTRHS;
                }
                next_m = next_k;
            }

            if (m == k) {
                while (progress[(k) + MT * (n)] != k - 1)
                    Dcommon.delay();
                DcoreBLAS.core_DLARFB(Dplasma.PlasmaLeft, Dplasma.PlasmaNoTrans, Dplasma.PlasmaForward,
                        Dplasma.PlasmaRowwise,
                        NB, //m == MT-1 ? M-m*NB : NB,
                        NB, //k == NT-1 ? N-k*NB : NB,
                        NB, IB, A, A_offset + NBNBSIZE * (k) + NBNBSIZE * NT * (k), NB, T, T_offset + IBNBSIZE * (k)
                                + IBNBSIZE * NT * (k), IB, B, B_offset + NBNBSIZE * (k) + NBNBSIZE * MT * (n), NB,
                        WORK, 0, NB);
                progress[(k) + MT * (n)] = k;
            } else {
                while (progress[(k) + MT * (n)] != k)
                    Dcommon.delay();
                while (progress[(m) + MT * (n)] != k - 1)
                    Dcommon.delay();
                DcoreBLAS.core_DSSRFB(Dplasma.PlasmaRight, Dplasma.PlasmaRowwise, NB,
                        NB, //n == NT-1 ? N-n*NB : NB,
                        NB, //m == MT-1 ? M-m*NB : NB,
                        IB, NB, B, B_offset + NBNBSIZE * (k) + NBNBSIZE * MT * (n), NB, B, B_offset + NBNBSIZE * (m)
                                + NBNBSIZE * MT * (n), NB, A, A_offset + NBNBSIZE * (k) + NBNBSIZE * NT * (m), NB, T,
                        T_offset + IBNBSIZE * (k) + IBNBSIZE * NT * (m), IB, WORK, 0);
                progress[(m) + MT * (n)] = k;
            }
            m = next_m;
            n = next_n;
            k = next_k;
        }
    }
}
