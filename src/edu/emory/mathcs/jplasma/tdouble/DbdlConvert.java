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

class DbdlConvert {

    private DbdlConvert() {
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Conversion from LAPACK F77 matrix layout to Block Data Layout
     */
    protected static void plasma_lapack_to_bdl(double[] Af77, int Af77_offset, double[] Abdl, int Abdl_offset, int M,
            int N, int LDA, int NB, int MT, int NT, int NBNBSIZE, int cores_num, int my_core_id) {
        int F77_offset;
        int BDL_offset;
        int x, y;
        int X, Y;
        int n, m;
        int next_m;
        int next_n;

        n = 0;
        m = my_core_id;
        while (m >= MT && n < NT) {
            n++;
            m = m - MT;
        }

        while (n < NT) {
            next_m = m;
            next_n = n;

            next_m += cores_num;
            while (next_m >= MT && next_n < NT) {
                next_n++;
                next_m = next_m - MT;
            }

            X = n == NT - 1 ? N - NB * n : NB;
            Y = m == MT - 1 ? M - NB * m : NB;
            F77_offset = Af77_offset + NB * (LDA * n + m);
            BDL_offset = Abdl_offset + NBNBSIZE * (MT * n + m);

            for (x = 0; x < X; x++)
                for (y = 0; y < Y; y++)
                    Abdl[BDL_offset + NB * x + y] = Af77[F77_offset + LDA * x + y];

            m = next_m;
            n = next_n;
        }
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Conversion from Block Data Layout to LAPACK F77 matrix layout
     */
    protected static void plasma_bdl_to_lapack(double[] Abdl, int Abdl_offset, double[] Af77, int Af77_offset, int M,
            int N, int LDA, int NB, int MT, int NT, int NBNBSIZE, int cores_num, int my_core_id) {
        int F77_offset;
        int BDL_offset;
        int x, y;
        int X, Y;
        int n, m;
        int next_m;
        int next_n;

        n = 0;
        m = my_core_id;
        while (m >= MT && n < NT) {
            n++;
            m = m - MT;
        }

        while (n < NT) {
            next_m = m;
            next_n = n;

            next_m += cores_num;
            while (next_m >= MT && next_n < NT) {
                next_n++;
                next_m = next_m - MT;
            }

            X = n == NT - 1 ? N - NB * n : NB;
            Y = m == MT - 1 ? M - NB * m : NB;
            F77_offset = Af77_offset + NB * (LDA * n + m);
            BDL_offset = Abdl_offset + NBNBSIZE * (MT * n + m);

            for (x = 0; x < X; x++)
                for (y = 0; y < Y; y++)
                    Af77[F77_offset + LDA * x + y] = Abdl[BDL_offset + NB * x + y];

            m = next_m;
            n = next_n;
        }
    }
}
