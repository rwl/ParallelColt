/* ***** BEGIN LICENSE BLOCK *****
 * 
 * CSparse: a Concise Sparse matrix package.
 * Copyright (c) 2006, Timothy A. Davis.
 * http://www.cise.ufl.edu/research/sparse/CSparse
 *
 * -------------------------------------------------------------------------
 * 
 * CSparseJ is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * CSparseJ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this Module; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.csparsej.tdouble;

import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;

/**
 * Print a sparse matrix.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_print {

    /**
     * Prints a sparse matrix.
     * 
     * @param A
     *            sparse matrix (triplet ot column-compressed)
     * @param brief
     *            print all of A if false, a few entries otherwise
     * @return true if successful, false on error
     */
    public static boolean cs_print(Dcs A, boolean brief) {
        int p, j, m, n, nzmax, nz, Ap[], Ai[];
        double Ax[];
        if (A == null) {
            System.out.print("(null)\n");
            return (false);
        }
        m = A.m;
        n = A.n;
        Ap = A.p;
        Ai = A.i;
        Ax = A.x;
        nzmax = A.nzmax;
        nz = A.nz;
        System.out.print(String.format("CSparseJ Version %d.%d.%d, %s.  %s\n", Dcs_common.CS_VER, Dcs_common.CS_SUBVER,
                Dcs_common.CS_SUBSUB, Dcs_common.CS_DATE, Dcs_common.CS_COPYRIGHT));
        if (nz < 0) {
            System.out.print(String.format("%d-by-%d, nzmax: %d nnz: %d, 1-norm: %g\n", m, n, nzmax, Ap[n], Dcs_norm
                    .cs_norm(A)));
            for (j = 0; j < n; j++) {
                System.out.print(String.format("    col %d : locations %d to %d\n", j, Ap[j], Ap[j + 1] - 1));
                for (p = Ap[j]; p < Ap[j + 1]; p++) {
                    System.out.print(String.format("      %d : %g\n", Ai[p], Ax != null ? Ax[p] : 1));
                    if (brief && p > 20) {
                        System.out.print("  ...\n");
                        return (true);
                    }
                }
            }
        } else {
            System.out.print(String.format("triplet: %d-by-%d, nzmax: %d nnz: %d\n", m, n, nzmax, nz));
            for (p = 0; p < nz; p++) {
                System.out.print(String.format("    %d %d : %g\n", Ai[p], Ap[p], Ax != null ? Ax[p] : 1));
                if (brief && p > 20) {
                    System.out.print("  ...\n");
                    return (true);
                }
            }
        }
        return (true);
    }

}
