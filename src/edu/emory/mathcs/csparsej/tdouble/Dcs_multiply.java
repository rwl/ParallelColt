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
 * Sparse matrix multiply.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_multiply {

    /**
     * Sparse matrix multiplication, C = A*B
     * 
     * @param A
     *            column-compressed matrix
     * @param B
     *            column-compressed matrix
     * @return C = A*B, null on error
     */
    public static Dcs cs_multiply(Dcs A, Dcs B) {
        int p, j, nz = 0, anz, Cp[], Ci[], Bp[], m, n, bnz, w[], Bi[];
        double x[], Bx[], Cx[];
        boolean values;
        Dcs C;
        if (!Dcs_util.CS_CSC(A) || !Dcs_util.CS_CSC(B))
            return (null); /* check inputs */
        if (A.n != B.m)
            return (null);
        m = A.m;
        anz = A.p[A.n];
        n = B.n;
        Bp = B.p;
        Bi = B.i;
        Bx = B.x;
        bnz = Bp[n];
        w = new int[m]; /* get workspace */
        values = (A.x != null) && (Bx != null);
        x = values ? new double[m] : null; /* get workspace */
        C = Dcs_util.cs_spalloc(m, n, anz + bnz, values, false); /* allocate result */
        Cp = C.p;
        for (j = 0; j < n; j++) {
            if (nz + m > C.nzmax) {
                Dcs_util.cs_sprealloc(C, 2 * (C.nzmax) + m);
            }
            Ci = C.i;
            Cx = C.x; /* C.i and C.x may be reallocated */
            Cp[j] = nz; /* column j of C starts here */
            for (p = Bp[j]; p < Bp[j + 1]; p++) {
                nz = Dcs_scatter.cs_scatter(A, Bi[p], (Bx != null) ? Bx[p] : 1, w, x, j + 1, C, nz);
            }
            if (values)
                for (p = Cp[j]; p < nz; p++)
                    Cx[p] = x[Ci[p]];
        }
        Cp[n] = nz; /* finalize the last column of C */
        Dcs_util.cs_sprealloc(C, 0); /* remove extra space from C */
        return C;
    }

}
