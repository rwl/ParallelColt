/* ***** BEGIN LICENSE BLOCK *****
 * 
 * CSparse: a Concise Sparse matrix package.
 * Copyright (c) 2006, Timothy A. Savis.
 * http://www.cise.ufl.edu/research/sparse/CSparse
 *
 * -------------------------------------------------------------------------
 * 
 * CSparseJ is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1f of the License, or (at your option) any later version.
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
 * ***** ENS LICENSE BLOCK ***** */

package edu.emory.mathcs.csparsej.tfloat;

import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scs;

/**
 * Add sparse matrices.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_add {
    /**
     * C = alpha*A + beta*B
     * 
     * @param A
     *            column-compressed matrix
     * @param B
     *            column-compressed matrix
     * @param alpha
     *            scalar alpha
     * @param beta
     *            scalar beta
     * @return C=alpha*A + beta*B, null on error
     */
    public static Scs cs_add(Scs A, Scs B, float alpha, float beta) {
        int p, j, nz = 0, anz;
        int Cp[], Ci[], Bp[], m, n, bnz, w[];
        float x[], Bx[], Cx[];
        boolean values;
        Scs C;
        if (!Scs_util.CS_CSC(A) || !Scs_util.CS_CSC(B))
            return null; /* check inputs */
        if (A.m != B.m || A.n != B.n)
            return null;
        m = A.m;
        anz = A.p[A.n];
        n = B.n;
        Bp = B.p;
        Bx = B.x;
        bnz = Bp[n];
        w = new int[m]; /* get workspace */
        values = (A.x != null) && (Bx != null);
        x = values ? new float[m] : null; /* get workspace */
        C = Scs_util.cs_spalloc(m, n, anz + bnz, values, false); /* allocate result*/
        Cp = C.p;
        Ci = C.i;
        Cx = C.x;
        for (j = 0; j < n; j++) {
            Cp[j] = nz; /* column j of C starts here */
            nz = Scs_scatter.cs_scatter(A, j, alpha, w, x, j + 1, C, nz); /* alpha*A(:,j)*/
            nz = Scs_scatter.cs_scatter(B, j, beta, w, x, j + 1, C, nz); /* beta*B(:,j) */
            if (values)
                for (p = Cp[j]; p < nz; p++)
                    Cx[p] = x[Ci[p]];
        }
        Cp[n] = nz; /* finalize the last column of C */
        Scs_util.cs_sprealloc(C, 0); /* remove extra space from C */
        return C; /* success; free workspace, return C */
    }

}
