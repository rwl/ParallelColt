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
 * Sparse lower or upper triangular solve. x=G\b where G, x, and b are sparse,
 * and G upper/lower triangular.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_spsolve {
    /**
     * solve Gx=b(:,k), where G is either upper (lo=false) or lower (lo=true)
     * triangular.
     * 
     * @param G
     *            lower or upper triangular matrix in column-compressed form
     * @param B
     *            right hand side, b=B(:,k)
     * @param k
     *            use kth column of B as right hand side
     * @param xi
     *            size 2*n, nonzero pattern of x in xi[top..n-1]
     * @param x
     *            size n, x in x[xi[top..n-1]]
     * @param pinv
     *            mapping of rows to columns of G, ignored if null
     * @param lo
     *            true if lower triangular, false if upper
     * @return top, -1 in error
     */
    public static int cs_spsolve(Scs G, Scs B, int k, int[] xi, float[] x, int[] pinv, boolean lo) {
        int j, J, p, q, px, top, n, Gp[], Gi[], Bp[], Bi[];
        float Gx[], Bx[];
        if (!Scs_util.CS_CSC(G) || !Scs_util.CS_CSC(B) || xi == null || x == null)
            return (-1);
        Gp = G.p;
        Gi = G.i;
        Gx = G.x;
        n = G.n;
        Bp = B.p;
        Bi = B.i;
        Bx = B.x;
        top = Scs_reach.cs_reach(G, B, k, xi, pinv); /* xi[top..n-1]=Reach(B(:,k)) */
        for (p = top; p < n; p++)
            x[xi[p]] = 0; /* clear x */
        for (p = Bp[k]; p < Bp[k + 1]; p++)
            x[Bi[p]] = Bx[p]; /* scatter B */
        for (px = top; px < n; px++) {
            j = xi[px]; /* x(j) is nonzero */
            J = pinv != null ? (pinv[j]) : j; /* j maps to col J of G */
            if (J < 0)
                continue; /* column J is empty */
            x[j] /= Gx[lo ? (Gp[J]) : (Gp[J + 1] - 1)];/* x(j) /= G(j,j) */
            p = lo ? (Gp[J] + 1) : (Gp[J]); /* lo: L(j,j) 1st entry */
            q = lo ? (Gp[J + 1]) : (Gp[J + 1] - 1); /* up: U(j,j) last entry */
            for (; p < q; p++) {
                x[Gi[p]] -= Gx[p] * x[j]; /* x(i) -= G(i,j) * x(j) */
            }
        }
        return (top); /* return top of stack */
    }
}
