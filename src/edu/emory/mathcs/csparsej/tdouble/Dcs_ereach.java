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
 * Nonzero pattern of kth row of Cholesky factor, L(k,1:k-1).
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_ereach {

    /**
     * Find nonzero pattern of Cholesky L(k,1:k-1) using etree and triu(A(:,k)).
     * If ok, s[top..n-1] contains pattern of L(k,:).
     * 
     * @param A
     *            column-compressed matrix; L is the Cholesky factor of A
     * @param k
     *            find kth row of L
     * @param parent
     *            elimination tree of A
     * @param s
     *            size n, s[top..n-1] is nonzero pattern of L(k,1:k-1)
     * @param s_offset
     *            the index of the first element in array s
     * @param w
     *            size n, work array, w[0..n-1]>=0 on input, unchanged on output
     * 
     * @return top in successful, -1 on error
     */
    public static int cs_ereach(Dcs A, int k, int[] parent, int[] s, int s_offset, int[] w) {
        int i, p, n, len, top, Ap[], Ai[];
        if (!Dcs_util.CS_CSC(A) || parent == null || s == null || w == null)
            return (-1); /* check inputs */
        top = n = A.n;
        Ap = A.p;
        Ai = A.i;
        Dcs_util.CS_MARK(w, k); /* mark node k as visited */
        for (p = Ap[k]; p < Ap[k + 1]; p++) {
            i = Ai[p]; /* A(i,k) is nonzero */
            if (i > k)
                continue; /* only use upper triangular part of A */
            for (len = 0; !Dcs_util.CS_MARKED(w, i); i = parent[i]) /* traverse up etree*/
            {
                s[s_offset + len++] = i; /* L(k,i) is nonzero */
                Dcs_util.CS_MARK(w, i); /* mark i as visited */
            }
            while (len > 0)
                s[s_offset + --top] = s[s_offset + --len]; /* push path onto stack */
        }
        for (p = top; p < n; p++)
            Dcs_util.CS_MARK(w, s[s_offset + p]); /* unmark all nodes */
        Dcs_util.CS_MARK(w, k); /* unmark node k */
        return (top); /* s [top..n-1] contains pattern of L(k,:)*/
    }

}
