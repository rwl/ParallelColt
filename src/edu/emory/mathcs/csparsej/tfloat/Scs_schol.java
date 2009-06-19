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
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scss;

/**
 * Symbolic Cholesky ordering and analysis.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_schol {
    /**
     * Ordering and symbolic analysis for a Cholesky factorization.
     * 
     * @param order
     *            ordering option (0 or 1)
     * @param A
     *            column-compressed matrix
     * @return symbolic analysis for Cholesky, null on error
     */
    public static Scss cs_schol(int order, Scs A) {
        int n, c[], post[], P[];
        Scs C;
        Scss S;
        if (!Scs_util.CS_CSC(A))
            return (null); /* check inputs */
        n = A.n;
        S = new Scss(); /* allocate result S */
        P = Scs_amd.cs_amd(order, A); /* P = amd(A+A'), or natural */
        S.pinv = Scs_pinv.cs_pinv(P, n); /* find inverse permutation */
        if (order != 0 && S.pinv == null)
            return null;
        C = Scs_symperm.cs_symperm(A, S.pinv, false); /* C = spones(triu(A(P,P))) */
        S.parent = Scs_etree.cs_etree(C, false); /* find etree of C */
        post = Scs_post.cs_post(S.parent, n); /* postorder the etree */
        c = Scs_counts.cs_counts(C, S.parent, post, false); /* find column counts of chol(C) */
        S.cp = new int[n + 1]; /* allocate result S.cp */
        S.unz = S.lnz = Scs_cumsum.cs_cumsum(S.cp, c, n); /* find column pointers for L */
        return ((S.lnz >= 0) ? S : null);
    }
}
