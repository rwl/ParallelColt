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
 * Column counts for Cholesky and QR.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_counts {

    private static int HEAD(int k, int j, int[] head, int head_offset, boolean ata) {
        return ata ? head[head_offset + k] : j;
    }

    private static int NEXT(int J, int[] next, int next_offset, boolean ata) {
        return ata ? next[next_offset + J] : -1;
    }

    private static int[] init_ata(Dcs AT, int[] post, int[] w) {
        int i, k, p, m = AT.n, n = AT.m, ATp[] = AT.p, ATi[] = AT.i;
        int[] head = w;
        int head_offset = 4 * n;
        int[] next = w;
        int next_offset = 5 * n + 1;
        for (k = 0; k < n; k++)
            w[post[k]] = k; /* invert post */
        for (i = 0; i < m; i++) {
            for (k = n, p = ATp[i]; p < ATp[i + 1]; p++)
                k = Math.min(k, w[ATi[p]]);
            next[next_offset + i] = head[head_offset + k]; /* place row i in linked list k */
            head[head_offset + k] = i;
        }
        return new int[] { head_offset, next_offset };
    }

    /**
     * Column counts of LL'=A or LL'=A'A, given parent & postordering
     * 
     * @param A
     *            column-compressed matrix
     * @param parent
     *            elimination tree of A
     * @param post
     *            postordering of parent
     * @param ata
     *            analyze A if false, A'A otherwise
     * @return column counts of LL'=A or LL'=A'A, null on error
     */
    public static int[] cs_counts(Dcs A, int[] parent, int[] post, boolean ata) {
        int i, j, k, n, m, J, s, p, q, ATp[], ATi[], maxfirst[], prevleaf[], ancestor[], colcount[], w[], first[], delta[];
        int[] head = null, next = null;
        int[] jleaf = new int[1];
        int head_offset = 0, next_offset = 0;
        Dcs AT;
        if (!Dcs_util.CS_CSC(A) || parent == null || post == null)
            return (null); /* check inputs */
        m = A.m;
        n = A.n;
        s = 4 * n + (ata ? (n + m + 1) : 0);
        delta = colcount = new int[n]; /* allocate result */
        w = new int[s]; /* get workspace */
        AT = Dcs_transpose.cs_transpose(A, false); /* AT = A' */
        ancestor = w;
        maxfirst = w;
        int maxfirst_offset = n;
        prevleaf = w;
        int prevleaf_offset = 2 * n;
        first = w;
        int first_offset = 3 * n;
        for (k = 0; k < s; k++)
            w[k] = -1; /* clear workspace w [0..s-1] */
        for (k = 0; k < n; k++) /* find first [j] */
        {
            j = post[k];
            delta[j] = (first[first_offset + j] == -1) ? 1 : 0; /* delta[j]=1 if j is a leaf */
            for (; j != -1 && first[first_offset + j] == -1; j = parent[j])
                first[first_offset + j] = k;
        }
        ATp = AT.p;
        ATi = AT.i;
        if (ata) {
            int[] offsets = init_ata(AT, post, w);
            head = w;
            head_offset = offsets[0];
            next = w;
            next_offset = offsets[1];
        }
        for (i = 0; i < n; i++)
            ancestor[i] = i; /* each node in its own set */
        for (k = 0; k < n; k++) {
            j = post[k]; /* j is the kth node in postordered etree */
            if (parent[j] != -1)
                delta[parent[j]]--; /* j is not a root */
            for (J = HEAD(k, j, head, head_offset, ata); J != -1; J = NEXT(J, next, next_offset, ata)) /* J=j for LL'=A case */
            {
                for (p = ATp[J]; p < ATp[J + 1]; p++) {
                    i = ATi[p];
                    q = Dcs_leaf.cs_leaf(i, j, first, first_offset, maxfirst, maxfirst_offset, prevleaf,
                            prevleaf_offset, ancestor, 0, jleaf);
                    if (jleaf[0] >= 1)
                        delta[j]++; /* A(i,j) is in skeleton */
                    if (jleaf[0] == 2)
                        delta[q]--; /* account for overlap in q */
                }
            }
            if (parent[j] != -1)
                ancestor[j] = parent[j];
        }
        for (j = 0; j < n; j++) /* sum up delta's of each child */
        {
            if (parent[j] != -1)
                colcount[parent[j]] += colcount[j];
        }
        return colcount;
    }
}
