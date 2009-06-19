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
 * Approximate minimum degree ordering.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_amd {

    /* clear w */
    private static int cs_wclear(int mark, int lemax, int[] w, int w_offset, int n) {
        int k;
        if (mark < 2 || (mark + lemax < 0)) {
            for (k = 0; k < n; k++)
                if (w[w_offset + k] != 0)
                    w[w_offset + k] = 1;
            mark = 2;
        }
        return (mark); /* at this point, w [0..n-1] < mark holds */
    }

    /* keep off-diagonal entries; drop diagonal entries */
    private static class Cs_diag implements Dcs_ifkeep {
        @Override
        public boolean fkeep(int i, int j, double aij, Object other) {
            return (i != j);
        }
    }

    /**
     * Minimum degree ordering of A+A' (if A is symmetric) or A'A.
     * 
     * @param order
     *            0:natural, 1:Chol, 2:LU, 3:QR
     * @param A
     *            column-compressed matrix
     * @return amd(A+A') if A is symmetric, or amd(A'A) otherwise, null on error
     *         or for natural ordering
     */
    public static int[] cs_amd(int order, Dcs A) {
        Dcs C, A2, AT;
        int Cp[], Ci[], last[], W[], len[], nv[], next[], P[], head[], elen[], degree[], w[], hhead[], ATp[], ATi[], d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1, k2, k3, jlast, ln, dense, nzmax, mindeg = 0, nvi, nvj, nvk, mark, wnvi, cnz, nel = 0, p, p1, p2, p3, p4, pj, pk, pk1, pk2, pn, q, n, m, t;
        int h;
        boolean ok;
        /* --- Construct matrix C ----------------------------------------------- */
        if (!Dcs_util.CS_CSC(A) || order <= 0 || order > 3)
            return (null); /* check */
        AT = Dcs_transpose.cs_transpose(A, false); /* compute A' */
        if (AT == null)
            return (null);
        m = A.m;
        n = A.n;
        dense = Math.max(16, 10 * (int) Math.sqrt(n)); /* find dense threshold */
        dense = Math.min(n - 2, dense);
        if (order == 1 && n == m) {
            C = Dcs_add.cs_add(A, AT, 0, 0); /* C = A+A' */
        } else if (order == 2) {
            ATp = AT.p; /* drop dense columns from AT */
            ATi = AT.i;
            for (p2 = 0, j = 0; j < m; j++) {
                p = ATp[j]; /* column j of AT starts here */
                ATp[j] = p2; /* new column j starts here */
                if (ATp[j + 1] - p > dense)
                    continue; /* skip dense col j */
                for (; p < ATp[j + 1]; p++)
                    ATi[p2++] = ATi[p];
            }
            ATp[m] = p2; /* finalize AT */
            A2 = Dcs_transpose.cs_transpose(AT, false); /* A2 = AT' */
            C = (A2 != null) ? Dcs_multiply.cs_multiply(AT, A2) : null; /* C=A'*A with no dense rows */
            A2 = null;
        } else {
            C = Dcs_multiply.cs_multiply(AT, A); /* C=A'*A */
        }
        AT = null;
        if (C == null)
            return (null);
        Dcs_fkeep.cs_fkeep(C, new Cs_diag(), null); /* drop diagonal entries */
        Cp = C.p;
        cnz = Cp[n];
        P = new int[n + 1]; /* allocate result */
        W = new int[8 * (n + 1)]; /* get workspace */
        t = cnz + cnz / 5 + 2 * n; /* add elbow room to C */
        Dcs_util.cs_sprealloc(C, t);
        len = W;
        nv = W;
        int nv_offset = n + 1;
        next = W;
        int next_offset = 2 * (n + 1);
        head = W;
        int head_offset = 3 * (n + 1);
        elen = W;
        int elen_offset = 4 * (n + 1);
        degree = W;
        int degree_offset = 5 * (n + 1);
        w = W;
        int w_offset = 6 * (n + 1);
        hhead = W;
        int hhead_offset = 7 * (n + 1);
        last = P; /* use P as workspace for last */
        /* --- Initialize quotient graph ---------------------------------------- */
        for (k = 0; k < n; k++)
            len[k] = Cp[k + 1] - Cp[k];
        len[n] = 0;
        nzmax = C.nzmax;
        Ci = C.i;
        for (i = 0; i <= n; i++) {
            head[head_offset + i] = -1; /* degree list i is empty */
            last[i] = -1;
            next[next_offset + i] = -1;
            hhead[hhead_offset + i] = -1; /* hash list i is empty */
            nv[nv_offset + i] = 1; /* node i is just one node */
            w[w_offset + i] = 1; /* node i is alive */
            elen[elen_offset + i] = 0; /* Ek of node i is empty */
            degree[degree_offset + i] = len[i]; /* degree of node i */
        }
        mark = cs_wclear(0, 0, w, w_offset, n); /* clear w */
        elen[elen_offset + n] = -2; /* n is a dead element */
        Cp[n] = -1; /* n is a root of assembly tree */
        w[w_offset + n] = 0; /* n is a dead element */
        /* --- Initialize degree lists ------------------------------------------ */
        for (i = 0; i < n; i++) {
            d = degree[degree_offset + i];
            if (d == 0) /* node i is empty */
            {
                elen[elen_offset + i] = -2; /* element i is dead */
                nel++;
                Cp[i] = -1; /* i is a root of assembly tree */
                w[w_offset + i] = 0;
            } else if (d > dense) /* node i is dense */
            {
                nv[nv_offset + i] = 0; /* absorb i into element n */
                elen[elen_offset + i] = -1; /* node i is dead */
                nel++;
                Cp[i] = Dcs_util.CS_FLIP(n);
                nv[nv_offset + n]++;
            } else {
                if (head[head_offset + d] != -1)
                    last[head[head_offset + d]] = i;
                next[next_offset + i] = head[head_offset + d]; /* put node i in degree list d */
                head[head_offset + d] = i;
            }
        }
        while (nel < n) /* while (selecting pivots) do */
        {
            /* --- Select node of minimum approximate degree -------------------- */
            for (k = -1; mindeg < n && (k = head[head_offset + mindeg]) == -1; mindeg++)
                ;
            if (next[next_offset + k] != -1)
                last[next[next_offset + k]] = -1;
            head[head_offset + mindeg] = next[next_offset + k]; /* remove k from degree list */
            elenk = elen[elen_offset + k]; /* elenk = |Ek| */
            nvk = nv[nv_offset + k]; /* # of nodes k represents */
            nel += nvk; /* nv[nv_offset+k] nodes of A eliminated */
            /* --- Garbage collection ------------------------------------------- */
            if (elenk > 0 && cnz + mindeg >= nzmax) {
                for (j = 0; j < n; j++) {
                    if ((p = Cp[j]) >= 0) /* j is a live node or element */
                    {
                        Cp[j] = Ci[p]; /* save first entry of object */
                        Ci[p] = Dcs_util.CS_FLIP(j); /* first entry is now CS_FLIP(j) */
                    }
                }
                for (q = 0, p = 0; p < cnz;) /* scan all of memory */
                {
                    if ((j = Dcs_util.CS_FLIP(Ci[p++])) >= 0) /* found object j */
                    {
                        Ci[q] = Cp[j]; /* restore first entry of object */
                        Cp[j] = q++; /* new pointer to object j */
                        for (k3 = 0; k3 < len[j] - 1; k3++)
                            Ci[q++] = Ci[p++];
                    }
                }
                cnz = q; /* Ci [cnz...nzmax-1] now free */
            }
            /* --- Construct new element ---------------------------------------- */
            dk = 0;
            nv[nv_offset + k] = -nvk; /* flag k as in Lk */
            p = Cp[k];
            pk1 = (elenk == 0) ? p : cnz; /* do in place if elen[elen_offset+k] == 0 */
            pk2 = pk1;
            for (k1 = 1; k1 <= elenk + 1; k1++) {
                if (k1 > elenk) {
                    e = k; /* search the nodes in k */
                    pj = p; /* list of nodes starts at Ci[pj]*/
                    ln = len[k] - elenk; /* length of list of nodes in k */
                } else {
                    e = Ci[p++]; /* search the nodes in e */
                    pj = Cp[e];
                    ln = len[e]; /* length of list of nodes in e */
                }
                for (k2 = 1; k2 <= ln; k2++) {
                    i = Ci[pj++];
                    if ((nvi = nv[nv_offset + i]) <= 0)
                        continue; /* node i dead, or seen */
                    dk += nvi; /* degree[degree_offset+Lk] += size of node i */
                    nv[nv_offset + i] = -nvi; /* negate nv[nv_offset+i] to denote i in Lk*/
                    Ci[pk2++] = i; /* place i in Lk */
                    if (next[next_offset + i] != -1)
                        last[next[next_offset + i]] = last[i];
                    if (last[i] != -1) /* remove i from degree list */
                    {
                        next[next_offset + last[i]] = next[next_offset + i];
                    } else {
                        head[head_offset + degree[degree_offset + i]] = next[next_offset + i];
                    }
                }
                if (e != k) {
                    Cp[e] = Dcs_util.CS_FLIP(k); /* absorb e into k */
                    w[w_offset + e] = 0; /* e is now a dead element */
                }
            }
            if (elenk != 0)
                cnz = pk2; /* Ci [cnz...nzmax] is free */
            degree[degree_offset + k] = dk; /* external degree of k - |Lk\i| */
            Cp[k] = pk1; /* element k is in Ci[pk1..pk2-1] */
            len[k] = pk2 - pk1;
            elen[elen_offset + k] = -2; /* k is now an element */
            /* --- Find set differences ----------------------------------------- */
            mark = cs_wclear(mark, lemax, w, w_offset, n); /* clear w if necessary */
            for (pk = pk1; pk < pk2; pk++) /* scan 1: find |Le\Lk| */
            {
                i = Ci[pk];
                if ((eln = elen[elen_offset + i]) <= 0)
                    continue;/* skip if elen[elen_offset+i] empty */
                nvi = -nv[nv_offset + i]; /* nv [i] was negated */
                wnvi = mark - nvi;
                for (p = Cp[i]; p <= Cp[i] + eln - 1; p++) /* scan Ei */
                {
                    e = Ci[p];
                    if (w[w_offset + e] >= mark) {
                        w[w_offset + e] -= nvi; /* decrement |Le\Lk| */
                    } else if (w[w_offset + e] != 0) /* ensure e is a live element */
                    {
                        w[w_offset + e] = degree[degree_offset + e] + wnvi; /* 1st time e seen in scan 1 */
                    }
                }
            }
            /* --- Degree update ------------------------------------------------ */
            for (pk = pk1; pk < pk2; pk++) /* scan2: degree update */
            {
                i = Ci[pk]; /* consider node i in Lk */
                p1 = Cp[i];
                p2 = p1 + elen[elen_offset + i] - 1;
                pn = p1;
                for (h = 0, d = 0, p = p1; p <= p2; p++) /* scan Ei */
                {
                    e = Ci[p];
                    if (w[w_offset + e] != 0) /* e is an unabsorbed element */
                    {
                        dext = w[w_offset + e] - mark; /* dext = |Le\Lk| */
                        if (dext > 0) {
                            d += dext; /* sum up the set differences */
                            Ci[pn++] = e; /* keep e in Ei */
                            h += e; /* compute the hash of node i */
                        } else {
                            Cp[e] = Dcs_util.CS_FLIP(k); /* aggressive absorb. e.k */
                            w[w_offset + e] = 0; /* e is a dead element */
                        }
                    }
                }
                elen[elen_offset + i] = pn - p1 + 1; /* elen[elen_offset+i] = |Ei| */
                p3 = pn;
                p4 = p1 + len[i];
                for (p = p2 + 1; p < p4; p++) /* prune edges in Ai */
                {
                    j = Ci[p];
                    if ((nvj = nv[nv_offset + j]) <= 0)
                        continue; /* node j dead or in Lk */
                    d += nvj; /* degree(i) += |j| */
                    Ci[pn++] = j; /* place j in node list of i */
                    h += j; /* compute hash for node i */
                }
                if (d == 0) /* check for mass elimination */
                {
                    Cp[i] = Dcs_util.CS_FLIP(k); /* absorb i into k */
                    nvi = -nv[nv_offset + i];
                    dk -= nvi; /* |Lk| -= |i| */
                    nvk += nvi; /* |k| += nv[nv_offset+i] */
                    nel += nvi;
                    nv[nv_offset + i] = 0;
                    elen[elen_offset + i] = -1; /* node i is dead */
                } else {
                    degree[degree_offset + i] = Math.min(degree[degree_offset + i], d); /* update degree(i) */
                    Ci[pn] = Ci[p3]; /* move first node to end */
                    Ci[p3] = Ci[p1]; /* move 1st el. to end of Ei */
                    Ci[p1] = k; /* add k as 1st element in of Ei */
                    len[i] = pn - p1 + 1; /* new len of adj. list of node i */
                    h %= n; /* finalize hash of i */
                    next[next_offset + i] = hhead[hhead_offset + h]; /* place i in hash bucket */
                    hhead[hhead_offset + h] = i;
                    last[i] = h; /* save hash of i in last[i] */
                }
            } /* scan2 is done */
            degree[degree_offset + k] = dk; /* finalize |Lk| */
            lemax = Math.max(lemax, dk);
            mark = cs_wclear(mark + lemax, lemax, w, w_offset, n); /* clear w */
            /* --- Supernode detection ------------------------------------------ */
            for (pk = pk1; pk < pk2; pk++) {
                i = Ci[pk];
                if (nv[nv_offset + i] >= 0)
                    continue; /* skip if i is dead */
                h = last[i]; /* scan hash bucket of node i */
                i = hhead[hhead_offset + h];
                hhead[hhead_offset + h] = -1; /* hash bucket will be empty */
                for (; i != -1 && next[next_offset + i] != -1; i = next[next_offset + i], mark++) {
                    ln = len[i];
                    eln = elen[elen_offset + i];
                    for (p = Cp[i] + 1; p <= Cp[i] + ln - 1; p++)
                        w[w_offset + Ci[p]] = mark;
                    jlast = i;
                    for (j = next[next_offset + i]; j != -1;) /* compare i with all j */
                    {
                        ok = (len[j] == ln) && (elen[elen_offset + j] == eln);
                        for (p = Cp[j] + 1; ok && p <= Cp[j] + ln - 1; p++) {
                            if (w[w_offset + Ci[p]] != mark)
                                ok = false; /* compare i and j*/
                        }
                        if (ok) /* i and j are identical */
                        {
                            Cp[j] = Dcs_util.CS_FLIP(i); /* absorb j into i */
                            nv[nv_offset + i] += nv[nv_offset + j];
                            nv[nv_offset + j] = 0;
                            elen[elen_offset + j] = -1; /* node j is dead */
                            j = next[next_offset + j]; /* delete j from hash bucket */
                            next[next_offset + jlast] = j;
                        } else {
                            jlast = j; /* j and i are different */
                            j = next[next_offset + j];
                        }
                    }
                }
            }
            /* --- Finalize new element------------------------------------------ */
            for (p = pk1, pk = pk1; pk < pk2; pk++) /* finalize Lk */
            {
                i = Ci[pk];
                if ((nvi = -nv[nv_offset + i]) <= 0)
                    continue;/* skip if i is dead */
                nv[nv_offset + i] = nvi; /* restore nv[nv_offset+i] */
                d = degree[degree_offset + i] + dk - nvi; /* compute external degree(i) */
                d = Math.min(d, n - nel - nvi);
                if (head[head_offset + d] != -1)
                    last[head[head_offset + d]] = i;
                next[next_offset + i] = head[head_offset + d]; /* put i back in degree list */
                last[i] = -1;
                head[head_offset + d] = i;
                mindeg = Math.min(mindeg, d); /* find new minimum degree */
                degree[degree_offset + i] = d;
                Ci[p++] = i; /* place i in Lk */
            }
            nv[nv_offset + k] = nvk; /* # nodes absorbed into k */
            if ((len[k] = p - pk1) == 0) /* length of adj list of element k*/
            {
                Cp[k] = -1; /* k is a root of the tree */
                w[w_offset + k] = 0; /* k is now a dead element */
            }
            if (elenk != 0)
                cnz = p; /* free unused space in Lk */
        }
        /* --- Postordering ----------------------------------------------------- */
        for (i = 0; i < n; i++)
            Cp[i] = Dcs_util.CS_FLIP(Cp[i]);/* fix assembly tree */
        for (j = 0; j <= n; j++)
            head[head_offset + j] = -1;
        for (j = n; j >= 0; j--) /* place unordered nodes in lists */
        {
            if (nv[nv_offset + j] > 0)
                continue; /* skip if j is an element */
            next[next_offset + j] = head[head_offset + Cp[j]]; /* place j in list of its parent */
            head[head_offset + Cp[j]] = j;
        }
        for (e = n; e >= 0; e--) /* place elements in lists */
        {
            if (nv[nv_offset + e] <= 0)
                continue; /* skip unless e is an element */
            if (Cp[e] != -1) {
                next[next_offset + e] = head[head_offset + Cp[e]]; /* place e in list of its parent */
                head[head_offset + Cp[e]] = e;
            }
        }
        for (k = 0, i = 0; i <= n; i++) /* postorder the assembly tree */
        {
            if (Cp[i] == -1)
                k = Dcs_tdfs.cs_tdfs(i, k, head, head_offset, next, next_offset, P, 0, w, w_offset);
        }
        return P;
    }
}
