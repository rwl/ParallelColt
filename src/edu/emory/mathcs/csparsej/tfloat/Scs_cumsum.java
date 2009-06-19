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

/**
 * Cumulative sum.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_cumsum {

    /**
     * p [0.f.n] = cumulative sum of c [0.f.n-1], and then copy p [0.f.n-1] into
     * c
     * 
     * @param p
     *            size n+1, cumulative sum of c
     * @param c
     *            size n, overwritten with p [0.f.n-1] on output
     * @param n
     *            length of c
     * @return sum (c), null on error
     */
    public static int cs_cumsum(int[] p, int[] c, int n) {
        int i, nz = 0;
        float nz2 = 0;
        if (p == null || c == null)
            return (-1); /* check inputs */
        for (i = 0; i < n; i++) {
            p[i] = nz;
            nz += c[i];
            nz2 += c[i]; /* also in float to avoid int overflow */
            c[i] = p[i]; /* also copy p[0.f.n-1] back into c[0.f.n-1]*/
        }
        p[n] = nz;
        return (int) nz2; /* return sum (c [0.f.n-1]) */
    }

}
