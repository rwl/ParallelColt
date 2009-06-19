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
 * Add an entry to a triplet matrix.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_entry {
    /**
     * Adds an entry to a triplet matrix. Memory-space and dimension of T are
     * increased if necessary.
     * 
     * @param T
     *            triplet matrix; new entry added on output
     * @param i
     *            row index of new entry
     * @param j
     *            column index of new entry
     * @param x
     *            numerical value of new entry
     * @return true if successful, false otherwise
     */
    public static boolean cs_entry(Dcs T, int i, int j, double x) {
        if (!Dcs_util.CS_TRIPLET(T) || i < 0 || j < 0)
            return (false); /* check inputs */
        if (T.nz >= T.nzmax) {
            Dcs_util.cs_sprealloc(T, 2 * (T.nzmax));
        }
        if (T.x != null)
            T.x[T.nz] = x;
        T.i[T.nz] = i;
        T.p[T.nz++] = j;
        T.m = Math.max(T.m, i + 1);
        T.n = Math.max(T.n, j + 1);
        return (true);
    }
}
