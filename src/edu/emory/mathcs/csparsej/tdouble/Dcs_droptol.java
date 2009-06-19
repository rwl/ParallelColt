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
 * Drop small entries from a sparse matrix.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_droptol {

    private static class Cs_tol implements Dcs_ifkeep {
        @Override
        public boolean fkeep(int i, int j, double aij, Object other) {
            return (Math.abs(aij) > (Double) other);
        }
    }

    /**
     * Removes entries from a matrix with absolute value <= tol.
     * 
     * @param A
     *            column-compressed matrix
     * @param tol
     *            drop tolerance
     * @return nz, new number of entries in A, -1 on error
     */
    public static int cs_droptol(Dcs A, double tol) {
        return (Dcs_fkeep.cs_fkeep(A, new Cs_tol(), tol)); /* keep all large entries */
    }
}
