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

package edu.emory.mathcs.csparsej.tdouble.demo;

import edu.emory.mathcs.csparsej.tdouble.demo.Dcs_demo.Dproblem;

/**
 * Read a matrix from a file and solve a linear system.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_demo2 {
    public static void main(String[] args) {
        Dproblem Prob = null;
        if (args.length == 0) {
            throw new IllegalArgumentException("Usage: java edu.emory.mathcs.csparsej.tdouble.demo.Dcs_demo2 fileName");
        }
        Prob = Dcs_demo.get_problem(args[0], 1e-14);
        Dcs_demo.demo2(Prob);
    }
}
