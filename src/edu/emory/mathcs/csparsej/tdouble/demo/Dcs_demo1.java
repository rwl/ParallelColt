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

import edu.emory.mathcs.csparsej.tdouble.Dcs_add;
import edu.emory.mathcs.csparsej.tdouble.Dcs_compress;
import edu.emory.mathcs.csparsej.tdouble.Dcs_entry;
import edu.emory.mathcs.csparsej.tdouble.Dcs_load;
import edu.emory.mathcs.csparsej.tdouble.Dcs_multiply;
import edu.emory.mathcs.csparsej.tdouble.Dcs_norm;
import edu.emory.mathcs.csparsej.tdouble.Dcs_print;
import edu.emory.mathcs.csparsej.tdouble.Dcs_transpose;
import edu.emory.mathcs.csparsej.tdouble.Dcs_util;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;

/**
 * Read a matrix from a file and perform basic matrix operations.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_demo1 {
    public static void main(String[] args) {
        Dcs T = null, A, Eye, AT, C, D;
        int i, m;
        if (args.length == 0) {
            throw new IllegalArgumentException("Usage: java edu.emory.mathcs.csparsej.tdouble.demo.Dcs_demo1 fileName");
        }
        T = Dcs_load.cs_load(args[0]); /* load triplet matrix T from file */
        System.out.print("T:\n");
        Dcs_print.cs_print(T, false); /* print T */
        A = Dcs_compress.cs_compress(T); /* A = compressed-column form of T */
        System.out.print("A:\n");
        Dcs_print.cs_print(A, false); /* print A */
        T = null; /* clear T */
        AT = Dcs_transpose.cs_transpose(A, true); /* AT = A' */
        System.out.print("AT:\n");
        Dcs_print.cs_print(AT, false); /* print AT */
        m = A != null ? A.m : 0; /* m = # of rows of A */
        T = Dcs_util.cs_spalloc(m, m, m, true, true); /* create triplet identity matrix */
        for (i = 0; i < m; i++)
            Dcs_entry.cs_entry(T, i, i, 1);
        Eye = Dcs_compress.cs_compress(T); /* Eye = speye (m) */
        T = null;
        C = Dcs_multiply.cs_multiply(A, AT); /* C = A*A' */
        D = Dcs_add.cs_add(C, Eye, 1, Dcs_norm.cs_norm(C)); /* D = C + Eye*norm (C,1) */
        System.out.print("D:\n");
        Dcs_print.cs_print(D, false); /* print D */
    }
}
