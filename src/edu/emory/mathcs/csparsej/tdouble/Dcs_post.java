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

/**
 * Postorder a tree or forest.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_post {

    /**
     * Postorders a tree of forest.
     * 
     * @param parent
     *            defines a tree of n nodes
     * @param n
     *            length of parent
     * @return post[k]=i, null on error
     */
    public static int[] cs_post(int[] parent, int n) {
        int j, k = 0, post[], w[], head[], next[], stack[];
        if (parent == null)
            return (null); /* check inputs */
        post = new int[n]; /* allocate result */
        w = new int[3 * n]; /* get workspace */
        head = w;
        next = w;
        int next_offset = n;
        stack = w;
        int stack_offset = 2 * n;
        for (j = 0; j < n; j++)
            head[j] = -1; /* empty linked lists */
        for (j = n - 1; j >= 0; j--) /* traverse nodes in reverse order*/
        {
            if (parent[j] == -1)
                continue; /* j is a root */
            next[next_offset + j] = head[parent[j]]; /* add j to list of its parent */
            head[parent[j]] = j;
        }
        for (j = 0; j < n; j++) {
            if (parent[j] != -1)
                continue; /* skip j if it is not a root */
            k = Dcs_tdfs.cs_tdfs(j, k, head, 0, next, next_offset, post, 0, stack, stack_offset);
        }
        return post;
    }

}
