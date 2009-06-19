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
 * Septh-first-search of a tree.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_tdfs {

    /**
     * Septh-first search and postorder of a tree rooted at node j
     * 
     * @param j
     *            postorder of a tree rooted at node j
     * @param k
     *            number of nodes ordered so far
     * @param head
     *            head[i] is first child of node i; -1 on output
     * @param head_offset
     *            the index of the first element in array head
     * @param next
     *            next[i] is next sibling of i or -1 if none
     * @param next_offset
     *            the index of the first element in array next
     * @param post
     *            postordering
     * @param post_offset
     *            the index of the first element in array post
     * @param stack
     *            size n, work array
     * @param stack_offset
     *            the index of the first element in array stack
     * @return new value of k, -1 on error
     */
    public static int cs_tdfs(int j, int k, int[] head, int head_offset, int[] next, int next_offset, int[] post,
            int post_offset, int[] stack, int stack_offset) {
        int i, p, top = 0;
        if (head == null || next == null || post == null || stack == null)
            return (-1); /* check inputs */
        stack[stack_offset + 0] = j; /* place j on the stack */
        while (top >= 0) /* while (stack is not empty) */
        {
            p = stack[stack_offset + top]; /* p = top of stack */
            i = head[head_offset + p]; /* i = youngest child of p */
            if (i == -1) {
                top--; /* p has no unordered children left */
                post[post_offset + (k++)] = p; /* node p is the kth postordered node */
            } else {
                head[head_offset + p] = next[next_offset + i]; /* remove i from children of p */
                stack[stack_offset + (++top)] = i; /* start dfs on child node i */
            }
        }
        return (k);
    }

}
