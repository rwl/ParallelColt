/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong;

import cern.colt.matrix.tlong.impl.DenseLongMatrix1D;
import cern.colt.matrix.tlong.impl.SparseLongMatrix1D;
import cern.jet.math.tlong.LongFunctions;

/**
 * Factory for convenient construction of 1-d matrices holding <tt>int</tt>
 * cells. Use idioms like <tt>LongFactory1D.dense.make(1000)</tt> to construct
 * dense matrices, <tt>LongFactory1D.sparse.make(1000)</tt> to construct sparse
 * matrices.
 * 
 * If the factory is used frequently it might be useful to streamline the
 * notation. For example by aliasing:
 * <table>
 * <td class="PRE">
 * 
 * <pre>
 *  LongFactory1D F = LongFactory1D.dense;
 *  F.make(1000);
 *  F.descending(10);
 *  F.random(3);
 *  ...
 * </pre>
 * 
 * </td>
 * </table>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
public class LongFactory1D extends cern.colt.PersistentObject {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /**
     * A factory producing dense matrices.
     */
    public static final LongFactory1D dense = new LongFactory1D();

    /**
     * A factory producing sparse matrices.
     */
    public static final LongFactory1D sparse = new LongFactory1D();

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected LongFactory1D() {
    }

    /**
     * C = A||B; Constructs a new matrix which is the concatenation of two other
     * matrices. Example: <tt>0 1</tt> append <tt>3 4</tt> --> <tt>0 1 3 4</tt>.
     */
    public LongMatrix1D append(LongMatrix1D A, LongMatrix1D B) {
        // concatenate
        LongMatrix1D matrix = make((int) (A.size() + B.size()));
        matrix.viewPart(0, (int) A.size()).assign(A);
        matrix.viewPart((int) A.size(), (int) B.size()).assign(B);
        return matrix;
    }

    /**
     * Constructs a matrix with cells having ascending values. For debugging
     * purposes. Example: <tt>0 1 2</tt>
     */
    public LongMatrix1D ascending(int size) {
        cern.jet.math.tlong.LongFunctions F = cern.jet.math.tlong.LongFunctions.longFunctions;
        return descending(size).assign(LongFunctions.chain(LongFunctions.neg, LongFunctions.minus(size)));
    }

    /**
     * Constructs a matrix with cells having descending values. For debugging
     * purposes. Example: <tt>2 1 0</tt>
     */
    public LongMatrix1D descending(int size) {
        LongMatrix1D matrix = make(size);
        int v = 0;
        for (int i = size; --i >= 0;) {
            matrix.setQuick(i, v++);
        }
        return matrix;
    }

    /**
     * Constructs a matrix with the given cell values. The values are copied. So
     * subsequent changes in <tt>values</tt> are not reflected in the matrix,
     * and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     */
    public LongMatrix1D make(long[] values) {
        if (this == sparse)
            return new SparseLongMatrix1D(values);
        else
            return new DenseLongMatrix1D(values);
    }

    /**
     * Constructs a matrix which is the concatenation of all given parts. Cells
     * are copied.
     */
    public LongMatrix1D make(LongMatrix1D[] parts) {
        if (parts.length == 0)
            return make(0);

        int size = 0;
        for (int i = 0; i < parts.length; i++)
            size += parts[i].size();

        LongMatrix1D vector = make(size);
        size = 0;
        for (int i = 0; i < parts.length; i++) {
            vector.viewPart(size, (int) parts[i].size()).assign(parts[i]);
            size += parts[i].size();
        }

        return vector;
    }

    /**
     * Constructs a matrix with the given shape, each cell initialized with
     * zero.
     */
    public LongMatrix1D make(int size) {
        if (this == sparse)
            return new SparseLongMatrix1D(size);
        return new DenseLongMatrix1D(size);
    }

    /**
     * Constructs a matrix with the given shape, each cell initialized with the
     * given value.
     */
    public LongMatrix1D make(int size, long initialValue) {
        return make(size).assign(initialValue);
    }

    /**
     * Constructs a matrix from the values of the given list. The values are
     * copied. So subsequent changes in <tt>values</tt> are not reflected in the
     * matrix, and vice-versa.
     * 
     * @param values
     *            The values to be filled into the new matrix.
     * @return a new matrix.
     */
    public LongMatrix1D make(cern.colt.list.tlong.AbstractLongList values) {
        int size = values.size();
        LongMatrix1D vector = make(size);
        for (int i = size; --i >= 0;)
            vector.set(i, values.get(i));
        return vector;
    }

    /**
     * Constructs a matrix with uniformly distributed values in <tt>(0,1)</tt>
     * (exclusive).
     */
    public LongMatrix1D random(int size) {
        return make(size).assign(cern.jet.math.tlong.LongFunctions.random());
    }

    /**
     * C = A||A||..||A; Constructs a new matrix which is concatenated
     * <tt>repeat</tt> times. Example:
     * 
     * <pre>
     *   0 1
     *   repeat(3) --&gt;
     *   0 1 0 1 0 1
     * 
     * </pre>
     */
    public LongMatrix1D repeat(LongMatrix1D A, int repeat) {
        int size = (int) A.size();
        LongMatrix1D matrix = make(repeat * size);
        for (int i = repeat; --i >= 0;) {
            matrix.viewPart(size * i, size).assign(A);
        }
        return matrix;
    }

    /**
     * Constructs a randomly sampled matrix with the given shape. Randomly picks
     * exactly <tt>Math.round(size*nonZeroFraction)</tt> cells and initializes
     * them to <tt>value</tt>, all the rest will be initialized to zero. Note
     * that this is not the same as setting each cell with probability
     * <tt>nonZeroFraction</tt> to <tt>value</tt>.
     * 
     * @throws IllegalArgumentException
     *             if <tt>nonZeroFraction < 0 || nonZeroFraction > 1</tt>.
     * @see cern.jet.random.tdouble.sampling.DoubleRandomSamplingAssistant
     */
    public LongMatrix1D sample(int size, int value, int nonZeroFraction) {
        double epsilon = 1e-09;
        if (nonZeroFraction < 0 - epsilon || nonZeroFraction > 1 + epsilon)
            throw new IllegalArgumentException();
        if (nonZeroFraction < 0)
            nonZeroFraction = 0;
        if (nonZeroFraction > 1)
            nonZeroFraction = 1;

        LongMatrix1D matrix = make(size);

        int n = Math.round(size * nonZeroFraction);
        if (n == 0)
            return matrix;

        cern.jet.random.tdouble.sampling.DoubleRandomSamplingAssistant sampler = new cern.jet.random.tdouble.sampling.DoubleRandomSamplingAssistant(
                n, size, new cern.jet.random.tdouble.engine.DoubleMersenneTwister());
        for (int i = size; --i >= 0;) {
            if (sampler.sampleNextElement()) {
                matrix.set(i, value);
            }
        }

        return matrix;
    }

    /**
     * Constructs a list from the given matrix. The values are copied. So
     * subsequent changes in <tt>values</tt> are not reflected in the list, and
     * vice-versa.
     * 
     * @param values
     *            The values to be filled into the new list.
     * @return a new list.
     */
    public cern.colt.list.tlong.LongArrayList toList(LongMatrix1D values) {
        int size = (int) values.size();
        cern.colt.list.tlong.LongArrayList list = new cern.colt.list.tlong.LongArrayList(size);
        list.setSize(size);
        for (int i = size; --i >= 0;)
            list.set(i, values.get(i));
        return list;
    }
}
