/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.map.tfloat;

import java.util.concurrent.Future;

import cern.colt.function.tfloat.LongFloatProcedure;
import cern.colt.function.tlong.LongProcedure;
import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.map.AbstractMap;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Abstract base class for hash maps holding (key,value) associations of type
 * <tt>(int-->float)</tt>. First see the <a href="package-summary.html">package
 * summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the
 * broad picture.
 * <p>
 * <b>Implementation</b>:
 * <p>
 * Almost all methods are expressed in terms of
 * {@link #forEachKey(LongProcedure)}. As such they are fully functional, but
 * inefficient. Override them in subclasses if necessary.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * @see java.util.HashMap
 */
public abstract class AbstractLongFloatMap extends AbstractMap {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    // public static int hashCollisions = 0; // for debug only
    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected AbstractLongFloatMap() {
    }

    /**
     * Assigns the result of a function to each value;
     * <tt>v[i] = function(v[i])</tt>.
     * 
     * @param function
     *            a function object taking as argument the current association's
     *            value.
     */
    public void assign(final cern.colt.function.tfloat.FloatFunction function) {
        copy().forEachPair(new cern.colt.function.tfloat.LongFloatProcedure() {
            public boolean apply(long key, float value) {
                put(key, function.apply(value));
                return true;
            }
        });
    }

    /**
     * Clears the receiver, then adds all (key,value) pairs of <tt>other</tt>
     * values to it.
     * 
     * @param other
     *            the other map to be copied into the receiver.
     */
    public void assign(AbstractLongFloatMap other) {
        clear();
        other.forEachPair(new LongFloatProcedure() {
            public boolean apply(long key, float value) {
                put(key, value);
                return true;
            }
        });
    }

    /**
     * Returns <tt>true</tt> if the receiver contains the specified key.
     * 
     * @return <tt>true</tt> if the receiver contains the specified key.
     */
    public boolean containsKey(final long key) {
        return !forEachKey(new LongProcedure() {
            public boolean apply(long iterKey) {
                return (key != iterKey);
            }
        });
    }

    /**
     * Returns <tt>true</tt> if the receiver contains the specified value.
     * 
     * @return <tt>true</tt> if the receiver contains the specified value.
     */
    public boolean containsValue(final float value) {
        return !forEachPair(new LongFloatProcedure() {
            public boolean apply(long iterKey, float iterValue) {
                return (value != iterValue);
            }
        });
    }

    /**
     * Returns a deep copy of the receiver; uses <code>clone()</code> and casts
     * the result.
     * 
     * @return a deep copy of the receiver.
     */
    public AbstractLongFloatMap copy() {
        return (AbstractLongFloatMap) clone();
    }

    /**
     * Compares the specified object with this map for equality. Returns
     * <tt>true</tt> if the given object is also a map and the two maps
     * represent the same mappings. More formally, two maps <tt>m1</tt> and
     * <tt>m2</tt> represent the same mappings iff
     * 
     * <pre>
     * m1.forEachPair(
     *  new IntFloatProcedure() {
     *      public boolean apply(int key, float value) {
     *          return m2.containsKey(key) &amp;&amp; m2.get(key) == value;
     *      }
     *  }
     * )
     * &amp;&amp;
     * m2.forEachPair(
     *  new IntFloatProcedure() {
     *      public boolean apply(int key, float value) {
     *          return m1.containsKey(key) &amp;&amp; m1.get(key) == value;
     *      }
     *  }
     * );
     * </pre>
     * 
     * This implementation first checks if the specified object is this map; if
     * so it returns <tt>true</tt>. Then, it checks if the specified object is a
     * map whose size is identical to the size of this set; if not, it it
     * returns <tt>false</tt>. If so, it applies the iteration as described
     * above.
     * 
     * @param obj
     *            object to be compared for equality with this map.
     * @return <tt>true</tt> if the specified object is equal to this map.
     */

    public boolean equals(Object obj) {
        if (obj == this)
            return true;

        if (!(obj instanceof AbstractLongFloatMap))
            return false;
        final AbstractLongFloatMap other = (AbstractLongFloatMap) obj;
        if (other.size() != size())
            return false;

        return forEachPair(new LongFloatProcedure() {
            public boolean apply(long key, float value) {
                return other.containsKey(key) && other.get(key) == value;
            }
        }) && other.forEachPair(new LongFloatProcedure() {
            public boolean apply(long key, float value) {
                return containsKey(key) && get(key) == value;
            }
        });
    }

    /**
     * Applies a procedure to each key of the receiver, if any. Note: Iterates
     * over the keys in no particular order. Subclasses can define a particular
     * order, for example, "sorted by key". All methods which <i>can</i> be
     * expressed in terms of this method (most methods can) <i>must
     * guarantee</i> to use the <i>same</i> order defined by this method, even
     * if it is no particular order. This is necessary so that, for example,
     * methods <tt>keys</tt> and <tt>values</tt> will yield association pairs,
     * not two uncorrelated lists.
     * 
     * @param procedure
     *            the procedure to be applied. Stops iteration if the procedure
     *            returns <tt>false</tt>, otherwise continues.
     * @return <tt>false</tt> if the procedure stopped before all keys where
     *         iterated over, <tt>true</tt> otherwise.
     */
    public abstract boolean forEachKey(LongProcedure procedure);

    /**
     * Applies a procedure to each (key,value) pair of the receiver, if any.
     * Iteration order is guaranteed to be <i>identical</i> to the order used by
     * method {@link #forEachKey(LongProcedure)}.
     * 
     * @param procedure
     *            the procedure to be applied. Stops iteration if the procedure
     *            returns <tt>false</tt>, otherwise continues.
     * @return <tt>false</tt> if the procedure stopped before all keys where
     *         iterated over, <tt>true</tt> otherwise.
     */
    public boolean forEachPair(final LongFloatProcedure procedure) {
        return forEachKey(new LongProcedure() {
            public boolean apply(long key) {
                return procedure.apply(key, get(key));
            }
        });
    }

    /**
     * Returns the value associated with the specified key. It is often a good
     * idea to first check with {@link #containsKey(long)} whether the given key
     * has a value associated or not, i.e. whether there exists an association
     * for the given key or not.
     * 
     * @param key
     *            the key to be searched for.
     * @return the value associated with the specified key; <tt>0</tt> if no
     *         such key is present.
     */
    public abstract float get(long key);

    /**
     * Returns the first key the given value is associated with. It is often a
     * good idea to first check with {@link #containsValue(float)} whether there
     * exists an association from a key to this value. Search order is
     * guaranteed to be <i>identical</i> to the order used by method
     * {@link #forEachKey(LongProcedure)}.
     * 
     * @param value
     *            the value to search for.
     * @return the first key for which holds <tt>get(key) == value</tt>; returns
     *         <tt>Integer.MIN_VALUE</tt> if no such key exists.
     */
    public long keyOf(final float value) {
        final long[] foundKey = new long[1];
        boolean notFound = forEachPair(new LongFloatProcedure() {
            public boolean apply(long iterKey, float iterValue) {
                boolean found = value == iterValue;
                if (found)
                    foundKey[0] = iterKey;
                return !found;
            }
        });
        if (notFound)
            return Long.MIN_VALUE;
        return foundKey[0];
    }

    /**
     * Returns a list filled with all keys contained in the receiver. The
     * returned list has a size that equals <tt>this.size()</tt>. Iteration
     * order is guaranteed to be <i>identical</i> to the order used by method
     * {@link #forEachKey(LongProcedure)}.
     * <p>
     * This method can be used to iterate over the keys of the receiver.
     * 
     * @return the keys.
     */
    public LongArrayList keys() {
        LongArrayList list = new LongArrayList(size());
        keys(list);
        return list;
    }

    /**
     * Fills all keys contained in the receiver into the specified list. Fills
     * the list, starting at index 0. After this call returns the specified list
     * has a new size that equals <tt>this.size()</tt>. Iteration order is
     * guaranteed to be <i>identical</i> to the order used by method
     * {@link #forEachKey(LongProcedure)}.
     * <p>
     * This method can be used to iterate over the keys of the receiver.
     * 
     * @param list
     *            the list to be filled, can have any size.
     */
    public void keys(final LongArrayList list) {
        list.clear();
        forEachKey(new LongProcedure() {
            public boolean apply(long key) {
                list.add(key);
                return true;
            }
        });
    }

    /**
     * Fills all keys <i>sorted ascending by their associated value</i> into the
     * specified list. Fills into the list, starting at index 0. After this call
     * returns the specified list has a new size that equals
     * <tt>this.size()</tt>. Primary sort criterium is "value", secondary sort
     * criterium is "key". This means that if any two values are equal, the
     * smaller key comes first.
     * <p>
     * <b>Example:</b> <br>
     * <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (8,6,7)</tt>
     * 
     * @param keyList
     *            the list to be filled, can have any size.
     */
    public void keysSortedByValue(final LongArrayList keyList) {
        pairsSortedByValue(keyList, new FloatArrayList(size()));
    }

    /**
     * Fills all pairs satisfying a given condition into the specified lists.
     * Fills into the lists, starting at index 0. After this call returns the
     * specified lists both have a new size, the number of pairs satisfying the
     * condition. Iteration order is guaranteed to be <i>identical</i> to the
     * order used by method {@link #forEachKey(LongProcedure)}.
     * <p>
     * <b>Example:</b> <br>
     * 
     * <pre>
     *   LongFloatProcedure condition = new LongFloatProcedure() { // match even keys only
     *   public boolean apply(long key, float value) { return key%2==0; }
     *   }
     *   keys = (8,7,6), values = (1,2,2) --&gt; keyList = (6,8), valueList = (2,1)
     * &lt;/tt&gt;
     * </pre>
     * 
     * @param condition
     *            the condition to be matched. Takes the current key as first
     *            and the current value as second argument.
     * @param keyList
     *            the list to be filled with keys, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void pairsMatching(final LongFloatProcedure condition, final LongArrayList keyList,
            final FloatArrayList valueList) {
        keyList.clear();
        valueList.clear();

        forEachPair(new LongFloatProcedure() {
            public boolean apply(long key, float value) {
                if (condition.apply(key, value)) {
                    keyList.add(key);
                    valueList.add(value);
                }
                return true;
            }
        });
    }

    /**
     * Fills all keys and values <i>sorted ascending by key</i> into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists both have a new size that equals
     * <tt>this.size()</tt>.
     * <p>
     * <b>Example:</b> <br>
     * <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (6,7,8), valueList = (2,2,1)</tt>
     * 
     * @param keyList
     *            the list to be filled with keys, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void pairsSortedByKey(final LongArrayList keyList, final FloatArrayList valueList) {
        keys(keyList);
        keyList.sort();
        int size = keyList.size();
        valueList.setSize(size);
        final float[] valueListElements = valueList.elements();
        final long[] keyListElements = keyList.elements();
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, size);
            Future<?>[] futures = new Future[nthreads];
            int k = size / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstIdx = size - j * k;
                final int lastIdx = (j == (nthreads - 1)) ? 0 : firstIdx - k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        for (int i = firstIdx; --i >= lastIdx;) {
                            valueListElements[i] = get(keyListElements[i]);
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            for (int i = keyList.size(); --i >= 0;) {
                valueListElements[i] = get(keyListElements[i]);
            }
        }
    }

    /**
     * Fills all keys and values <i>sorted ascending by value</i> into the
     * specified lists. Fills into the lists, starting at index 0. After this
     * call returns the specified lists both have a new size that equals
     * <tt>this.size()</tt>. Primary sort criterium is "value", secondary sort
     * criterium is "key". This means that if any two values are equal, the
     * smaller key comes first.
     * <p>
     * <b>Example:</b> <br>
     * <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (8,6,7), valueList = (1,2,2)</tt>
     * 
     * @param keyList
     *            the list to be filled with keys, can have any size.
     * @param valueList
     *            the list to be filled with values, can have any size.
     */
    public void pairsSortedByValue(final LongArrayList keyList, final FloatArrayList valueList) {
        keys(keyList);
        values(valueList);

        final long[] k = keyList.elements();
        final float[] v = valueList.elements();
        cern.colt.Swapper swapper = new cern.colt.Swapper() {
            public void swap(int a, int b) {
                long t2;
                float t1;
                t1 = v[a];
                v[a] = v[b];
                v[b] = t1;
                t2 = k[a];
                k[a] = k[b];
                k[b] = t2;
            }
        };

        cern.colt.function.tint.IntComparator comp = new cern.colt.function.tint.IntComparator() {
            public int compare(int a, int b) {
                return v[a] < v[b] ? -1 : v[a] > v[b] ? 1 : (k[a] < k[b] ? -1 : (k[a] == k[b] ? 0 : 1));
            }
        };

        cern.colt.GenericSorting.quickSort(0, keyList.size(), comp, swapper);
    }

    /**
     * Associates the given key with the given value. Replaces any old
     * <tt>(key,someOtherValue)</tt> association, if existing.
     * 
     * @param key
     *            the key the value shall be associated with.
     * @param value
     *            the value to be associated.
     * @return <tt>true</tt> if the receiver did not already contain such a key;
     *         <tt>false</tt> if the receiver did already contain such a key -
     *         the new value has now replaced the formerly associated value.
     */
    public abstract boolean put(long key, float value);

    /**
     * Removes the given key with its associated element from the receiver, if
     * present.
     * 
     * @param key
     *            the key to be removed from the receiver.
     * @return <tt>true</tt> if the receiver contained the specified key,
     *         <tt>false</tt> otherwise.
     */
    public abstract boolean removeKey(long key);

    /**
     * Returns a string representation of the receiver, containing the String
     * representation of each key-value pair, sorted ascending by key.
     */

    public String toString() {
        LongArrayList theKeys = keys();
        String tmp = theKeys.toString() + "\n";
        theKeys.sort();

        StringBuffer buf = new StringBuffer(tmp);
        // StringBuffer buf = new StringBuffer();
        buf.append("[");
        int maxIndex = theKeys.size() - 1;
        for (int i = 0; i <= maxIndex; i++) {
            long key = theKeys.get(i);
            buf.append(String.valueOf(key));
            buf.append("->");
            buf.append(String.valueOf(get(key)));
            if (i < maxIndex)
                buf.append(", ");
        }
        buf.append("]");
        return buf.toString();
    }

    /**
     * Returns a string representation of the receiver, containing the String
     * representation of each key-value pair, sorted ascending by value.
     */
    public String toStringByValue() {
        LongArrayList theKeys = new LongArrayList();
        keysSortedByValue(theKeys);

        StringBuffer buf = new StringBuffer();
        buf.append("[");
        int maxIndex = theKeys.size() - 1;
        for (int i = 0; i <= maxIndex; i++) {
            long key = theKeys.get(i);
            buf.append(String.valueOf(key));
            buf.append("->");
            buf.append(String.valueOf(get(key)));
            if (i < maxIndex)
                buf.append(", ");
        }
        buf.append("]");
        return buf.toString();
    }

    /**
     * Returns a list filled with all values contained in the receiver. The
     * returned list has a size that equals <tt>this.size()</tt>. Iteration
     * order is guaranteed to be <i>identical</i> to the order used by method
     * {@link #forEachKey(LongProcedure)}.
     * <p>
     * This method can be used to iterate over the values of the receiver.
     * 
     * @return the values.
     */
    public FloatArrayList values() {
        FloatArrayList list = new FloatArrayList(size());
        values(list);
        return list;
    }

    /**
     * Fills all values contained in the receiver into the specified list. Fills
     * the list, starting at index 0. After this call returns the specified list
     * has a new size that equals <tt>this.size()</tt>. Iteration order is
     * guaranteed to be <i>identical</i> to the order used by method
     * {@link #forEachKey(LongProcedure)}.
     * <p>
     * This method can be used to iterate over the values of the receiver.
     * 
     * @param list
     *            the list to be filled, can have any size.
     */
    public void values(final FloatArrayList list) {
        list.clear();
        forEachKey(new LongProcedure() {
            public boolean apply(long key) {
                list.add(get(key));
                return true;
            }
        });
    }
}
