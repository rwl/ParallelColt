/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.random.tfloat;

import cern.jet.random.tfloat.engine.FloatRandomEngine;

/**
 * Uniform distribution; <A HREF=
 * "http://www.cern.ch/RD11/rkb/AN16pp/node292.html#SECTION0002920000000000000000"
 * > Math definition</A> and <A
 * HREF="http://www.statsoft.com/textbook/glosu.html#Uniform Distribution">
 * animated definition</A>.
 * <p>
 * Instance methods operate on a user supplied uniform random number generator;
 * they are unsynchronized.
 * <dt>Static methods operate on a default uniform random number generator; they
 * are synchronized.
 * <p>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
public class FloatUniform extends AbstractContinousFloatDistribution {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    protected float min;

    protected float max;

    // The uniform random number generated shared by all <b>static</b> methods.
    protected static FloatUniform shared = new FloatUniform(makeDefaultGenerator());

    /**
     * Constructs a uniform distribution with the given minimum and maximum,
     * using a {@link cern.jet.random.tfloat.engine.FloatMersenneTwister} seeded
     * with the given seed.
     */
    public FloatUniform(float min, float max, int seed) {
        this(min, max, new cern.jet.random.tfloat.engine.FloatMersenneTwister(seed));
    }

    /**
     * Constructs a uniform distribution with the given minimum and maximum.
     */
    public FloatUniform(float min, float max, FloatRandomEngine randomGenerator) {
        setRandomGenerator(randomGenerator);
        setState(min, max);
    }

    /**
     * Constructs a uniform distribution with <tt>min=0.0</tt> and
     * <tt>max=1.0</tt>.
     */
    public FloatUniform(FloatRandomEngine randomGenerator) {
        this(0, 1, randomGenerator);
    }

    /**
     * Returns the cumulative distribution function (assuming a continous
     * uniform distribution).
     */
    public float cdf(float x) {
        if (x <= min)
            return 0.0f;
        if (x >= max)
            return 1.0f;
        return (x - min) / (max - min);
    }

    /**
     * Returns a uniformly distributed random <tt>boolean</tt>.
     */
    public boolean nextBoolean() {
        return randomGenerator.raw() > 0.5;
    }

    /**
     * Returns a uniformly distributed random number in the open interval
     * <tt>(min,max)</tt> (excluding <tt>min</tt> and <tt>max</tt>).
     */

    public float nextFloat() {
        return min + (max - min) * randomGenerator.raw();
    }

    /**
     * Returns a uniformly distributed random number in the open interval
     * <tt>(from,to)</tt> (excluding <tt>from</tt> and <tt>to</tt>). Pre
     * conditions: <tt>from &lt;= to</tt>.
     */
    public float nextFloatFromTo(float from, float to) {
        return from + (to - from) * randomGenerator.raw();
    }

    /**
     * Returns a uniformly distributed random number in the closed interval
     * <tt>[min,max]</tt> (including <tt>min</tt> and <tt>max</tt>).
     */

    public int nextInt() {
        return nextIntFromTo(Math.round(min), Math.round(max));
    }

    /**
     * Returns a uniformly distributed random number in the closed interval
     * <tt>[from,to]</tt> (including <tt>from</tt> and <tt>to</tt>). Pre
     * conditions: <tt>from &lt;= to</tt>.
     */
    public int nextIntFromTo(int from, int to) {
        return (int) (from + (long) ((1L + to - from) * randomGenerator.raw()));
    }

    /**
     * Returns a uniformly distributed random number in the closed interval
     * <tt>[from,to]</tt> (including <tt>from</tt> and <tt>to</tt>). Pre
     * conditions: <tt>from &lt;= to</tt>.
     */
    public long nextLongFromTo(long from, long to) {
        /*
         * Doing the thing turns out to be more tricky than expected. avoids
         * overflows and underflows. treats cases like from=-1, to=1 and the
         * like right. the following code would NOT solve the problem: return
         * (long) (Floats.randomFromTo(from,to));
         * 
         * rounding avoids the unsymmetric behaviour of casts from float to
         * long: (long) -0.7 = 0, (long) 0.7 = 0. checking for overflows and
         * underflows is also necessary.
         */

        // first the most likely and also the fastest case.
        if (from >= 0 && to < Long.MAX_VALUE) {
            return from + (long) (nextFloatFromTo(0.0f, to - from + 1));
        }

        // would we get a numeric overflow?
        // if not, we can still handle the case rather efficient.
        float diff = ((float) to) - (float) from + 1.0f;
        if (diff <= Long.MAX_VALUE) {
            return from + (long) (nextFloatFromTo(0.0f, diff));
        }

        // now the pathologic boundary cases.
        // they are handled rather slow.
        long random;
        if (from == Long.MIN_VALUE) {
            if (to == Long.MAX_VALUE) {
                // return Math.round(nextFloatFromTo(from,to));
                int i1 = nextIntFromTo(Integer.MIN_VALUE, Integer.MAX_VALUE);
                int i2 = nextIntFromTo(Integer.MIN_VALUE, Integer.MAX_VALUE);
                return ((i1 & 0xFFFFFFFFL) << 32) | (i2 & 0xFFFFFFFFL);
            }
            random = Math.round(nextFloatFromTo(from, to + 1));
            if (random > to)
                random = from;
        } else {
            random = Math.round(nextFloatFromTo(from - 1, to));
            if (random < from)
                random = to;
        }
        return random;
    }

    /**
     * Returns the probability distribution function (assuming a continous
     * uniform distribution).
     */
    public float pdf(float x) {
        if (x <= min || x >= max)
            return 0.0f;
        return (float) (1.0 / (max - min));
    }

    /**
     * Sets the internal state.
     */
    public void setState(float min, float max) {
        if (max < min) {
            setState(max, min);
            return;
        }
        this.min = min;
        this.max = max;
    }

    /**
     * Returns a uniformly distributed random <tt>boolean</tt>.
     */
    public static boolean staticNextBoolean() {
        synchronized (shared) {
            return shared.nextBoolean();
        }
    }

    /**
     * Returns a uniformly distributed random number in the open interval
     * <tt>(0,1)</tt> (excluding <tt>0</tt> and <tt>1</tt>).
     */
    public static float staticNextFloat() {
        synchronized (shared) {
            return shared.nextFloat();
        }
    }

    /**
     * Returns a uniformly distributed random number in the open interval
     * <tt>(from,to)</tt> (excluding <tt>from</tt> and <tt>to</tt>). Pre
     * conditions: <tt>from &lt;= to</tt>.
     */
    public static float staticNextFloatFromTo(float from, float to) {
        synchronized (shared) {
            return shared.nextFloatFromTo(from, to);
        }
    }

    /**
     * Returns a uniformly distributed random number in the closed interval
     * <tt>[from,to]</tt> (including <tt>from</tt> and <tt>to</tt>). Pre
     * conditions: <tt>from &lt;= to</tt>.
     */
    public static int staticNextIntFromTo(int from, int to) {
        synchronized (shared) {
            return shared.nextIntFromTo(from, to);
        }
    }

    /**
     * Returns a uniformly distributed random number in the closed interval
     * <tt>[from,to]</tt> (including <tt>from</tt> and <tt>to</tt>). Pre
     * conditions: <tt>from &lt;= to</tt>.
     */
    public static long staticNextLongFromTo(long from, long to) {
        synchronized (shared) {
            return shared.nextLongFromTo(from, to);
        }
    }

    /**
     * Sets the uniform random number generation engine shared by all
     * <b>static</b> methods.
     * 
     * @param randomGenerator
     *            the new uniform random number generation engine to be shared.
     */
    public static void staticSetRandomEngine(FloatRandomEngine randomGenerator) {
        synchronized (shared) {
            shared.setRandomGenerator(randomGenerator);
        }
    }

    /**
     * Returns a String representation of the receiver.
     */

    public String toString() {
        return this.getClass().getName() + "(" + min + "," + max + ")";
    }
}
