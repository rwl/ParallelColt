/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.math.tlong;

import cern.colt.function.tlong.LongFunction;
import cern.colt.function.tlong.LongLongFunction;
import cern.colt.function.tlong.LongLongProcedure;
import cern.colt.function.tlong.LongProcedure;
import cern.jet.math.tdouble.DoubleArithmetic;
import cern.jet.math.tdouble.DoubleFunctions;

/**
 * Long Function objects to be passed to generic methods. Same as
 * {@link DoubleFunctions} except operating on longs.
 * <p>
 * For aliasing see {@link #longFunctions}.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
public class LongFunctions extends Object {
    /**
     * Little trick to allow for "aliasing", that is, renaming this class.
     * Writing code like
     * <p>
     * <tt>LongFunctions.chain(LongFunctions.plus,LongFunctions.mult(3),LongFunctions.chain(LongFunctions.square,LongFunctions.div(2)));</tt>
     * <p>
     * is a bit awkward, to say the least. Using the aliasing you can instead
     * write
     * <p>
     * <tt>LongFunctions F = LongFunctions.longFunctions; <br>
    F.chain(F.plus,F.mult(3),F.chain(F.square,F.div(2)));</tt>
     * <p>
     */
    public static final LongFunctions longFunctions = new LongFunctions();

    /***************************************************************************
     * <H3>Unary functions</H3>
     **************************************************************************/
    /**
     * Function that returns <tt>Math.abs(a) == (a < 0) ? -a : a</tt>.
     */
    public static final LongFunction abs = new LongFunction() {
        public final long apply(long a) {
            return (a < 0) ? -a : a;
        }
    };

    /**
     * Function that returns <tt>a--</tt>.
     */
    public static final LongFunction dec = new LongFunction() {
        public final long apply(long a) {
            return a--;
        }
    };

    /**
     * Function that returns <tt>(long) Arithmetic.factorial(a)</tt>.
     */
    public static final LongFunction factorial = new LongFunction() {
        public final long apply(long a) {
            return (long) DoubleArithmetic.factorial(a);
        }
    };

    /**
     * Function that returns its argument.
     */
    public static final LongFunction identity = new LongFunction() {
        public final long apply(long a) {
            return a;
        }
    };

    /**
     * Function that returns <tt>a++</tt>.
     */
    public static final LongFunction inc = new LongFunction() {
        public final long apply(long a) {
            return a++;
        }
    };

    /**
     * Function that returns <tt>-a</tt>.
     */
    public static final LongFunction neg = new LongFunction() {
        public final long apply(long a) {
            return -a;
        }
    };

    /**
     * Function that returns <tt>~a</tt>.
     */
    public static final LongFunction not = new LongFunction() {
        public final long apply(long a) {
            return ~a;
        }
    };

    /**
     * Function that returns <tt>a < 0 ? -1 : a > 0 ? 1 : 0</tt>.
     */
    public static final LongFunction sign = new LongFunction() {
        public final long apply(long a) {
            return a < 0 ? -1 : a > 0 ? 1 : 0;
        }
    };

    /**
     * Function that returns <tt>a * a</tt>.
     */
    public static final LongFunction square = new LongFunction() {
        public final long apply(long a) {
            return a * a;
        }
    };

    /***************************************************************************
     * <H3>Binary functions</H3>
     **************************************************************************/

    /**
     * Function that returns <tt>a & b</tt>.
     */
    public static final LongLongFunction and = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a & b;
        }
    };

    /**
     * Function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>.
     */
    public static final LongLongFunction compare = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a < b ? -1 : a > b ? 1 : 0;
        }
    };

    /**
     * Function that returns <tt>a / b</tt>.
     */
    public static final LongLongFunction div = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a / b;
        }
    };

    /**
     * Function that returns <tt>-(a / b)</tt>.
     */
    public static final LongLongFunction divNeg = new LongLongFunction() {
        public final long apply(long a, long b) {
            return -(a / b);
        }
    };

    /**
     * Function that returns <tt>a == b ? 1 : 0</tt>.
     */
    public static final LongLongFunction equals = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a == b ? 1 : 0;
        }
    };

    /**
     * Function that returns <tt>a == b</tt>.
     */
    public static final LongLongProcedure isEqual = new LongLongProcedure() {
        public final boolean apply(long a, long b) {
            return a == b;
        }
    };

    /**
     * Function that returns <tt>a < b</tt>.
     */
    public static final LongLongProcedure isLess = new LongLongProcedure() {
        public final boolean apply(long a, long b) {
            return a < b;
        }
    };

    /**
     * Function that returns <tt>a > b</tt>.
     */
    public static final LongLongProcedure isGreater = new LongLongProcedure() {
        public final boolean apply(long a, long b) {
            return a > b;
        }
    };

    /**
     * Function that returns <tt>Math.max(a,b)</tt>.
     */
    public static final LongLongFunction max = new LongLongFunction() {
        public final long apply(long a, long b) {
            return (a >= b) ? a : b;
        }
    };

    /**
     * Function that returns <tt>Math.min(a,b)</tt>.
     */
    public static final LongLongFunction min = new LongLongFunction() {
        public final long apply(long a, long b) {
            return (a <= b) ? a : b;
        }
    };

    /**
     * Function that returns <tt>a - b</tt>.
     */
    public static final LongLongFunction minus = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a - b;
        }
    };

    /**
     * Function that returns <tt>a % b</tt>.
     */
    public static final LongLongFunction mod = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a % b;
        }
    };

    /**
     * Function that returns <tt>a * b</tt>.
     */
    public static final LongLongFunction mult = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a * b;
        }
    };

    /**
     * Function that returns <tt>-(a * b)</tt>.
     */
    public static final LongLongFunction multNeg = new LongLongFunction() {
        public final long apply(long a, long b) {
            return -(a * b);
        }
    };

    /**
     * Function that returns <tt>a * b^2</tt>.
     */
    public static final LongLongFunction multSquare = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a * b * b;
        }
    };

    /**
     * Function that returns <tt>a | b</tt>.
     */
    public static final LongLongFunction or = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a | b;
        }
    };

    /**
     * Function that returns <tt>a + b</tt>.
     */
    public static final LongLongFunction plus = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a + b;
        }
    };

    /**
     * Function that returns <tt>Math.abs(a) + Math.abs(b)</tt>.
     */
    public static final LongLongFunction plusAbs = new LongLongFunction() {
        public final long apply(long a, long b) {
            return Math.abs(a) + Math.abs(b);
        }
    };

    /**
     * Function that returns <tt>(long) Math.pow(a,b)</tt>.
     */
    public static final LongLongFunction pow = new LongLongFunction() {
        public final long apply(long a, long b) {
            return (long) Math.pow(a, b);
        }
    };

    /**
     * Function that returns <tt>a << b</tt>.
     */
    public static final LongLongFunction shiftLeft = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a << b;
        }
    };

    /**
     * Function that returns <tt>a >> b</tt>.
     */
    public static final LongLongFunction shiftRightSigned = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a >> b;
        }
    };

    /**
     * Function that returns <tt>a >>> b</tt>.
     */
    public static final LongLongFunction shiftRightUnsigned = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a >>> b;
        }
    };

    /**
     * Function that returns <tt>a ^ b</tt>.
     */
    public static final LongLongFunction xor = new LongLongFunction() {
        public final long apply(long a, long b) {
            return a ^ b;
        }
    };

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected LongFunctions() {
    }

    /**
     * Constructs a function that returns <tt>a & b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction and(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a & b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>(from<=a && a<=to) ? 1 : 0</tt>.
     * <tt>a</tt> is a variable, <tt>from</tt> and <tt>to</tt> are fixed.
     */
    public static LongFunction between(final long from, final long to) {
        return new LongFunction() {
            public final long apply(long a) {
                return (from <= a && a <= to) ? 1 : 0;
            }
        };
    }

    /**
     * Constructs a unary function from a binary function with the first operand
     * (argument) fixed to the given constant <tt>c</tt>. The second operand is
     * variable (free).
     * 
     * @param function
     *            a binary function taking operands in the form
     *            <tt>function.apply(c,var)</tt>.
     * @return the unary function <tt>function(c,var)</tt>.
     */
    public static LongFunction bindArg1(final LongLongFunction function, final long c) {
        return new LongFunction() {
            public final long apply(long var) {
                return function.apply(c, var);
            }
        };
    }

    /**
     * Constructs a unary function from a binary function with the second
     * operand (argument) fixed to the given constant <tt>c</tt>. The first
     * operand is variable (free).
     * 
     * @param function
     *            a binary function taking operands in the form
     *            <tt>function.apply(var,c)</tt>.
     * @return the unary function <tt>function(var,c)</tt>.
     */
    public static LongFunction bindArg2(final LongLongFunction function, final long c) {
        return new LongFunction() {
            public final long apply(long var) {
                return function.apply(var, c);
            }
        };
    }

    /**
     * Constructs the function <tt>g( h(a) )</tt>.
     * 
     * @param g
     *            a unary function.
     * @param h
     *            a unary function.
     * @return the unary function <tt>g( h(a) )</tt>.
     */
    public static LongFunction chain(final LongFunction g, final LongFunction h) {
        return new LongFunction() {
            public final long apply(long a) {
                return g.apply(h.apply(a));
            }
        };
    }

    /**
     * Constructs the function <tt>g( h(a,b) )</tt>.
     * 
     * @param g
     *            a unary function.
     * @param h
     *            a binary function.
     * @return the unary function <tt>g( h(a,b) )</tt>.
     */
    public static LongLongFunction chain(final LongFunction g, final LongLongFunction h) {
        return new LongLongFunction() {
            public final long apply(long a, long b) {
                return g.apply(h.apply(a, b));
            }
        };
    }

    /**
     * Constructs the function <tt>f( g(a), h(b) )</tt>.
     * 
     * @param f
     *            a binary function.
     * @param g
     *            a unary function.
     * @param h
     *            a unary function.
     * @return the binary function <tt>f( g(a), h(b) )</tt>.
     */
    public static LongLongFunction chain(final LongLongFunction f, final LongFunction g, final LongFunction h) {
        return new LongLongFunction() {
            public final long apply(long a, long b) {
                return f.apply(g.apply(a), h.apply(b));
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>.
     * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
     */
    public static LongFunction compare(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a < b ? -1 : a > b ? 1 : 0;
            }
        };
    }

    /**
     * Constructs a function that returns the constant <tt>c</tt>.
     */
    public static LongFunction constant(final long c) {
        return new LongFunction() {
            public final long apply(long a) {
                return c;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a / b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction div(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a / b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a == b ? 1 : 0</tt>. <tt>a</tt> is
     * a variable, <tt>b</tt> is fixed.
     */
    public static LongFunction equals(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a == b ? 1 : 0;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>from<=a && a<=to</tt>. <tt>a</tt>
     * is a variable, <tt>from</tt> and <tt>to</tt> are fixed.
     */
    public static LongProcedure isBetween(final long from, final long to) {
        return new LongProcedure() {
            public final boolean apply(long a) {
                return from <= a && a <= to;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a == b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongProcedure isEqual(final long b) {
        return new LongProcedure() {
            public final boolean apply(long a) {
                return a == b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a > b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongProcedure isGreater(final long b) {
        return new LongProcedure() {
            public final boolean apply(long a) {
                return a > b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a < b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongProcedure isLess(final long b) {
        return new LongProcedure() {
            public final boolean apply(long a) {
                return a < b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>Math.max(a,b)</tt>. <tt>a</tt> is
     * a variable, <tt>b</tt> is fixed.
     */
    public static LongFunction max(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return (a >= b) ? a : b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>Math.min(a,b)</tt>. <tt>a</tt> is
     * a variable, <tt>b</tt> is fixed.
     */
    public static LongFunction min(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return (a <= b) ? a : b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a - b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction minus(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a - b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a - b*constant</tt>. <tt>a</tt>
     * and <tt>b</tt> are variables, <tt>constant</tt> is fixed.
     */
    public static LongLongFunction minusMult(final long constant) {
        return plusMultSecond(-constant);
    }

    /**
     * Constructs a function that returns <tt>a % b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction mod(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a % b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a * b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction mult(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a * b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a | b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction or(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a | b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a + b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction plus(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a + b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>b*constant</tt>.
     */
    public static LongLongFunction multSecond(final long constant) {

        return new LongLongFunction() {
            public final long apply(long a, long b) {
                return b * constant;
            }
        };

    }

    /**
     * Constructs a function that returns <tt>(long) Math.pow(a,b)</tt>.
     * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
     */
    public static LongFunction pow(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return (long) Math.pow(a, b);
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a + b*constant</tt>. <tt>a</tt>
     * and <tt>b</tt> are variables, <tt>constant</tt> is fixed.
     */
    public static LongLongFunction plusMultSecond(final long constant) {
        return new LongPlusMultSecond(constant);
    }

    /**
     * Constructs a function that returns <tt>a * constant + b</tt>. <tt>a</tt>
     * and <tt>b</tt> are variables, <tt>constant</tt> is fixed.
     */
    public static LongLongFunction plusMultFirst(final long constant) {
        return new LongPlusMultFirst(constant);
    }

    /**
     * Constructs a function that returns a 32 bit uniformly distributed random
     * number in the closed longerval <tt>[Long.MIN_VALUE,Long.MAX_VALUE]</tt>
     * (including <tt>Long.MIN_VALUE</tt> and <tt>Long.MAX_VALUE</tt>).
     * Currently the engine is
     * {@link cern.jet.random.tdouble.engine.DoubleMersenneTwister} and is
     * seeded with the current time.
     * <p>
     * Note that any random engine derived from
     * {@link cern.jet.random.tdouble.engine.DoubleRandomEngine} and any random
     * distribution derived from
     * {@link cern.jet.random.tdouble.AbstractDoubleDistribution} are function
     * objects, because they implement the proper longerfaces. Thus, if you are
     * not happy with the default, just pass your favourite random generator to
     * function evaluating methods.
     */
    public static LongFunction random() {
        return new cern.jet.random.tdouble.engine.DoubleMersenneTwister(new java.util.Date());
    }

    /**
     * Constructs a function that returns <tt>a << b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction shiftLeft(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a << b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a >> b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction shiftRightSigned(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a >> b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a >>> b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction shiftRightUnsigned(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a >>> b;
            }
        };
    }

    /**
     * Constructs a function that returns <tt>function.apply(b,a)</tt>, i.e.
     * applies the function with the first operand as second operand and the
     * second operand as first operand.
     * 
     * @param function
     *            a function taking operands in the form
     *            <tt>function.apply(a,b)</tt>.
     * @return the binary function <tt>function(b,a)</tt>.
     */
    public static LongLongFunction swapArgs(final LongLongFunction function) {
        return new LongLongFunction() {
            public final long apply(long a, long b) {
                return function.apply(b, a);
            }
        };
    }

    /**
     * Constructs a function that returns <tt>a | b</tt>. <tt>a</tt> is a
     * variable, <tt>b</tt> is fixed.
     */
    public static LongFunction xor(final long b) {
        return new LongFunction() {
            public final long apply(long a) {
                return a ^ b;
            }
        };
    }
}
