package hep.aida.tfloat.bin;

/**
 * Function objects computing dynamic bin aggregations; to be passed to generic
 * methods.
 * 
 * @see cern.colt.matrix.tfloat.algo.FloatFormatter
 * @see cern.colt.matrix.tfloat.algo.FloatStatistic
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
public class FloatBinFunctions1D extends Object {
    /**
     * Little trick to allow for "aliasing", that is, renaming this class. Using
     * the aliasing you can instead write
     * <p>
     * <tt>BinFunctions F = BinFunctions.functions; <br>
    someAlgo(F.max);</tt>
     */
    public static final FloatBinFunctions1D functions = new FloatBinFunctions1D();

    /**
     * Function that returns <tt>bin.max()</tt>.
     */
    public static final FloatBinFunction1D max = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.max();
        }

        public final String name() {
            return "Max";
        }
    };

    /**
     * Function that returns <tt>bin.mean()</tt>.
     */
    public static final FloatBinFunction1D mean = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.mean();
        }

        public final String name() {
            return "Mean";
        }
    };

    /**
     * Function that returns <tt>bin.median()</tt>.
     */
    public static final FloatBinFunction1D median = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.median();
        }

        public final String name() {
            return "Median";
        }
    };

    /**
     * Function that returns <tt>bin.min()</tt>.
     */
    public static final FloatBinFunction1D min = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.min();
        }

        public final String name() {
            return "Min";
        }
    };

    /**
     * Function that returns <tt>bin.rms()</tt>.
     */
    public static final FloatBinFunction1D rms = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.rms();
        }

        public final String name() {
            return "RMS";
        }
    };

    /**
     * Function that returns <tt>bin.size()</tt>.
     */
    public static final FloatBinFunction1D size = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.size();
        }

        public final String name() {
            return "Size";
        }
    };

    /**
     * Function that returns <tt>bin.standardDeviation()</tt>.
     */
    public static final FloatBinFunction1D stdDev = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.standardDeviation();
        }

        public final String name() {
            return "StdDev";
        }
    };

    /**
     * Function that returns <tt>bin.sum()</tt>.
     */
    public static final FloatBinFunction1D sum = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.sum();
        }

        public final String name() {
            return "Sum";
        }
    };

    /**
     * Function that returns <tt>bin.sumOfLogarithms()</tt>.
     */
    public static final FloatBinFunction1D sumLog = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.sumOfLogarithms();
        }

        public final String name() {
            return "SumLog";
        }
    };

    /**
     * Function that returns <tt>bin.geometricMean()</tt>.
     */
    public static final FloatBinFunction1D geometricMean = new FloatBinFunction1D() {
        public final float apply(DynamicFloatBin1D bin) {
            return bin.geometricMean();
        }

        public final String name() {
            return "GeomMean";
        }
    };

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected FloatBinFunctions1D() {
    }

    /**
     * Function that returns <tt>bin.quantile(percentage)</tt>.
     * 
     * @param percentage
     *            the percentage of the quantile (<tt>0 <= percentage <= 1</tt>
     *            ).
     */
    public static FloatBinFunction1D quantile(final float percentage) {
        return new FloatBinFunction1D() {
            public final float apply(DynamicFloatBin1D bin) {
                return bin.quantile(percentage);
            }

            public final String name() {
                return new cern.colt.matrix.FormerFactory().create("%1.2G").form(percentage * 100) + "% Q.";
            }
        };
    }
}
