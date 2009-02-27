/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library
 * www.jcuda.de
 * 
 * DISCLAIMER: THIS SOFTWARE IS PROVIDED WITHOUT WARRANTY OF ANY KIND
 * If you find any bugs or errors, please contact me: javagl@javagl.de 
 * 
 * LICENSE: THIS SOFTWARE IS FREE FOR NON-COMMERCIAL USE ONLY
 * For non-commercial applications, you may use this software without
 * any restrictions. If you wish to use it for commercial purposes, 
 * please contact me: javagl@javagl.de 
 *
 * Comments are taken from the the header files for the CUBLAS library
 */

package jcuda.jcublas;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.security.CodeSource;

/**
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library<br />
 * www.jcuda.de<br />
 * <br />
 * <br />
 * This file comment is partially taken from the cublas.h header file:<br />
 * <br />
 * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines) on top
 * of the CUDA driver. It allows access to the computational resources of NVIDIA
 * GPUs. The library is self-contained at the API level, i.e. no direct
 * interaction with the CUDA driver is necessary.<br />
 * <br />
 * The basic model by which applications use the CUBLAS library is to create
 * matrix and vector object in GPU memory space, fill them with data, then call
 * a sequence of BLAS functions, and finally upload the results from GPU memory
 * space back to the host. To accomplish this, CUBLAS provides helper functions
 * for creating and destroying objects in GPU space, and to write data to, and
 * retrieve data from, these objects.<br />
 * <br />
 * Since the BLAS core functions (as opposed to the helper functions) do not
 * return error status directly (for reasons of compatibility with existing BLAS
 * libraries) CUBLAS provides a separate function to retrieve the last error
 * that was recorded, to aid in debugging.<br />
 * <br />
 * Currently, only a subset of the BLAS core functions is implemented.<br />
 * <br />
 */
public class JCublas {
    /**
     * Enumeration of common CPU architectures.
     */
    public enum ARCHType {
        PPC, PPC_64, SPARC, UNKNOWN, X86, X86_64
    };

    /**
     * Enumeration of common operating systems, independent of version or
     * architecture.
     */
    public enum OSType {
        APPLE, LINUX, SUN, UNKNOWN, WINDOWS
    };

    /* CUBLAS status returns */

    /** Operation completed successfully */
    public static final int CUBLAS_STATUS_SUCCESS = 0x00000000;

    /** Library not initialized */
    public static final int CUBLAS_STATUS_NOT_INITIALIZED = 0x00000001;

    /** Resource allocation failed */
    public static final int CUBLAS_STATUS_ALLOC_FAILED = 0x00000003;

    /** Unsupported numerical value was passed to function */
    public static final int CUBLAS_STATUS_INVALID_VALUE = 0x00000007;

    /**
     * function requires an architectural feature absent from the architecture
     * of the device
     */
    public static final int CUBLAS_STATUS_ARCH_MISMATCH = 0x00000008;

    /** Access to GPU memory space failed */
    public static final int CUBLAS_STATUS_MAPPING_ERROR = 0x0000000B;

    /** GPU program failed to execute */
    public static final int CUBLAS_STATUS_EXECUTION_FAILED = 0x0000000D;

    /** An internal CUBLAS operation failed */
    public static final int CUBLAS_STATUS_INTERNAL_ERROR = 0x0000000E;

    /* JCUBLAS status returns */

    /** Device memory with the specified name was already allocated */
    public static final int JCUBLAS_STATUS_MEMORY_ALREADY_USED = 0x10000001;

    /** Device memory with the specified name could not be found */
    public static final int JCUBLAS_STATUS_MEMORY_NOT_FOUND = 0x10000002;

    /** An internal JCublas operation failed */
    public static final int JCUBLAS_STATUS_INTERNAL_ERROR = 0x10000003;

    /**
     * The log levels which may be used to control the internal logging of the
     * JCublas library
     */
    public enum LogLevel {
        LOG_QUIET, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG, LOG_TRACE, LOG_DEBUGTRACE
    }

    /**
     * The flag which indicates whether a call to cublasInit should initialize
     * JCublas in emulation mode
     */
    private static boolean defaultEmulation = false;

    /* Private constructor to prevent instantiation */
    private JCublas() {
    }

    /**
     * Set the flag which indicates whether a call to cublasInit should
     * initialize JCublas in emulation mode
     * 
     * @param emulation
     *            Whether emulation mode should be used
     */
    public static void setEmulation(boolean emulation) {
        defaultEmulation = emulation;
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * @return The status result of cublasInit
     * @throws IOException
     * @throws URISyntaxException
     * @throws UnsatisfiedLinkError
     */
    public static int cublasInit() throws Throwable {
        return cublasInit(defaultEmulation);
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * The emulation flag indicates whether the emulation mode of CUBLAS should
     * be used. This will cause the appropriate library to be used, so the first
     * call to this method determines whether the emulation mode is used, and
     * the mode can not be changed with subsequent calls to this method.<br />
     * <br />
     * The emulation mode is MUCH slower than the real, hardware-accelerated
     * CUBLAS, but also works when no CUDA driver is installed and no CUDA
     * hardware is available<br />
     * <br />
     * 
     * @param emulation
     *            Indicates whether emulation mode should be used
     * @return The status result of cublasInit
     * @throws Throwable
     */
    public static int cublasInit(boolean emulation) throws Throwable {
        try {
            if (emulation) {
                System.loadLibrary("JCublasEmu");
            } else {
                System.loadLibrary("JCublas");
            }
        } catch (Throwable e) {
            CodeSource sc = jcuda.jcublas.JCublas.class.getProtectionDomain().getCodeSource();
            if (sc == null) {
                throw new NullPointerException("sc == null");
            } else {
                File jarFile = new File(sc.getLocation().toURI());
                if (emulation) {
                    System.load(jarFile.getParentFile().getCanonicalPath().toString() + System.getProperty("file.separator") + "lib" + System.getProperty("file.separator") + getNativeLibraryName("JCublasEmu"));
                } else {
                    System.load(jarFile.getParentFile().getCanonicalPath().toString() + System.getProperty("file.separator") + "lib" + System.getProperty("file.separator") + getNativeLibraryName("JCublas"));
                }
            }
        }
        return cublasInitNative();
    }

    private static OSType calculateOS() {
        String osName = System.getProperty("os.name").toLowerCase();
        assert osName != null;
        if (osName.startsWith("mac os x")) {
            return OSType.APPLE;
        }
        if (osName.startsWith("windows")) {
            return OSType.WINDOWS;
        }
        if (osName.startsWith("linux")) {
            return OSType.LINUX;
        }
        if (osName.startsWith("sun")) {
            return OSType.SUN;
        }
        return OSType.UNKNOWN;
    }

    private static ARCHType calculateArch() {
        String osArch = System.getProperty("os.arch").toLowerCase();
        assert osArch != null;
        if (osArch.equals("x86") || osArch.equals("i386")) {
            return ARCHType.X86;
        }
        if (osArch.startsWith("amd64") || osArch.startsWith("x86_64")) {
            return ARCHType.X86_64;
        }
        if (osArch.equals("ppc")) {
            return ARCHType.PPC;
        }
        if (osArch.startsWith("ppc")) {
            return ARCHType.PPC_64;
        }
        if (osArch.startsWith("sparc")) {
            return ARCHType.SPARC;
        }
        return ARCHType.UNKNOWN;
    }

    private static String getNativeLibraryName(String name) {
        OSType os = calculateOS();
        String nativeName = null;
        switch (os) {
        case APPLE:
            nativeName = name.toLowerCase() + "-" + os.toString().toLowerCase() + "-" + calculateArch().toString().toLowerCase() + ".jnilib";
            break;
        case LINUX:
            nativeName = name.toLowerCase() + "-" + os.toString().toLowerCase() + "-" + calculateArch().toString().toLowerCase() + ".so";
            break;
        case SUN:
            nativeName = name.toLowerCase() + "-" + os.toString().toLowerCase() + "-" + calculateArch().toString().toLowerCase() + ".so";
            break;
        case WINDOWS:
            nativeName = name.toLowerCase() + "-" + os.toString().toLowerCase() + "-" + calculateArch().toString().toLowerCase() + ".dll";
            break;
        case UNKNOWN:
            nativeName = "unknown";
            break;
        }
        return nativeName;
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasInit()<br />
     *<br />
     * initializes the CUBLAS library and must be called before any other CUBLAS
     * API function is invoked. It allocates hardware resources necessary for
     * accessing the GPU.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_ALLOC_FAILED if resources could not be allocated<br />
     * CUBLAS_STATUS_SUCCESS if CUBLAS library initialized successfully<br />
     */
    private static native int cublasInitNative();

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasShutdown()<br />
     *<br />
     * releases CPU-side resources used by the CUBLAS library. The release of
     * GPU-side resources may be deferred until the application shuts down.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_SUCCESS if CUBLAS library shut down successfully<br />
     */
    public static native int cublasShutdown();

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasGetError()<br />
     *<br />
     * returns the last error that occurred on invocation of any of the CUBLAS
     * BLAS functions. While the CUBLAS helper functions return status directly,
     * the BLAS functions do not do so for improved compatibility with existing
     * environments that do not expect BLAS functions to return status. Reading
     * the error status via cublasGetError() resets the internal error state to
     * CUBLAS_STATUS_SUCCESS.
     */
    public static native int cublasGetError();

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasAlloc (int n, int elemSize, void **devicePtr)<br />
     *<br />
     * creates an object in GPU memory space capable of holding an array of n
     * elements, where each element requires elemSize bytes of storage. If the
     * function call is successful, a pointer to the object in GPU memory space
     * is placed in devicePtr. Note that this is a device pointer that cannot be
     * dereferenced in host code.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if n <= 0, or elemSize <= 0<br />
     * CUBLAS_STATUS_ALLOC_FAILED if the object could not be allocated due to
     * lack of resources.<br />
     * CUBLAS_STATUS_SUCCESS if storage was successfully allocated<br />
     */
    public static native int cublasAlloc(int n, int elemSize, String name);

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasFree (const void *devicePtr)<br />
     *<br />
     * destroys the object in GPU memory space pointed to by devicePtr.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INTERNAL_ERROR if the object could not be deallocated<br />
     * CUBLAS_STATUS_SUCCESS if object was destroyed successfully<br />
     */
    public static native int cublasFree(String name);

    /**
     * Set the specified log level for the JCublas library.<br />
     * <br />
     * <b><u>This method may only be called after JCublas has been initialized
     * with a call to JCublas.cublasInit() !</u></b> <br />
     * <br />
     * Currently supported log levels: <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_INFO: Print information about the CUBLAS functions that are executed <br />
     * LOG_TRACE: Print fine-grained memory management information in methods
     * like cublasSetVector <br />
     * 
     * @param logLevel
     *            The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel) {
        setLogLevel(logLevel.ordinal());
    }

    private static native void setLogLevel(int logLevel);

    // Debug method
    public static native void printVector(int n, String x);

    // Debug method
    public static native void printMatrix(int cols, String A, int lda);

    /*
     * Internal method to which all calls to an implementation of
     * cublasSetVector are finally delegated
     */
    private static native int cublasSetVector(int n, int elemSize, Buffer x, int offsetx, int incx, String y, int offsety, int incy);

    /*
     * Internal method to which all calls to an implementation of
     * cublasGetVector are finally delegated
     */
    private static native int cublasGetVector(int n, int elemSize, String x, int offsetx, int incx, Buffer y, int offsety, int incy);

    /*
     * Internal method to which all calls to an implementation of
     * cublasSetMatrix are finally delegated
     */
    private static native int cublasSetMatrix(int rows, int cols, int elemSize, Buffer A, int offsetA, int lda, String B, int offsetB, int ldb);

    /*
     * Internal method to which all calls to an implementation of
     * cublasGetMatrix are finally delegated
     */
    private static native int cublasGetMatrix(int rows, int cols, int elemSize, String A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    //============================================================================
    // Memory management methods for single precision data:

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus<br />
     * cublasSetVector (int n, int elemSize, const void *x, int incx, void *y,
     * int incy)<br />
     *<br />
     * copies n elements from a vector x in CPU memory space to a vector y in
     * GPU memory space. Elements in both vectors are assumed to have a size of
     * elemSize bytes. Storage spacing between consecutive elements is incx for
     * the source vector x and incy for the destination vector y. In general, y
     * points to an object, or part of an object, allocated via cublasAlloc().
     * Column major format for two-dimensional matrices is assumed throughout
     * CUBLAS. Therefore, if the increment for a vector is equal to 1, this
     * access a column vector while using an increment equal to the leading
     * dimension of the respective matrix accesses a row vector.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if incx, incy, or elemSize <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if an error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasSetVector(int n, FloatBuffer x, int incx, String y, int incy) {
        return cublasSetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasSetVector(int n, FloatBuffer x, int offsetx, int incx, String y, int offsety, int incy) {
        return cublasSetVector(n, 4, x, offsetx, incx, y, offsety, incy);
    }

    /**
     * Extended wrapper supporting float array arguments
     */
    public static int cublasSetVector(int n, float x[], int incx, String y, int incy) {
        return cublasSetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasSetVector(int n, float x[], int offsetx, int incx, String y, int offsety, int incy) {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(x.length * 4);
        byteBufferx.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferx = byteBufferx.asFloatBuffer();
        floatBufferx.put(x);
        return cublasSetVector(n, 4, floatBufferx, offsetx, incx, y, offsety, incy);
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasSetVector(int n, JCuComplex x[], int incx, String y, int incy) {
        return cublasSetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasSetVector(int n, JCuComplex x[], int offsetx, int incx, String y, int offsety, int incy) {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(x.length * 4 * 2);
        byteBufferx.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferx = byteBufferx.asFloatBuffer();

        int indexx = offsetx;
        for (int i = 0; i < n; i++, indexx += incx) {
            floatBufferx.put(indexx * 2 + 0, x[indexx].x);
            floatBufferx.put(indexx * 2 + 1, x[indexx].y);
        }
        return cublasSetVector(n, 8, floatBufferx, 0, 1, y, offsety, incy);
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus<br />
     * cublasGetVector (int n, int elemSize, const void *x, int incx, void *y,
     * int incy)<br />
     *<br />
     * copies n elements from a vector x in GPU memory space to a vector y in
     * CPU memory space. Elements in both vectors are assumed to have a size of
     * elemSize bytes. Storage spacing between consecutive elements is incx for
     * the source vector x and incy for the destination vector y. In general, x
     * points to an object, or part of an object, allocated via cublasAlloc().
     * Column major format for two-dimensional matrices is assumed throughout
     * CUBLAS. Therefore, if the increment for a vector is equal to 1, this
     * access a column vector while using an increment equal to the leading
     * dimension of the respective matrix accesses a row vector.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if incx, incy, or elemSize <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if an error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasGetVector(int n, String x, int incx, FloatBuffer y, int incy) {
        return cublasGetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasGetVector(int n, String x, int offsetx, int incx, FloatBuffer y, int offsety, int incy) {
        return cublasGetVector(n, 4, x, offsetx, incx, y, offsety, incy);
    }

    /**
     * Extended wrapper supporting float array arguments
     */
    public static int cublasGetVector(int n, String x, int incx, float y[], int incy) {
        return cublasGetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasGetVector(int n, String x, int offsetx, int incx, float y[], int offsety, int incy) {
        ByteBuffer byteBuffery = ByteBuffer.allocateDirect(y.length * 4);
        byteBuffery.order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffery = byteBuffery.asFloatBuffer();
        int status = cublasGetVector(n, 4, x, offsetx, incx, floatBuffery, offsety, incy);
        if (status == CUBLAS_STATUS_SUCCESS) {
            floatBuffery.rewind();
            if (incx == 1 && incy == 1) {
                floatBuffery.get(y, offsety, n);
            } else {
                int indexy = offsety;
                for (int i = 0; i < n; i++, indexy += incy) {
                    y[indexy] = floatBuffery.get(indexy);
                }
            }
        }
        return status;
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasGetVector(int n, String x, int incx, JCuComplex y[], int incy) {
        return cublasGetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasGetVector(int n, String x, int offsetx, int incx, JCuComplex y[], int offsety, int incy) {
        ByteBuffer byteBuffery = ByteBuffer.allocateDirect(y.length * 4 * 2);
        byteBuffery.order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffery = byteBuffery.asFloatBuffer();
        int status = cublasGetVector(n, 8, x, offsetx, incx, floatBuffery, offsety, incy);
        if (status == CUBLAS_STATUS_SUCCESS) {
            floatBuffery.rewind();
            int indexy = offsety;
            for (int i = 0; i < n; i++, indexy += incy) {
                y[indexy].x = floatBuffery.get(indexy * 2 + 0);
                y[indexy].y = floatBuffery.get(indexy * 2 + 1);
            }
        }
        return status;
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasSetMatrix (int rows, int cols, int elemSize, const
     * void *A, int lda, void *B, int ldb)<br />
     *<br />
     * copies a tile of rows x cols elements from a matrix A in CPU memory space
     * to a matrix B in GPU memory space. Each element requires storage of
     * elemSize bytes. Both matrices are assumed to be stored in column major
     * format, with the leading dimension (i.e. number of rows) of source matrix
     * A provided in lda, and the leading dimension of matrix B provided in ldb.
     * In general, B points to an object, or part of an object, that was
     * allocated via cublasAlloc().<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if rows or cols < 0, or elemSize, lda, or ldb
     * <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasSetMatrix(int rows, int cols, FloatBuffer A, int lda, String B, int ldb) {
        return cublasSetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasSetMatrix(int rows, int cols, FloatBuffer A, int offsetA, int lda, String B, int offsetB, int ldb) {
        return cublasSetMatrix(rows, cols, 4, A, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Extended wrapper supporting float array arguments
     */
    public static int cublasSetMatrix(int rows, int cols, float[] A, int lda, String B, int ldb) {
        return cublasSetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasSetMatrix(int rows, int cols, float[] A, int offsetA, int lda, String B, int offsetB, int ldb) {
        ByteBuffer byteBufferA = ByteBuffer.allocateDirect(A.length * 4);
        byteBufferA.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferA = byteBufferA.asFloatBuffer();
        floatBufferA.put(A);
        return cublasSetMatrix(rows, cols, 4, floatBufferA, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasSetMatrix(int rows, int cols, JCuComplex A[], int lda, String B, int ldb) {
        return cublasSetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasSetMatrix(int rows, int cols, JCuComplex A[], int offsetA, int lda, String B, int offsetB, int ldb) {
        ByteBuffer byteBufferA = ByteBuffer.allocateDirect(A.length * 4 * 2);
        byteBufferA.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferA = byteBufferA.asFloatBuffer();
        for (int i = 0; i < A.length; i++) {
            floatBufferA.put(A[i].x);
            floatBufferA.put(A[i].y);
        }
        return cublasSetMatrix(rows, cols, 4, floatBufferA, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasGetMatrix (int rows, int cols, int elemSize, const
     * void *A, int lda, void *B, int ldb)<br />
     *<br />
     * copies a tile of rows x cols elements from a matrix A in GPU memory space
     * to a matrix B in CPU memory space. Each element requires storage of
     * elemSize bytes. Both matrices are assumed to be stored in column major
     * format, with the leading dimension (i.e. number of rows) of source matrix
     * A provided in lda, and the leading dimension of matrix B provided in ldb.
     * In general, A points to an object, or part of an object, that was
     * allocated via cublasAlloc().<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if rows, cols, eleSize, lda, or ldb <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int lda, FloatBuffer B, int ldb) {
        return cublasGetMatrix(rows, cols, 4, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int offsetA, int lda, FloatBuffer B, int offsetB, int ldb) {
        return cublasGetMatrix(rows, cols, 4, A, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Extended wrapper supporting float array arguments
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int lda, float B[], int ldb) {
        return cublasGetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int offsetA, int lda, float B[], int offsetB, int ldb) {
        ByteBuffer byteBufferB = ByteBuffer.allocateDirect(B.length * 4);
        byteBufferB.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferB = byteBufferB.asFloatBuffer();
        int status = cublasGetMatrix(rows, cols, 4, A, offsetA, lda, floatBufferB, offsetB, ldb);
        if (status == CUBLAS_STATUS_SUCCESS) {
            floatBufferB.rewind();
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    int index = c * ldb + r + offsetB;
                    B[index] = floatBufferB.get(index);
                }
            }
        }
        return status;
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int lda, JCuComplex B[], int ldb) {
        return cublasGetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int offsetA, int lda, JCuComplex B[], int offsetB, int ldb) {
        ByteBuffer byteBufferB = ByteBuffer.allocateDirect(B.length * 4 * 2);
        byteBufferB.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferB = byteBufferB.asFloatBuffer();
        int status = cublasGetMatrix(rows, cols, 4, A, offsetA, lda, floatBufferB, offsetB, ldb);
        if (status == CUBLAS_STATUS_SUCCESS) {
            floatBufferB.rewind();
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    int index = c * ldb + r + offsetB;
                    B[index].x = floatBufferB.get(index * 2 + 0);
                    B[index].y = floatBufferB.get(index * 2 + 1);
                }
            }
        }
        return status;
    }

    //============================================================================
    // Memory management methods for double precision data:

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus<br />
     * cublasSetVector (int n, int elemSize, const void *x, int incx, void *y,
     * int incy)<br />
     *<br />
     * copies n elements from a vector x in CPU memory space to a vector y in
     * GPU memory space. Elements in both vectors are assumed to have a size of
     * elemSize bytes. Storage spacing between consecutive elements is incx for
     * the source vector x and incy for the destination vector y. In general, y
     * points to an object, or part of an object, allocated via cublasAlloc().
     * Column major format for two-dimensional matrices is assumed throughout
     * CUBLAS. Therefore, if the increment for a vector is equal to 1, this
     * access a column vector while using an increment equal to the leading
     * dimension of the respective matrix accesses a row vector.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if incx, incy, or elemSize <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if an error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasSetVector(int n, DoubleBuffer x, int incx, String y, int incy) {
        return cublasSetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasSetVector(int n, DoubleBuffer x, int offsetx, int incx, String y, int offsety, int incy) {
        return cublasSetVector(n, 8, x, offsetx, incx, y, offsety, incy);
    }

    /**
     * Extended wrapper supporting double array arguments
     */
    public static int cublasSetVector(int n, double x[], int incx, String y, int incy) {
        return cublasSetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasSetVector(int n, double x[], int offsetx, int incx, String y, int offsety, int incy) {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(x.length * 8);
        byteBufferx.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferx = byteBufferx.asDoubleBuffer();
        doubleBufferx.put(x);
        return cublasSetVector(n, 8, doubleBufferx, offsetx, incx, y, offsety, incy);
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasSetVector(int n, JCuDoubleComplex x[], int incx, String y, int incy) {
        return cublasSetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasSetVector(int n, JCuDoubleComplex x[], int offsetx, int incx, String y, int offsety, int incy) {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(x.length * 8 * 2);
        byteBufferx.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferx = byteBufferx.asDoubleBuffer();

        int indexx = offsetx;
        for (int i = 0; i < n; i++, indexx += incx) {
            doubleBufferx.put(indexx * 2 + 0, x[indexx].x);
            doubleBufferx.put(indexx * 2 + 1, x[indexx].y);
        }
        return cublasSetVector(n, 16, doubleBufferx, 0, 1, y, offsety, incy);
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus<br />
     * cublasGetVector (int n, int elemSize, const void *x, int incx, void *y,
     * int incy)<br />
     *<br />
     * copies n elements from a vector x in GPU memory space to a vector y in
     * CPU memory space. Elements in both vectors are assumed to have a size of
     * elemSize bytes. Storage spacing between consecutive elements is incx for
     * the source vector x and incy for the destination vector y. In general, x
     * points to an object, or part of an object, allocated via cublasAlloc().
     * Column major format for two-dimensional matrices is assumed throughout
     * CUBLAS. Therefore, if the increment for a vector is equal to 1, this
     * access a column vector while using an increment equal to the leading
     * dimension of the respective matrix accesses a row vector.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if incx, incy, or elemSize <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if an error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasGetVector(int n, String x, int incx, DoubleBuffer y, int incy) {
        return cublasGetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasGetVector(int n, String x, int offsetx, int incx, DoubleBuffer y, int offsety, int incy) {
        return cublasGetVector(n, 8, x, offsetx, incx, y, offsety, incy);
    }

    /**
     * Extended wrapper supporting double array arguments
     */
    public static int cublasGetVector(int n, String x, int incx, double y[], int incy) {
        return cublasGetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasGetVector(int n, String x, int offsetx, int incx, double y[], int offsety, int incy) {
        ByteBuffer byteBuffery = ByteBuffer.allocateDirect(y.length * 8);
        byteBuffery.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBuffery = byteBuffery.asDoubleBuffer();
        int status = cublasGetVector(n, 8, x, offsetx, incx, doubleBuffery, offsety, incy);
        if (status == CUBLAS_STATUS_SUCCESS) {
            doubleBuffery.rewind();
            if (incx == 1 && incy == 1) {
                doubleBuffery.get(y, offsety, n);
            } else {
                int indexy = offsety;
                for (int i = 0; i < n; i++, indexy += incy) {
                    y[indexy] = doubleBuffery.get(indexy);
                }
            }
        }
        return status;
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasGetVector(int n, String x, int incx, JCuDoubleComplex y[], int incy) {
        return cublasGetVector(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the vectors.
     */
    public static int cublasGetVector(int n, String x, int offsetx, int incx, JCuDoubleComplex y[], int offsety, int incy) {
        ByteBuffer byteBuffery = ByteBuffer.allocateDirect(y.length * 8 * 2);
        byteBuffery.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBuffery = byteBuffery.asDoubleBuffer();
        int status = cublasGetVector(n, 8, x, offsetx, incx, doubleBuffery, offsety, incy);
        if (status == CUBLAS_STATUS_SUCCESS) {
            doubleBuffery.rewind();
            int indexy = offsety;
            for (int i = 0; i < n; i++, indexy += incy) {
                y[indexy].x = doubleBuffery.get(indexy * 2 + 0);
                y[indexy].y = doubleBuffery.get(indexy * 2 + 1);
            }
        }
        return status;
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasSetMatrix (int rows, int cols, int elemSize, const
     * void *A, int lda, void *B, int ldb)<br />
     *<br />
     * copies a tile of rows x cols elements from a matrix A in CPU memory space
     * to a matrix B in GPU memory space. Each element requires storage of
     * elemSize bytes. Both matrices are assumed to be stored in column major
     * format, with the leading dimension (i.e. number of rows) of source matrix
     * A provided in lda, and the leading dimension of matrix B provided in ldb.
     * In general, B points to an object, or part of an object, that was
     * allocated via cublasAlloc().<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if rows or cols < 0, or elemSize, lda, or ldb
     * <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasSetMatrix(int rows, int cols, DoubleBuffer A, int lda, String B, int ldb) {
        return cublasSetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasSetMatrix(int rows, int cols, DoubleBuffer A, int offsetA, int lda, String B, int offsetB, int ldb) {
        return cublasSetMatrix(rows, cols, 8, A, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Extended wrapper supporting double array arguments
     */
    public static int cublasSetMatrix(int rows, int cols, double[] A, int lda, String B, int ldb) {
        return cublasSetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasSetMatrix(int rows, int cols, double[] A, int offsetA, int lda, String B, int offsetB, int ldb) {
        ByteBuffer byteBufferA = ByteBuffer.allocateDirect(A.length * 8);
        byteBufferA.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferA = byteBufferA.asDoubleBuffer();
        doubleBufferA.put(A);
        return cublasSetMatrix(rows, cols, 8, doubleBufferA, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasSetMatrix(int rows, int cols, JCuDoubleComplex A[], int lda, String B, int ldb) {
        return cublasSetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasSetMatrix(int rows, int cols, JCuDoubleComplex A[], int offsetA, int lda, String B, int offsetB, int ldb) {
        ByteBuffer byteBufferA = ByteBuffer.allocateDirect(A.length * 4 * 2);
        byteBufferA.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferA = byteBufferA.asDoubleBuffer();
        for (int i = 0; i < A.length; i++) {
            doubleBufferA.put(A[i].x);
            doubleBufferA.put(A[i].y);
        }
        return cublasSetMatrix(rows, cols, 8, doubleBufferA, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus cublasGetMatrix (int rows, int cols, int elemSize, const
     * void *A, int lda, void *B, int ldb)<br />
     *<br />
     * copies a tile of rows x cols elements from a matrix A in GPU memory space
     * to a matrix B in CPU memory space. Each element requires storage of
     * elemSize bytes. Both matrices are assumed to be stored in column major
     * format, with the leading dimension (i.e. number of rows) of source matrix
     * A provided in lda, and the leading dimension of matrix B provided in ldb.
     * In general, A points to an object, or part of an object, that was
     * allocated via cublasAlloc().<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE if rows, cols, eleSize, lda, or ldb <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR if error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS if the operation completed successfully<br />
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int lda, DoubleBuffer B, int ldb) {
        return cublasGetMatrix(rows, cols, 8, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int offsetA, int lda, DoubleBuffer B, int offsetB, int ldb) {
        return cublasGetMatrix(rows, cols, 8, A, offsetA, lda, B, offsetB, ldb);
    }

    /**
     * Extended wrapper supporting double array arguments
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int lda, double B[], int ldb) {
        return cublasGetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int offsetA, int lda, double B[], int offsetB, int ldb) {
        ByteBuffer byteBufferB = ByteBuffer.allocateDirect(B.length * 8);
        byteBufferB.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferB = byteBufferB.asDoubleBuffer();
        int status = cublasGetMatrix(rows, cols, 8, A, offsetA, lda, doubleBufferB, offsetB, ldb);
        if (status == CUBLAS_STATUS_SUCCESS) {
            doubleBufferB.rewind();
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    int index = c * ldb + r + offsetB;
                    B[index] = doubleBufferB.get(index);
                }
            }
        }
        return status;
    }

    /**
     * Extended wrapper supporting complex array arguments
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int lda, JCuDoubleComplex B[], int ldb) {
        return cublasGetMatrix(rows, cols, A, 0, lda, B, 0, ldb);
    }

    /**
     * Extended wrapper offering additional parameters to specify the offsets
     * inside the matrices
     */
    public static int cublasGetMatrix(int rows, int cols, String A, int offsetA, int lda, JCuDoubleComplex B[], int offsetB, int ldb) {
        ByteBuffer byteBufferB = ByteBuffer.allocateDirect(B.length * 8 * 2);
        byteBufferB.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferB = byteBufferB.asDoubleBuffer();
        int status = cublasGetMatrix(rows, cols, 8, A, offsetA, lda, doubleBufferB, offsetB, ldb);
        if (status == CUBLAS_STATUS_SUCCESS) {
            doubleBufferB.rewind();
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    int index = c * ldb + r + offsetB;
                    B[index].x = doubleBufferB.get(index * 2 + 0);
                    B[index].y = doubleBufferB.get(index * 2 + 1);
                }
            }
        }
        return status;
    }

    //============================================================================
    // Methods that are not handled by the code generator:

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasSrotm (int n, float *x, int incx, float *y, int incy,
     *              const float* sparam)
     * 
     * applies the modified Givens transformation, h, to the 2 x n matrix
     * 
     *    ( transpose(x) )
     *    ( transpose(y) )
     * 
     * The elements of x are in x[lx + i * incx], i = 0 to n-1, where lx = 1 if
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy. With sparam[0] = sflag, h has one of the following forms:
     * 
     *        sflag = -1.0f   sflag = 0.0f    sflag = 1.0f    sflag = -2.0f
     * 
     *        (sh00  sh01)    (1.0f  sh01)    (sh00  1.0f)    (1.0f  0.0f)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10  1.0f)    (-1.0f sh11)    (0.0f  1.0f)
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     * 
     * Output
     * ------
     * x     rotated vector x (unchanged if n &lt;= 0)
     * y     rotated vector y (unchanged if n &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/srotm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */
    public static native void cublasSrotm(int n, String x, int offsetx, int incx, String y, int offsety, int incy, float sparam[]);

    public static void cublasSrotm(int n, String x, int incx, String y, int incy, float sparam[]) {
        cublasSrotm(n, x, 0, incx, y, 0, incy, sparam);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasSrotmg (float *psd1, float *psd2, float *psx1, const float *psy1,
     *                float *sparam)
     * 
     * constructs the modified Givens transformation matrix h which zeros
     * the second component of the 2-vector transpose(sqrt(sd1)*sx1,sqrt(sd2)*sy1).
     * With sparam[0] = sflag, h has one of the following forms:
     * 
     *        sflag = -1.0f   sflag = 0.0f    sflag = 1.0f    sflag = -2.0f
     * 
     *        (sh00  sh01)    (1.0f  sh01)    (sh00  1.0f)    (1.0f  0.0f)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10  1.0f)    (-1.0f sh11)    (0.0f  1.0f)
     * 
     * sparam[1] through sparam[4] contain sh00, sh10, sh01, sh11,
     * respectively. Values of 1.0f, -1.0f, or 0.0f implied by the value
     * of sflag are not stored in sparam.
     * 
     * Input
     * -----
     * sd1    single precision scalar
     * sd2    single precision scalar
     * sx1    single precision scalar
     * sy1    single precision scalar
     * 
     * Output
     * ------
     * sd1    changed to represent the effect of the transformation
     * sd2    changed to represent the effect of the transformation
     * sx1    changed to represent the effect of the transformation
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     * 
     * Reference: http://www.netlib.org/blas/srotmg.f
     * 
     * This functions does not set any error status.
     * </pre>
     */
    public static native void cublasSrotmg(float sd1[], float sd2[], float sx1[], float sy1, float sparam[]);

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDrotm (int n, double *x, int incx, double *y, int incy, 
     *              const double* sparam)
     * 
     * applies the modified Givens transformation, h, to the 2 x n matrix
     * 
     *    ( transpose(x) )
     *    ( transpose(y) )
     * 
     * The elements of x are in x[lx + i * incx], i = 0 to n-1, where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and 
     * incy. With sparam[0] = sflag, h has one of the following forms:
     * 
     *        sflag = -1.0    sflag = 0.0     sflag = 1.0     sflag = -2.0
     * 
     *        (sh00  sh01)    (1.0   sh01)    (sh00   1.0)    (1.0    0.0)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10   1.0)    (-1.0  sh11)    (0.0    1.0)
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1] 
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     * 
     * Output
     * ------
     * x     rotated vector x (unchanged if n &lt;= 0)
     * y     rotated vector y (unchanged if n &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/drotm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */
    public static native void cublasDrotm(int n, String x, int offsetx, int incx, String y, int offsety, int incy, double sparam[]);

    public static void cublasSrotm(int n, String x, int incx, String y, int incy, double sparam[]) {
        cublasDrotm(n, x, 0, incx, y, 0, incy, sparam);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDrotmg (double *psd1, double *psd2, double *psx1, const double *psy1,
     *               double *sparam)
     * 
     * constructs the modified Givens transformation matrix h which zeros
     * the second component of the 2-vector transpose(sqrt(sd1)*sx1,sqrt(sd2)*sy1).
     * With sparam[0] = sflag, h has one of the following forms:
     * 
     *        sflag = -1.0    sflag = 0.0     sflag = 1.0     sflag = -2.0
     * 
     *        (sh00  sh01)    (1.0   sh01)    (sh00   1.0)    (1.0    0.0)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10   1.0)    (-1.0  sh11)    (0.0    1.0)
     * 
     * sparam[1] through sparam[4] contain sh00, sh10, sh01, sh11, 
     * respectively. Values of 1.0, -1.0, or 0.0 implied by the value 
     * of sflag are not stored in sparam.
     * 
     * Input
     * -----
     * sd1    single precision scalar
     * sd2    single precision scalar
     * sx1    single precision scalar
     * sy1    single precision scalar
     * 
     * Output
     * ------
     * sd1    changed to represent the effect of the transformation
     * sd2    changed to represent the effect of the transformation
     * sx1    changed to represent the effect of the transformation
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1] 
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     * 
     * Reference: http://www.netlib.org/blas/drotmg.f
     * 
     * This functions does not set any error status.
     * 
     * </pre>
     */
    public static native void cublasDrotmg(double sd1[], double sd2[], double sx1[], double sy1, double sparam[]);

    //============================================================================
    // Auto-generated part:

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * int 
     * cublasIsamax (int n, const float *x, int incx)
     * 
     * finds the smallest index of the maximum magnitude element of single
     * precision vector x; that is, the result is the first i, i = 0 to n - 1, 
     * that maximizes abs(x[1 + i * incx])).
     * 
     * Input
     * -----
     * n      number of elements in input vector
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * returns the smallest index (0 if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/isamax.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native int cublasIsamax(int n, String x, int offsetx, int incx);

    public static int cublasIsamax(int n, String x, int incx) {
        return cublasIsamax(n, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * int 
     * cublasIsamin (int n, const float *x, int incx)
     * 
     * finds the smallest index of the minimum magnitude element of single
     * precision vector x; that is, the result is the first i, i = 0 to n - 1, 
     * that minimizes abs(x[1 + i * incx])).
     * 
     * Input
     * -----
     * n      number of elements in input vector
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * returns the smallest index (0 if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/scilib/blass.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native int cublasIsamin(int n, String x, int offsetx, int incx);

    public static int cublasIsamin(int n, String x, int incx) {
        return cublasIsamin(n, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasSaxpy (int n, float alpha, const float *x, int incx, float *y, 
     *              int incy)
     * 
     * multiplies single precision vector x by single precision scalar alpha 
     * and adds the result to single precision vector y; that is, it overwrites 
     * single precision y with single precision alpha * x + y. For i = 0 to n - 1, 
     * it replaces y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i *
     * incy], where lx = 1 if incx &gt;= 0, else lx = 1 +(1 - n) * incx, and ly is 
     * defined in a similar way using incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single precision scalar multiplier
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     * 
     * Output
     * ------
     * y      single precision result (unchanged if n &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/saxpy.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSaxpy(int n, float alpha, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasSaxpy(int n, float alpha, String x, int incx, String y, int incy) {
        cublasSaxpy(n, alpha, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasScopy (int n, const float *x, int incx, float *y, int incy)
     * 
     * copies the single precision vector x to the single precision vector y. For 
     * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar 
     * way using incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     * 
     * Output
     * ------
     * y      contains single precision vector x
     * 
     * Reference: http://www.netlib.org/blas/scopy.f
     * 
     * Error status for this function can be retrieved via cublasGetError(). 
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasScopy(int n, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasScopy(int n, String x, int incx, String y, int incy) {
        cublasScopy(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSrot (int n, float *x, int incx, float *y, int incy, float sc, 
     *             float ss)
     * 
     * multiplies a 2x2 matrix ( sc ss) with the 2xn matrix ( transpose(x) )
     *                         (-ss sc)                     ( transpose(y) )
     * 
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and 
     * incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     * sc     element of rotation matrix
     * ss     element of rotation matrix
     * 
     * Output
     * ------
     * x      rotated vector x (unchanged if n &lt;= 0)
     * y      rotated vector y (unchanged if n &lt;= 0)
     * 
     * Reference  http://www.netlib.org/blas/srot.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSrot(int n, String x, int offsetx, int incx, String y, int offsety, int incy, float sc, float ss);

    public static void cublasSrot(int n, String x, int incx, String y, int incy, float sc, float ss) {
        cublasSrot(n, x, 0, incx, y, 0, incy, sc, ss);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSrotg (float *sa, float *sb, float *sc, float *ss)
     * 
     * constructs the Givens tranformation
     * 
     *        ( sc  ss )
     *    G = (        ) ,  sc&circ;2 + ss&circ;2 = 1,
     *        (-ss  sc )
     * 
     * which zeros the second entry of the 2-vector transpose(sa, sb).
     * 
     * The quantity r = (+/-) sqrt (sa&circ;2 + sb&circ;2) overwrites sa in storage. The 
     * value of sb is overwritten by a value z which allows sc and ss to be 
     * recovered by the following algorithm:
     * 
     *    if z=1          set sc = 0.0 and ss = 1.0
     *    if abs(z) &lt; 1   set sc = sqrt(1-z&circ;2) and ss = z
     *    if abs(z) &gt; 1   set sc = 1/z and ss = sqrt(1-sc&circ;2)
     * 
     * The function srot (n, x, incx, y, incy, sc, ss) normally is called next
     * to apply the transformation to a 2 x n matrix.
     * 
     * Input
     * -----
     * sa     single precision scalar
     * sb     single precision scalar
     * 
     * Output
     * ------
     * sa     single precision r
     * sb     single precision z
     * sc     single precision result
     * ss     single precision result
     * 
     * Reference: http://www.netlib.org/blas/srotg.f
     * 
     * This function does not set any error status.
     * </pre>
     */

    public static native void cublasSrotg(String sa, int offsetsa, String sb, int offsetsb, String sc, int offsetsc, String ss, int offsetss);

    public static void cublasSrotg(String sa, String sb, String sc, String ss) {
        cublasSrotg(sa, 0, sb, 0, sc, 0, ss, 0);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * sscal (int n, float alpha, float *x, int incx)
     * 
     * replaces single precision vector x with single precision alpha * x. For i 
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
     * where ix = 1 if incx &gt;= 0, else ix = 1 + (1 - n) * incx.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single precision scalar multiplier
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * x      single precision result (unchanged if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/sscal.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSscal(int n, float alpha, String x, int offsetx, int incx);

    public static void cublasSscal(int n, float alpha, String x, int incx) {
        cublasSscal(n, alpha, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasSswap (int n, float *x, int incx, float *y, int incy)
     * 
     * replaces single precision vector x with single precision alpha * x. For i 
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
     * where ix = 1 if incx &gt;= 0, else ix = 1 + (1 - n) * incx.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single precision scalar multiplier
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * x      single precision result (unchanged if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/sscal.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSswap(int n, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasSswap(int n, String x, int incx, String y, int incy) {
        cublasSswap(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasCaxpy (int n, cuComplex alpha, const cuComplex *x, int incx, 
     *              cuComplex *y, int incy)
     * 
     * multiplies single-complex vector x by single-complex scalar alpha and adds 
     * the result to single-complex vector y; that is, it overwrites single-complex
     * y with single-complex alpha * x + y. For i = 0 to n - 1, it replaces 
     * y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i * incy], where 
     * lx = 0 if incx &gt;= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a 
     * similar way using incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single-complex scalar multiplier
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     * 
     * Output
     * ------
     * y      single-complex result (unchanged if n &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/caxpy.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCaxpy(int n, JCuComplex alpha, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasCaxpy(int n, JCuComplex alpha, String x, int incx, String y, int incy) {
        cublasCaxpy(n, alpha, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y, int incy)
     * 
     * copies the single-complex vector x to the single-complex vector y. For 
     * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar 
     * way using incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     * 
     * Output
     * ------
     * y      contains single complex vector x
     * 
     * Reference: http://www.netlib.org/blas/ccopy.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCcopy(int n, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasCcopy(int n, String x, int incx, String y, int incy) {
        cublasCcopy(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx)
     * 
     * replaces single-complex vector x with single-complex alpha * x. For i 
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
     * where ix = 1 if incx &gt;= 0, else ix = 1 + (1 - n) * incx.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single-complex scalar multiplier
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * x      single-complex result (unchanged if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/cscal.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCscal(int n, JCuComplex alpha, String x, int offsetx, int incx);

    public static void cublasCscal(int n, JCuComplex alpha, String x, int incx) {
        cublasCscal(n, alpha, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasCrotg (cuComplex *ca, cuComplex cb, float *sc, cuComplex *cs)
     * 
     * constructs the complex Givens tranformation
     * 
     *        ( sc  cs )
     *    G = (        ) ,  sc&circ;2 + cabs(cs)&circ;2 = 1,
     *        (-cs  sc )
     * 
     * which zeros the second entry of the complex 2-vector transpose(ca, cb).
     * 
     * The quantity ca/cabs(ca)*norm(ca,cb) overwrites ca in storage. The 
     * function crot (n, x, incx, y, incy, sc, cs) is normally called next
     * to apply the transformation to a 2 x n matrix.
     * 
     * Input
     * -----
     * ca     single-precision complex precision scalar
     * cb     single-precision complex scalar
     * 
     * Output
     * ------
     * ca     single-precision complex ca/cabs(ca)*norm(ca,cb)
     * sc     single-precision cosine component of rotation matrix
     * cs     single-precision complex sine component of rotation matrix
     * 
     * Reference: http://www.netlib.org/blas/crotg.f
     * 
     * This function does not set any error status.
     * </pre>
     */

    public static native void cublasCrotg(String pca, int offsetpca, JCuComplex cb, String psc, int offsetpsc, String pcs, int offsetpcs);

    public static void cublasCrotg(String pca, JCuComplex cb, String psc, String pcs) {
        cublasCrotg(pca, 0, cb, psc, 0, pcs, 0);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasCrot (int n, cuComplex *x, int incx, cuComplex *y, int incy, float sc,
     *             cuComplex cs)
     * 
     * multiplies a 2x2 matrix ( sc       cs) with the 2xn matrix ( transpose(x) )
     *                         (-conj(cs) sc)                     ( transpose(y) )
     * 
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and 
     * incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-precision complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-precision complex vector with n elements
     * incy   storage spacing between elements of y
     * sc     single-precision cosine component of rotation matrix
     * cs     single-precision complex sine component of rotation matrix
     * 
     * Output
     * ------
     * x      rotated single-precision complex vector x (unchanged if n &lt;= 0)
     * y      rotated single-precision complex vector y (unchanged if n &lt;= 0)
     * 
     * Reference: http://netlib.org/lapack/explore-html/crot.f.html
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCrot(int n, String x, int offsetx, int incx, String y, int offsety, int incy, float c, JCuComplex s);

    public static void cublasCrot(int n, String x, int incx, String y, int incy, float c, JCuComplex s) {
        cublasCrot(n, x, 0, incx, y, 0, incy, c, s);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * csrot (int n, cuComplex *x, int incx, cuCumplex *y, int incy, float c, 
     *        float s)
     * 
     * multiplies a 2x2 rotation matrix ( c s) with a 2xn matrix ( transpose(x) )
     *                                  (-s c)                   ( transpose(y) )
     * 
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and 
     * incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-precision complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-precision complex vector with n elements
     * incy   storage spacing between elements of y
     * c      cosine component of rotation matrix
     * s      sine component of rotation matrix
     * 
     * Output
     * ------
     * x      rotated vector x (unchanged if n &lt;= 0)
     * y      rotated vector y (unchanged if n &lt;= 0)
     * 
     * Reference  http://www.netlib.org/blas/csrot.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCsrot(int n, String x, int offsetx, int incx, String y, int offsety, int incy, float c, float s);

    public static void cublasCsrot(int n, String x, int incx, String y, int incy, float c, float s) {
        cublasCsrot(n, x, 0, incx, y, 0, incy, c, s);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasCsscal (int n, float alpha, cuComplex *x, int incx)
     * 
     * replaces single-complex vector x with single-complex alpha * x. For i 
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
     * where ix = 1 if incx &gt;= 0, else ix = 1 + (1 - n) * incx.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single precision scalar multiplier
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * x      single-complex result (unchanged if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/csscal.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCsscal(int n, float alpha, String x, int offsetx, int incx);

    public static void cublasCsscal(int n, float alpha, String x, int incx) {
        cublasCsscal(n, alpha, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasCswap (int n, const cuComplex *x, int incx, cuComplex *y, int incy)
     * 
     * interchanges the single-complex vector x with the single-complex vector y. 
     * For i = 0 to n-1, interchanges x[lx + i * incx] with y[ly + i * incy], where
     * lx = 1 if incx &gt;= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a 
     * similar way using incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     * 
     * Output
     * ------
     * x      contains-single complex vector y
     * y      contains-single complex vector x
     * 
     * Reference: http://www.netlib.org/blas/cswap.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCswap(int n, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasCswap(int n, String x, int incx, String y, int incy) {
        cublasCswap(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * int 
     * cublasIcamax (int n, const float *x, int incx)
     * 
     * finds the smallest index of the element having maximum absolute value
     * in single-complex vector x; that is, the result is the first i, i = 0 
     * to n - 1 that maximizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
     * 
     * Input
     * -----
     * n      number of elements in input vector
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * returns the smallest index (0 if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/icamax.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native int cublasIcamax(int n, String x, int offsetx, int incx);

    public static int cublasIcamax(int n, String x, int incx) {
        return cublasIcamax(n, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * int 
     * cublasIcamin (int n, const float *x, int incx)
     * 
     * finds the smallest index of the element having minimum absolute value
     * in single-complex vector x; that is, the result is the first i, i = 0 
     * to n - 1 that minimizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
     * 
     * Input
     * -----
     * n      number of elements in input vector
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * returns the smallest index (0 if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: see ICAMAX.
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native int cublasIcamin(int n, String x, int offsetx, int incx);

    public static int cublasIcamin(int n, String x, int incx) {
        return cublasIcamin(n, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSgbmv (char trans, int m, int n, int kl, int ku, float alpha,
     *              const float *A, int lda, const float *x, int incx, float beta,
     *              float *y, int incy)
     * 
     * performs one of the matrix-vector operations
     * 
     *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
     * 
     * alpha and beta are single precision scalars. x and y are single precision
     * vectors. A is an m by n band matrix consisting of single precision elements
     * with kl sub-diagonals and ku super-diagonals.
     * 
     * Input
     * -----
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T', 
     *        't', 'C', or 'c', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least 
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * kl     specifies the number of sub-diagonals of matrix A. It must be at 
     *        least zero.
     * ku     specifies the number of super-diagonals of matrix A. It must be at 
     *        least zero.
     * alpha  single precision scalar multiplier applied to op(A).
     * A      single precision array of dimensions (lda, n). The leading
     *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
     *        supplied column by column, with the leading diagonal of the matrix 
     *        in row (ku + 1) of the array, the first super-diagonal starting at 
     *        position 2 in row ku, the first sub-diagonal starting at position 1
     *        in row (ku + 2), and so on. Elements in the array A that do not 
     *        correspond to elements in the band matrix (such as the top left 
     *        ku x ku triangle) are not referenced.
     * lda    leading dimension of A. lda must be at least (kl + ku + 1).
     * x      single precision array of length at least (1+(n-1)*abs(incx)) when 
     *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y. If beta is 
     *        zero, y is not read.
     * y      single precision array of length at least (1+(m-1)*abs(incy)) when 
     *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If 
     *        beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     * 
     * Output
     * ------
     * y      updated according to y = alpha*op(A)*x + beta*y
     * 
     * Reference: http://www.netlib.org/blas/sgbmv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n, kl, or ku &lt; 0; if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, String A, int offsetA, int lda, String x, int offsetx, int incx, float beta, String y, int offsety, int incy);

    public static void cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, String A, int lda, String x, int incx, float beta, String y, int incy) {
        cublasSgbmv(trans, m, n, kl, ku, alpha, A, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * cublasSgemv (char trans, int m, int n, float alpha, const float *A, int lda,
     *              const float *x, int incx, float beta, float *y, int incy)
     * 
     * performs one of the matrix-vector operations
     * 
     *    y = alpha * op(A) * x + beta * y,
     * 
     * where op(A) is one of
     * 
     *    op(A) = A   or   op(A) = transpose(A)
     * 
     * where alpha and beta are single precision scalars, x and y are single 
     * precision vectors, and A is an m x n matrix consisting of single precision
     * elements. Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array in which A is stored.
     * 
     * Input
     * -----
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
     *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least 
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least 
     *        zero.
     * alpha  single precision scalar multiplier applied to op(A).
     * A      single precision array of dimensions (lda, n) if trans = 'n' or 
     *        'N'), and of dimensions (lda, m) otherwise. lda must be at least 
     *        max(1, m) and at least max(1, n) otherwise.
     * lda    leading dimension of two-dimensional array used to store matrix A
     * x      single precision array of length at least (1 + (n - 1) * abs(incx))
     *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx)) 
     *        otherwise.
     * incx   specifies the storage spacing between elements of x. incx must not 
     *        be zero.
     * beta   single precision scalar multiplier applied to vector y. If beta 
     *        is zero, y is not read.
     * y      single precision array of length at least (1 + (m - 1) * abs(incy))
     *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy)) 
     *        otherwise.
     * incy   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * 
     * Output
     * ------
     * y      updated according to alpha * op(A) * x + beta * y
     * 
     * Reference: http://www.netlib.org/blas/sgemv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are &lt; 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSgemv(char trans, int m, int n, float alpha, String A, int offsetA, int lda, String x, int offsetx, int incx, float beta, String y, int offsety, int incy);

    public static void cublasSgemv(char trans, int m, int n, float alpha, String A, int lda, String x, int incx, float beta, String y, int incy) {
        cublasSgemv(trans, m, n, alpha, A, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * cublasSger (int m, int n, float alpha, const float *x, int incx, 
     *             const float *y, int incy, float *A, int lda)
     * 
     * performs the symmetric rank 1 operation
     * 
     *    A = alpha * x * transpose(y) + A,
     * 
     * where alpha is a single precision scalar, x is an m element single 
     * precision vector, y is an n element single precision vector, and A 
     * is an m by n matrix consisting of single precision elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     * 
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least 
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at 
     *        least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(y)
     * x      single precision array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      single precision array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not 
     *        be zero.
     * A      single precision array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     * 
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(y) + A
     * 
     * Reference: http://www.netlib.org/blas/sger.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSger(int m, int n, float alpha, String x, int offsetx, int incx, String y, int offsety, int incy, String A, int offsetA, int lda);

    public static void cublasSger(int m, int n, float alpha, String x, int incx, String y, int incy, String A, int lda) {
        cublasSger(m, n, alpha, x, 0, incx, y, 0, incy, A, 0, lda);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSsbmv (char uplo, int n, int k, float alpha, const float *A, int lda,
     *              const float *x, int incx, float beta, float *y, int incy)
     * 
     * performs the matrix-vector operation
     * 
     *     y := alpha*A*x + beta*y
     * 
     * alpha and beta are single precision scalars. x and y are single precision
     * vectors with n elements. A is an n x n symmetric band matrix consisting 
     * of single precision elements, with k super-diagonals and the same number
     * of sub-diagonals.
     * 
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the symmetric
     *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper 
     *        triangular part is being supplied. If uplo == 'L' or 'l', the lower 
     *        triangular part is being supplied.
     * n      specifies the number of rows and the number of columns of the
     *        symmetric matrix A. n must be at least zero.
     * k      specifies the number of super-diagonals of matrix A. Since the matrix
     *        is symmetric, this is also the number of sub-diagonals. k must be at
     *        least zero.
     * alpha  single precision scalar multiplier applied to A*x.
     * A      single precision array of dimensions (lda, n). When uplo == 'U' or 
     *        'u', the leading (k + 1) x n part of array A must contain the upper
     *        triangular band of the symmetric matrix, supplied column by column,
     *        with the leading diagonal of the matrix in row (k+1) of the array,
     *        the first super-diagonal starting at position 2 in row k, and so on.
     *        The top left k x k triangle of the array A is not referenced. When
     *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
     *        contain the lower triangular band part of the symmetric matrix, 
     *        supplied column by column, with the leading diagonal of the matrix in
     *        row 1 of the array, the first sub-diagonal starting at position 1 in
     *        row 2, and so on. The bottom right k x k triangle of the array A is
     *        not referenced.
     * lda    leading dimension of A. lda must be at least (k + 1).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y. If beta is 
     *        zero, y is not read.
     * y      single precision array of length at least (1 + (n - 1) * abs(incy)). 
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     * 
     * Output
     * ------
     * y      updated according to alpha*A*x + beta*y
     * 
     * Reference: http://www.netlib.org/blas/ssbmv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_INVALID_VALUE    if k or n &lt; 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSsbmv(char uplo, int n, int k, float alpha, String A, int offsetA, int lda, String x, int offsetx, int incx, float beta, String y, int offsety, int incy);

    public static void cublasSsbmv(char uplo, int n, int k, float alpha, String A, int lda, String x, int incx, float beta, String y, int incy) {
        cublasSsbmv(uplo, n, k, alpha, A, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSspmv (char uplo, int n, float alpha, const float *AP, const float *x,
     *              int incx, float beta, float *y, int incy)
     * 
     * performs the matrix-vector operation
     * 
     *    y = alpha * A * x + beta * y
     * 
     * Alpha and beta are single precision scalars, and x and y are single 
     * precision vectors with n elements. A is a symmetric n x n matrix 
     * consisting of single precision elements that is supplied in packed form.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper 
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then 
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to A*x.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &lt;= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If 
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &gt;= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y;
     * y      single precision array of length at least (1 + (n - 1) * abs(incy)). 
     *        If beta is zero, y is not read. 
     * incy   storage spacing between elements of y. incy must not be zero.
     * 
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     * 
     * Reference: http://www.netlib.org/blas/sspmv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSspmv(char uplo, int n, float alpha, String AP, int offsetAP, String x, int offsetx, int incx, float beta, String y, int offsety, int incy);

    public static void cublasSspmv(char uplo, int n, float alpha, String AP, String x, int incx, float beta, String y, int incy) {
        cublasSspmv(uplo, n, alpha, AP, 0, x, 0, incx, beta, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSspr (char uplo, int n, float alpha, const float *x, int incx, 
     *             float *AP)
     * 
     * performs the symmetric rank 1 operation
     * 
     *    A = alpha * x * transpose(x) + A,
     * 
     * where alpha is a single precision scalar and x is an n element single 
     * precision vector. A is a symmetric n x n matrix consisting of single 
     * precision elements that is supplied in packed form.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper 
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then 
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(x).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &lt;= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If 
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &gt;= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * 
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(x) + A
     * 
     * Reference: http://www.netlib.org/blas/sspr.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, or incx == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSspr(char uplo, int n, float alpha, String x, int offsetx, int incx, String AP, int offsetAP);

    public static void cublasSspr(char uplo, int n, float alpha, String x, int incx, String AP) {
        cublasSspr(uplo, n, alpha, x, 0, incx, AP, 0);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSspr2 (char uplo, int n, float alpha, const float *x, int incx, 
     *              const float *y, int incy, float *AP)
     * 
     * performs the symmetric rank 2 operation
     * 
     *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
     * 
     * where alpha is a single precision scalar, and x and y are n element single 
     * precision vectors. A is a symmetric n x n matrix consisting of single 
     * precision elements that is supplied in packed form.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the 
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower 
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(y) + 
     *        y * transpose(x).
     * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &lt;= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If 
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &gt;= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * 
     * Output
     * ------
     * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
     * 
     * Reference: http://www.netlib.org/blas/sspr2.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSspr2(char uplo, int n, float alpha, String x, int offsetx, int incx, String y, int offsety, int incy, String AP, int offsetAP);

    public static void cublasSspr2(char uplo, int n, float alpha, String x, int incx, String y, int incy, String AP) {
        cublasSspr2(uplo, n, alpha, x, 0, incx, y, 0, incy, AP, 0);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSsymv (char uplo, int n, float alpha, const float *A, int lda, 
     *              const float *x, int incx, float beta, float *y, int incy)
     * 
     * performs the matrix-vector operation
     * 
     *     y = alpha*A*x + beta*y
     * 
     * Alpha and beta are single precision scalars, and x and y are single 
     * precision vectors, each with n elements. A is a symmetric n x n matrix 
     * consisting of single precision elements that is stored in either upper or 
     * lower storage mode.
     * 
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the array A 
     *        is to be referenced. If uplo == 'U' or 'u', the symmetric matrix A 
     *        is stored in upper storage mode, i.e. only the upper triangular part
     *        of A is to be referenced while the lower triangular part of A is to 
     *        be inferred. If uplo == 'L' or 'l', the symmetric matrix A is stored
     *        in lower storage mode, i.e. only the lower triangular part of A is 
     *        to be referenced while the upper triangular part of A is to be 
     *        inferred.
     * n      specifies the number of rows and the number of columns of the 
     *        symmetric matrix A. n must be at least zero.
     * alpha  single precision scalar multiplier applied to A*x.
     * A      single precision array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular part of the symmetric matrix and the strictly
     *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
     *        the leading n x n lower triangular part of the array A must contain
     *        the lower triangular part of the symmetric matrix and the strictly
     *        upper triangular part of A is not referenced. 
     * lda    leading dimension of A. It must be at least max (1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y.
     * y      single precision array of length at least (1 + (n - 1) * abs(incy)). 
     *        If beta is zero, y is not read. 
     * incy   storage spacing between elements of y. incy must not be zero.
     * 
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     * 
     * Reference: http://www.netlib.org/blas/ssymv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSsymv(char uplo, int n, float alpha, String A, int offsetA, int lda, String x, int offsetx, int incx, float beta, String y, int offsety, int incy);

    public static void cublasSsymv(char uplo, int n, float alpha, String A, int lda, String x, int incx, float beta, String y, int incy) {
        cublasSsymv(uplo, n, alpha, A, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSsyr (char uplo, int n, float alpha, const float *x, int incx,
     *             float *A, int lda)
     * 
     * performs the symmetric rank 1 operation
     * 
     *    A = alpha * x * transpose(x) + A,
     * 
     * where alpha is a single precision scalar, x is an n element single 
     * precision vector and A is an n x n symmetric matrix consisting of 
     * single precision elements. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array 
     * containing A.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or 
     *        the lower triangular part of array A. If uplo = 'U' or 'u',
     *        then only the upper triangular part of A may be referenced.
     *        If uplo = 'L' or 'l', then only the lower triangular part of
     *        A may be referenced.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * alpha  single precision scalar multiplier applied to x * transpose(x)
     * x      single precision array of length at least (1 + (n - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must 
     *        not be zero.
     * A      single precision array of dimensions (lda, n). If uplo = 'U' or 
     *        'u', then A must contain the upper triangular part of a symmetric 
     *        matrix, and the strictly lower triangular part is not referenced. 
     *        If uplo = 'L' or 'l', then A contains the lower triangular part 
     *        of a symmetric matrix, and the strictly upper triangular part is 
     *        not referenced.
     * lda    leading dimension of the two-dimensional array containing A. lda
     *        must be at least max(1, n).
     * 
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(x) + A
     * 
     * Reference: http://www.netlib.org/blas/ssyr.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, or incx == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSsyr(char uplo, int n, float alpha, String x, int offsetx, int incx, String A, int offsetA, int lda);

    public static void cublasSsyr(char uplo, int n, float alpha, String x, int incx, String A, int lda) {
        cublasSsyr(uplo, n, alpha, x, 0, incx, A, 0, lda);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSsyr2 (char uplo, int n, float alpha, const float *x, int incx, 
     *              const float *y, int incy, float *A, int lda)
     * 
     * performs the symmetric rank 2 operation
     * 
     *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
     * 
     * where alpha is a single precision scalar, x and y are n element single 
     * precision vector and A is an n by n symmetric matrix consisting of single 
     * precision elements.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the 
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower 
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(y) + 
     *        y * transpose(x).
     * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * A      single precision array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        then A must contains the upper triangular part of a symmetric matrix,
     *        and the strictly lower triangular parts is not referenced. If uplo ==
     *        'L' or 'l', then A contains the lower triangular part of a symmetric 
     *        matrix, and the strictly upper triangular part is not referenced.
     * lda    leading dimension of A. It must be at least max(1, n).
     * 
     * Output
     * ------
     * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
     * 
     * Reference: http://www.netlib.org/blas/ssyr2.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSsyr2(char uplo, int n, float alpha, String x, int offsetx, int incx, String y, int offsety, int incy, String A, int offsetA, int lda);

    public static void cublasSsyr2(char uplo, int n, float alpha, String x, int incx, String y, int incy, String A, int lda) {
        cublasSsyr2(uplo, n, alpha, x, 0, incx, y, 0, incy, A, 0, lda);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasStbmv (char uplo, char trans, char diag, int n, int k, const float *A,
     *              int lda, float *x, int incx)
     * 
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A
     * or op(A) = transpose(A). x is an n-element single precision vector, and A is
     * an n x n, unit or non-unit upper or lower triangular band matrix consisting
     * of single precision elements.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular band
     *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero. In the current implementation n must not exceed 4070.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or 
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or 
     *        'l', k specifies the number of sub-diagonals. k must at least be 
     *        zero.
     * A      single precision array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper 
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first 
     *        super-diagonal starting at position 2 in row k, and so on. The top
     *        left k x k triangle of the array A is not referenced. If uplo == 'L'
     *        or 'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first 
     *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * lda    is the leading dimension of A. It must be at least (k + 1).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be 
     *        zero.
     * 
     * Output
     * ------
     * x      updated according to x = op(A) * x
     * 
     * Reference: http://www.netlib.org/blas/stbmv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, n &gt; 4070, k &lt; 0, or incx == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStbmv(char uplo, char trans, char diag, int n, int k, String A, int offsetA, int lda, String x, int offsetx, int incx);

    public static void cublasStbmv(char uplo, char trans, char diag, int n, int k, String A, int lda, String x, int incx) {
        cublasStbmv(uplo, trans, diag, n, k, A, 0, lda, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void cublasStbsv (char uplo, char trans, char diag, int n, int k,
     *                   const float *A, int lda, float *X, int incx)
     * 
     * solves one of the systems of equations op(A)*x = b, where op(A) is either 
     * op(A) = A or op(A) = transpose(A). b and x are n-element vectors, and A is
     * an n x n unit or non-unit, upper or lower triangular band matrix with k + 1
     * diagonals. No test for singularity or near-singularity is included in this
     * function. Such tests must be performed before calling this function.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular band 
     *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
     *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
     *        matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must be at least
     *        zero.
     * A      single precision array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first super-
     *        diagonal starting at position 2 in row k, and so on. The top left 
     *        k x k triangle of the array A is not referenced. If uplo == 'L' or 
     *        'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)). 
     *        On entry, x contains the n-element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   storage spacing between elements of x. incx must not be zero.
     * 
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     * 
     * Reference: http://www.netlib.org/blas/stbsv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n &lt; 0, or n &gt; 4070
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStbsv(char uplo, char trans, char diag, int n, int k, String A, int offsetA, int lda, String x, int offsetx, int incx);

    public static void cublasStbsv(char uplo, char trans, char diag, int n, int k, String A, int lda, String x, int incx) {
        cublasStbsv(uplo, trans, diag, n, k, A, 0, lda, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasStpmv (char uplo, char trans, char diag, int n, const float *AP, 
     *              float *x, int incx);
     * 
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * or op(A) = transpose(A). x is an n element single precision vector, and A 
     * is an n x n, unit or non-unit, upper or lower triangular matrix composed 
     * of single precision elements.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A 
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be 
     *        at least zero.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &lt;= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If 
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part 
     *        of the symmetric matrix A, packed sequentially, column by column; 
     *        that is, if i &gt;= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten 
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be 
     *        zero.
     * 
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     * 
     * Reference: http://www.netlib.org/blas/stpmv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStpmv(char uplo, char trans, char diag, int n, String AP, int offsetAP, String x, int offsetx, int incx);

    public static void cublasStpmv(char uplo, char trans, char diag, int n, String AP, String x, int incx) {
        cublasStpmv(uplo, trans, diag, n, AP, 0, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasStpsv (char uplo, char trans, char diag, int n, const float *AP,
     *              float *X, int incx)
     * 
     * solves one of the systems of equations op(A)*x = b, where op(A) is either 
     * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
     * an n x n unit or non-unit, upper or lower triangular matrix. No test for
     * singularity or near-singularity is included in this function. Such tests 
     * must be performed before calling this function.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular matrix
     *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero. In the current implementation n must not exceed 4070.
     * AP     single precision array with at least ((n*(n+1))/2) elements. If uplo
     *        == 'U' or 'u', the array AP contains the upper triangular matrix A,
     *        packed sequentially, column by column; that is, if i &lt;= j, then 
     *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the 
     *        array AP contains the lower triangular matrix A, packed sequentially,
     *        column by column; that is, if i &gt;= j, then A[i,j] is stored in 
     *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
     *        of A are not referenced and are assumed to be unity.
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)). 
     *        On entry, x contains the n-element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   storage spacing between elements of x. It must not be zero.
     * 
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     * 
     * Reference: http://www.netlib.org/blas/stpsv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n &lt; 0, or n &gt; 4070
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStpsv(char uplo, char trans, char diag, int n, String AP, int offsetAP, String x, int offsetx, int incx);

    public static void cublasStpsv(char uplo, char trans, char diag, int n, String AP, String x, int incx) {
        cublasStpsv(uplo, trans, diag, n, AP, 0, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasStrmv (char uplo, char trans, char diag, int n, const float *A,
     *              int lda, float *x, int incx);
     * 
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = 
     *      = A, or op(A) = transpose(A). x is an n-element single precision vector, and 
     * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed 
     * of single precision elements.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular 
     *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix. 
     *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa = 'N' or 'n', op(A) = A. If trans = 'T', 
     *        't', 'C', or 'c', op(A) = transpose(A)
     * diag   specifies whether or not matrix A is unit triangular. If diag = 'U' 
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A 
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be 
     *        at least zero.
     * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u', 
     *        the leading n x n upper triangular part of the array A must contain 
     *        the upper triangular matrix and the strictly lower triangular part 
     *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
     *        triangular part of the array A must contain the lower triangular 
     *        matrix and the strictly upper triangular part of A is not referenced.
     *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
     *        either, but are are assumed to be unity.
     * lda    is the leading dimension of A. It must be at least max (1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
     *        On entry, x contains the source vector. On exit, x is overwritten 
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be 
     *        zero.
     * 
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     * 
     * Reference: http://www.netlib.org/blas/strmv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStrmv(char uplo, char trans, char diag, int n, String A, int offsetA, int lda, String x, int offsetx, int incx);

    public static void cublasStrmv(char uplo, char trans, char diag, int n, String A, int lda, String x, int incx) {
        cublasStrmv(uplo, trans, diag, n, A, 0, lda, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasStrsv (char uplo, char trans, char diag, int n, const float *A,
     *              int lda, float *x, int incx)
     * 
     * solves a system of equations op(A) * x = b, where op(A) is either A or 
     * transpose(A). b and x are single precision vectors consisting of n
     * elements, and A is an n x n matrix composed of a unit or non-unit, upper
     * or lower triangular matrix. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-diemnsional array containing
     * A.
     * 
     * No test for singularity or near-singularity is included in this function. 
     * Such tests must be performed before calling this function.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the 
     *        lower triangular part of array A. If uplo = 'U' or 'u', then only 
     *        the upper triangular part of A may be referenced. If uplo = 'L' or 
     *        'l', then only the lower triangular part of A may be referenced.
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
     *        'T', 'c', or 'C', op(A) = transpose(A)
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If 
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0. In the current implementation n must be &lt;=
     *        4070.
     * A      is a single precision array of dimensions (lda, n). If uplo = 'U' 
     *        or 'u', then A must contains the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular parts is not referenced. 
     *        If uplo = 'L' or 'l', then A contains the lower triangular part of 
     *        a symmetric matrix, and the strictly upper triangular part is not 
     *        referenced. 
     * lda    is the leading dimension of the two-dimensional array containing A.
     *        lda must be at least max(1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   specifies the storage spacing between elements of x. incx must not 
     *        be zero.
     * 
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     * 
     * Reference: http://www.netlib.org/blas/strsv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n &lt; 0 or n &gt; 4070
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStrsv(char uplo, char trans, char diag, int n, String A, int offsetA, int lda, String x, int offsetx, int incx);

    public static void cublasStrsv(char uplo, char trans, char diag, int n, String A, int lda, String x, int incx) {
        cublasStrsv(uplo, trans, diag, n, A, 0, lda, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSgemm (char transa, char transb, int m, int n, int k, float alpha, 
     *              const float *A, int lda, const float *B, int ldb, float beta, 
     *              float *C, int ldc)
     * 
     * computes the product of matrix A and matrix B, multiplies the result 
     * by a scalar alpha, and adds the sum to the product of matrix C and
     * scalar beta. sgemm() performs one of the matrix-matrix operations:
     * 
     *     C = alpha * op(A) * op(B) + beta * C,
     * 
     * where op(X) is one of
     * 
     *     op(X) = X   or   op(X) = transpose(X)
     * 
     * alpha and beta are single precision scalars, and A, B and C are 
     * matrices consisting of single precision elements, with op(A) an m x k 
     * matrix, op(B) a k x n matrix, and C an m x n matrix. Matrices A, B, 
     * and C are stored in column major format, and lda, ldb, and ldc are
     * the leading dimensions of the two-dimensional arrays containing A, 
     * B, and C.
     * 
     * Input
     * -----
     * transa specifies op(A). If transa = 'n' or 'N', op(A) = A. If 
     *        transa = 't', 'T', 'c', or 'C', op(A) = transpose(A)
     * transb specifies op(B). If transb = 'n' or 'N', op(B) = B. If 
     *        transb = 't', 'T', 'c', or 'C', op(B) = transpose(B)
     * m      number of rows of matrix op(A) and rows of matrix C
     * n      number of columns of matrix op(B) and number of columns of C
     * k      number of columns of matrix op(A) and number of rows of op(B) 
     * alpha  single precision scalar multiplier applied to op(A)op(B)
     * A      single precision array of dimensions (lda, k) if transa = 
     *        'n' or 'N'), and of dimensions (lda, m) otherwise. When transa =
     *        'N' or 'n' then lda must be at least  max( 1, m ), otherwise lda
     *        must be at least max(1, k).
     * lda    leading dimension of two-dimensional array used to store matrix A
     * B      single precision array of dimensions  (ldb, n) if transb =
     *        'n' or 'N'), and of dimensions (ldb, k) otherwise. When transb =
     *        'N' or 'n' then ldb must be at least  max (1, k), otherwise ldb
     *        must be at least max (1, n).
     * ldb    leading dimension of two-dimensional array used to store matrix B
     * beta   single precision scalar multiplier applied to C. If 0, C does
     *        not have to be a valid input
     * C      single precision array of dimensions (ldc, n). ldc must be at 
     *        least max (1, m).
     * ldc    leading dimension of two-dimensional array used to store matrix C
     * 
     * Output
     * ------
     * C      updated based on C = alpha * op(A)*op(B) + beta * C
     * 
     * Reference: http://www.netlib.org/blas/sgemm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, float beta, String C, int offsetC, int ldc);

    public static void cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, String A, int lda, String B, int ldb, float beta, String C, int ldc) {
        cublasSgemm(transa, transb, m, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSsymm (char side, char uplo, int m, int n, float alpha, 
     *              const float *A, int lda, const float *B, int ldb, 
     *              float beta, float *C, int ldc);
     * 
     * performs one of the matrix-matrix operations
     * 
     *   C = alpha * A * B + beta * C, or 
     *   C = alpha * B * A + beta * C,
     * 
     * where alpha and beta are single precision scalars, A is a symmetric matrix
     * consisting of single precision elements and stored in either lower or upper 
     * storage mode, and B and C are m x n matrices consisting of single precision
     * elements.
     * 
     * Input
     * -----
     * side   specifies whether the symmetric matrix A appears on the left side 
     *        hand side or right hand side of matrix B, as follows. If side == 'L' 
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r', 
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the symmetric matrix A is stored in upper or lower 
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper 
     *        triangular part of the symmetric matrix is to be referenced, and the 
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the 
     *        lower triangular part of the symmetric matrix is to be referenced, 
     *        and the elements of the strictly upper triangular part are to be 
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of symmetric matrix A 
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of 
     *        columns of matrix B. It also specifies the dimensions of symmetric 
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  single precision scalar multiplier applied to A * B, or B * A
     * A      single precision array of dimensions (lda, ka), where ka is m when 
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the 
     *        leading m x m part of array A must contain the symmetric matrix, 
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the 
     *        upper triangular part of the symmetric matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u', 
     *        the leading m x m part stores the lower triangular part of the 
     *        symmetric matrix and the strictly upper triangular part is not 
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A 
     *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the 
     *        symmetric matrix and the strictly lower triangular part of A is not 
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part 
     *        stores the lower triangular part of the symmetric matrix and the 
     *        strictly upper triangular part is not referenced.
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least 
     *        max(1, m) and at least max(1, n) otherwise.
     * B      single precision array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   single precision scalar multiplier applied to C. If beta is zero, C 
     *        does not have to be a valid input
     * C      single precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     * 
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha * 
     *        B * A + beta * C
     * 
     * Reference: http://www.netlib.org/blas/ssymm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSsymm(char side, char uplo, int m, int n, float alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, float beta, String C, int offsetC, int ldc);

    public static void cublasSsymm(char side, char uplo, int m, int n, float alpha, String A, int lda, String B, int ldb, float beta, String C, int ldc) {
        cublasSsymm(side, uplo, m, n, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSsyrk (char uplo, char trans, int n, int k, float alpha, 
     *              const float *A, int lda, float beta, float *C, int ldc)
     * 
     * performs one of the symmetric rank k operations
     * 
     *   C = alpha * A * transpose(A) + beta * C, or 
     *   C = alpha * transpose(A) * A + beta * C.
     * 
     * Alpha and beta are single precision scalars. C is an n x n symmetric matrix 
     * consisting of single precision elements and stored in either lower or 
     * upper storage mode. A is a matrix consisting of single precision elements
     * with dimension of n x k in the first case, and k x n in the second case.
     * 
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower 
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper 
     *        triangular part of the symmetric matrix is to be referenced, and the 
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the 
     *        lower triangular part of the symmetric matrix is to be referenced, 
     *        and the elements of the strictly upper triangular part are to be 
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C = 
     *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c', 
     *        C = transpose(A) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If 
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If 
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A. 
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A. 
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of 
     *        matrix A. k must be at least zero.
     * alpha  single precision scalar multiplier applied to A * transpose(A) or 
     *        transpose(A) * A.
     * A      single precision array of dimensions (lda, ka), where ka is k when 
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n', 
     *        the leading n x k part of array A must contain the matrix A, 
     *        otherwise the leading k x n part of the array must contains the 
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   single precision scalar multiplier applied to C. If beta izs zero, C
     *        does not have to be a valid input
     * C      single precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the 
     *        upper triangular part of the symmetric matrix C and the strictly 
     *        lower triangular part of C is not referenced. On exit, the upper 
     *        triangular part of C is overwritten by the upper trinagular part of 
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n 
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is 
     *        overwritten by the lower trinagular part of the updated matrix.
     * ldc    leading dimension of C. It must be at least max(1, n).
     * 
     * Output
     * ------
     * C      updated according to C = alpha * A * transpose(A) + beta * C, or C = 
     *        alpha * transpose(A) * A + beta * C
     * 
     * Reference: http://www.netlib.org/blas/ssyrk.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0 or k &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSsyrk(char uplo, char trans, int n, int k, float alpha, String A, int offsetA, int lda, float beta, String C, int offsetC, int ldc);

    public static void cublasSsyrk(char uplo, char trans, int n, int k, float alpha, String A, int lda, float beta, String C, int ldc) {
        cublasSsyrk(uplo, trans, n, k, alpha, A, 0, lda, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasSsyr2k (char uplo, char trans, int n, int k, float alpha, 
     *               const float *A, int lda, const float *B, int ldb, 
     *               float beta, float *C, int ldc)
     * 
     * performs one of the symmetric rank 2k operations
     * 
     *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or 
     *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
     * 
     * Alpha and beta are single precision scalars. C is an n x n symmetric matrix
     * consisting of single precision elements and stored in either lower or upper 
     * storage mode. A and B are matrices consisting of single precision elements 
     * with dimension of n x k in the first case, and k x n in the second case.
     * 
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper 
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the 
     *        lower triangular part of the symmetric matrix is to be references, 
     *        and the elements of the strictly upper triangular part are to be 
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', 
     *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, 
     *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B + 
     *        alpha * transpose(B) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If 
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If 
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A. 
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A. 
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of 
     *        matrix A. k must be at least zero.
     * alpha  single precision scalar multiplier.
     * A      single precision array of dimensions (lda, ka), where ka is k when 
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n', 
     *        the leading n x k part of array A must contain the matrix A, 
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at 
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      single precision array of dimensions (lda, kb), where kb is k when 
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n', 
     *        the leading n x k part of array B must contain the matrix B, 
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   single precision scalar multiplier applied to C. If beta is zero, C 
     *        does not have to be a valid input.
     * C      single precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the 
     *        upper triangular part of the symmetric matrix C and the strictly 
     *        lower triangular part of C is not referenced. On exit, the upper 
     *        triangular part of C is overwritten by the upper trinagular part of 
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n 
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is 
     *        overwritten by the lower trinagular part of the updated matrix.
     * ldc    leading dimension of C. Must be at least max(1, n).
     * 
     * Output
     * ------
     * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) + 
     *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
     * 
     * Reference:   http://www.netlib.org/blas/ssyr2k.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0 or k &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, float beta, String C, int offsetC, int ldc);

    public static void cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, String A, int lda, String B, int ldb, float beta, String C, int ldc) {
        cublasSsyr2k(uplo, trans, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasStrmm (char side, char uplo, char transa, char diag, int m, int n, 
     *              float alpha, const float *A, int lda, const float *B, int ldb)
     * 
     * performs one of the matrix-matrix operations
     * 
     *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
     * 
     * where alpha is a single-precision scalar, B is an m x n matrix composed
     * of single precision elements, and A is a unit or non-unit, upper or lower, 
     * triangular matrix composed of single precision elements. op(A) is one of
     * 
     *   op(A) = A  or  op(A) = transpose(A)
     * 
     * Matrices A and B are stored in column major format, and lda and ldb are 
     * the leading dimensions of the two-dimensonials arrays that contain A and 
     * B, respectively.
     * 
     * Input
     * -----
     * side   specifies whether op(A) multiplies B from the left or right.
     *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
     *        'R' or 'r', then B = alpha * B * op(A).
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', A is a lower triangular matrix.
     * transa specifies the form of op(A) to be used in the matrix 
     *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
     *        transa = 'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
     *        'n', A is not assumed to be unit triangular.
     * m      the number of rows of matrix B. m must be at least zero.
     * n      the number of columns of matrix B. n must be at least zero.
     * alpha  single precision scalar multiplier applied to op(A)*B, or
     *        B*op(A), respectively. If alpha is zero no accesses are made
     *        to matrix A, and no read accesses are made to matrix B.
     * A      single precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     *        triangular part of A is not referenced. When diag = 'U' or 'u'
     *        the diagonal elements of A are no referenced and are assumed
     *        to be unity.
     * lda    leading dimension of A. When side = 'L' or 'l', it must be at
     *        least max(1,m) and at least max(1,n) otherwise
     * B      single precision array of dimensions (ldb, n). On entry, the 
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * 
     * Output
     * ------
     * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
     * 
     * Reference: http://www.netlib.org/blas/strmm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStrmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb);

    public static void cublasStrmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, String A, int lda, String B, int ldb) {
        cublasStrmm(side, uplo, transa, diag, m, n, alpha, A, 0, lda, B, 0, ldb);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasStrsm (char side, char uplo, char transa, char diag, int m, int n, 
     *              float alpha, const float *A, int lda, float *B, int ldb)
     * 
     * solves one of the matrix equations
     * 
     *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
     * 
     * where alpha is a single precision scalar, and X and B are m x n matrices 
     * that are composed of single precision elements. A is a unit or non-unit,
     * upper or lower triangular matrix, and op(A) is one of 
     * 
     *    op(A) = A  or  op(A) = transpose(A)
     * 
     * The result matrix X overwrites input matrix B; that is, on exit the result 
     * is stored in B. Matrices A and B are stored in column major format, and
     * lda and ldb are the leading dimensions of the two-dimensonials arrays that
     * contain A and B, respectively.
     * 
     * Input
     * -----
     * side   specifies whether op(A) appears on the left or right of X as
     *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
     *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
     *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
     *        triangular matrix.
     * transa specifies the form of op(A) to be used in matrix multiplication
     *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
     *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If 
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * m      specifies the number of rows of B. m must be at least zero.
     * n      specifies the number of columns of B. n must be at least zero.
     * alpha  is a single precision scalar to be multiplied with B. When alpha is 
     *        zero, then A is not referenced and B need not be set before entry.
     * A      is a single precision array of dimensions (lda, k), where k is
     *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
     *        uplo = 'U' or 'u', the leading k x k upper triangular part of
     *        the array A must contain the upper triangular matrix and the
     *        strictly lower triangular matrix of A is not referenced. When
     *        uplo = 'L' or 'l', the leading k x k lower triangular part of
     *        the array A must contain the lower triangular matrix and the 
     *        strictly upper triangular part of A is not referenced. Note that
     *        when diag = 'U' or 'u', the diagonal elements of A are not
     *        referenced, and are assumed to be unity.
     * lda    is the leading dimension of the two dimensional array containing A.
     *        When side = 'L' or 'l' then lda must be at least max(1, m), when 
     *        side = 'R' or 'r' then lda must be at least max(1, n).
     * B      is a single precision array of dimensions (ldb, n). ldb must be
     *        at least max (1,m). The leading m x n part of the array B must 
     *        contain the right-hand side matrix B. On exit B is overwritten 
     *        by the solution matrix X.
     * ldb    is the leading dimension of the two dimensional array containing B.
     *        ldb must be at least max(1, m).
     * 
     * Output
     * ------
     * B      contains the solution matrix X satisfying op(A) * X = alpha * B, 
     *        or X * op(A) = alpha * B
     * 
     * Reference: http://www.netlib.org/blas/strsm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb);

    public static void cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, String A, int lda, String B, int ldb) {
        cublasStrsm(side, uplo, transa, diag, m, n, alpha, A, 0, lda, B, 0, ldb);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void cublasCgemm (char transa, char transb, int m, int n, int k, 
     *                   cuComplex alpha, const cuComplex *A, int lda, 
     *                   const cuComplex *B, int ldb, cuComplex beta, 
     *                   cuComplex *C, int ldc)
     * 
     * performs one of the matrix-matrix operations
     * 
     *    C = alpha * op(A) * op(B) + beta*C,
     * 
     * where op(X) is one of
     * 
     *    op(X) = X   or   op(X) = transpose  or  op(X) = conjg(transpose(X))
     * 
     * alpha and beta are single-complex scalars, and A, B and C are matrices
     * consisting of single-complex elements, with op(A) an m x k matrix, op(B)
     * a k x n matrix and C an m x n matrix.
     * 
     * Input
     * -----
     * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. If transa == 
     *        'T' or 't', op(A) = transpose(A). If transa == 'C' or 'c', op(A) = 
     *        conjg(transpose(A)).
     * transb specifies op(B). If transa == 'N' or 'n', op(B) = B. If transb == 
     *        'T' or 't', op(B) = transpose(B). If transb == 'C' or 'c', op(B) = 
     *        conjg(transpose(B)).
     * m      number of rows of matrix op(A) and rows of matrix C. It must be at
     *        least zero.
     * n      number of columns of matrix op(B) and number of columns of C. It 
     *        must be at least zero.
     * k      number of columns of matrix op(A) and number of rows of op(B). It 
     *        must be at least zero.
     * alpha  single-complex scalar multiplier applied to op(A)op(B)
     * A      single-complex array of dimensions (lda, k) if transa ==  'N' or 
     *        'n'), and of dimensions (lda, m) otherwise.
     * lda    leading dimension of A. When transa == 'N' or 'n', it must be at 
     *        least max(1, m) and at least max(1, k) otherwise.
     * B      single-complex array of dimensions (ldb, n) if transb == 'N' or 'n', 
     *        and of dimensions (ldb, k) otherwise
     * ldb    leading dimension of B. When transb == 'N' or 'n', it must be at 
     *        least max(1, k) and at least max(1, n) otherwise.
     * beta   single-complex scalar multiplier applied to C. If beta is zero, C 
     *        does not have to be a valid input.
     * C      single precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m).
     * 
     * Output
     * ------
     * C      updated according to C = alpha*op(A)*op(B) + beta*C
     * 
     * Reference: http://www.netlib.org/blas/cgemm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are &lt; 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasCgemm(char transa, char transb, int m, int n, int k, JCuComplex alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, JCuComplex beta, String C, int offsetC, int ldc);

    public static void cublasCgemm(char transa, char transb, int m, int n, int k, JCuComplex alpha, String A, int lda, String B, int ldb, JCuComplex beta, String C, int ldc) {
        cublasCgemm(transa, transb, m, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasDaxpy (int n, double alpha, const double *x, int incx, double *y, 
     *              int incy)
     * 
     * multiplies double-precision vector x by double-precision scalar alpha 
     * and adds the result to double-precision vector y; that is, it overwrites 
     * double-precision y with double-precision alpha * x + y. For i = 0 to n-1,
     * it replaces y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i*incy],
     * where lx = 1 if incx &gt;= 0, else lx = 1 + (1 - n) * incx; ly is defined in a 
     * similar way using incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  double-precision scalar multiplier
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     * 
     * Output
     * ------
     * y      double-precision result (unchanged if n &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/daxpy.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library was not initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDaxpy(int n, double alpha, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasDaxpy(int n, double alpha, String x, int incx, String y, int incy) {
        cublasDaxpy(n, alpha, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDcopy (int n, const double *x, int incx, double *y, int incy)
     * 
     * copies the double-precision vector x to the double-precision vector y. For 
     * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar 
     * way using incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     * 
     * Output
     * ------
     * y      contains double precision vector x
     * 
     * Reference: http://www.netlib.org/blas/dcopy.f
     * 
     * Error status for this function can be retrieved via cublasGetError(). 
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDcopy(int n, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasDcopy(int n, String x, int incx, String y, int incy) {
        cublasDcopy(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDrot (int n, double *x, int incx, double *y, int incy, double sc, 
     *             double ss)
     * 
     * multiplies a 2x2 matrix ( sc ss) with the 2xn matrix ( transpose(x) )
     *                         (-ss sc)                     ( transpose(y) )
     * 
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if 
     * incx &gt;= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and 
     * incy.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     * sc     element of rotation matrix
     * ss     element of rotation matrix
     * 
     * Output
     * ------
     * x      rotated vector x (unchanged if n &lt;= 0)
     * y      rotated vector y (unchanged if n &lt;= 0)
     * 
     * Reference  http://www.netlib.org/blas/drot.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDrot(int n, String x, int offsetx, int incx, String y, int offsety, int incy, double sc, double ss);

    public static void cublasDrot(int n, String x, int incx, String y, int incy, double sc, double ss) {
        cublasDrot(n, x, 0, incx, y, 0, incy, sc, ss);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDrotg (double *sa, double *sb, double *sc, double *ss)
     * 
     * constructs the Givens tranformation
     * 
     *        ( sc  ss )
     *    G = (        ) ,  sc&circ;2 + ss&circ;2 = 1,
     *        (-ss  sc )
     * 
     * which zeros the second entry of the 2-vector transpose(sa, sb).
     * 
     * The quantity r = (+/-) sqrt (sa&circ;2 + sb&circ;2) overwrites sa in storage. The 
     * value of sb is overwritten by a value z which allows sc and ss to be 
     * recovered by the following algorithm:
     * 
     *    if z=1          set sc = 0.0 and ss = 1.0
     *    if abs(z) &lt; 1   set sc = sqrt(1-z&circ;2) and ss = z
     *    if abs(z) &gt; 1   set sc = 1/z and ss = sqrt(1-sc&circ;2)
     * 
     * The function drot (n, x, incx, y, incy, sc, ss) normally is called next
     * to apply the transformation to a 2 x n matrix.
     * 
     * Input
     * -----
     * sa     double-precision scalar
     * sb     double-precision scalar
     * 
     * Output
     * ------
     * sa     double-precision r
     * sb     double-precision z
     * sc     double-precision result
     * ss     double-precision result
     * 
     * Reference: http://www.netlib.org/blas/drotg.f
     * 
     * This function does not set any error status.
     * </pre>
     */

    public static native void cublasDrotg(String sa, int offsetsa, String sb, int offsetsb, String sc, int offsetsc, String ss, int offsetss);

    public static void cublasDrotg(String sa, String sb, String sc, String ss) {
        cublasDrotg(sa, 0, sb, 0, sc, 0, ss, 0);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasDscal (int n, double alpha, double *x, int incx)
     * 
     * replaces double-precision vector x with double-precision alpha * x. For 
     * i = 0 to n-1, it replaces x[lx + i * incx] with alpha * x[lx + i * incx],
     * where lx = 1 if incx &gt;= 0, else lx = 1 + (1 - n) * incx.
     * 
     * Input
     * -----
     * n      number of elements in input vector
     * alpha  double-precision scalar multiplier
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * x      double-precision result (unchanged if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/dscal.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library was not initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDscal(int n, double alpha, String x, int offsetx, int incx);

    public static void cublasDscal(int n, double alpha, String x, int incx) {
        cublasDscal(n, alpha, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasDswap (int n, double *x, int incx, double *y, int incy)
     * 
     * replaces double-precision vector x with double-precision alpha * x. For i 
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
     * where ix = 1 if incx &gt;= 0, else ix = 1 + (1 - n) * incx.
     * 
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  double-precision scalar multiplier
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * x      double precision result (unchanged if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/dswap.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDswap(int n, String x, int offsetx, int incx, String y, int offsety, int incy);

    public static void cublasDswap(int n, String x, int incx, String y, int incy) {
        cublasDswap(n, x, 0, incx, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * int 
     * idamax (int n, const double *x, int incx)
     * 
     * finds the smallest index of the maximum magnitude element of double-
     * precision vector x; that is, the result is the first i, i = 0 to n - 1, 
     * that maximizes abs(x[1 + i * incx])).
     * 
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * returns the smallest index (0 if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/blas/idamax.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native int cublasIdamax(int n, String x, int offsetx, int incx);

    public static int cublasIdamax(int n, String x, int incx) {
        return cublasIdamax(n, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * int 
     * idamin (int n, const double *x, int incx)
     * 
     * finds the smallest index of the minimum magnitude element of double-
     * precision vector x; that is, the result is the first i, i = 0 to n - 1, 
     * that minimizes abs(x[1 + i * incx])).
     * 
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * 
     * Output
     * ------
     * returns the smallest index (0 if n &lt;= 0 or incx &lt;= 0)
     * 
     * Reference: http://www.netlib.org/scilib/blass.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native int cublasIdamin(int n, String x, int offsetx, int incx);

    public static int cublasIdamin(int n, String x, int incx) {
        return cublasIdamin(n, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * cublasDgemv (char trans, int m, int n, double alpha, const double *A, 
     *              int lda, const double *x, int incx, double beta, double *y, 
     *              int incy)
     * 
     * performs one of the matrix-vector operations
     * 
     *    y = alpha * op(A) * x + beta * y,
     * 
     * where op(A) is one of
     * 
     *    op(A) = A   or   op(A) = transpose(A)
     * 
     * where alpha and beta are double precision scalars, x and y are double 
     * precision vectors, and A is an m x n matrix consisting of double precision
     * elements. Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array in which A is stored.
     * 
     * Input
     * -----
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
     *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least 
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least 
     *        zero.
     * alpha  double precision scalar multiplier applied to op(A).
     * A      double precision array of dimensions (lda, n) if trans = 'n' or 
     *        'N'), and of dimensions (lda, m) otherwise. lda must be at least 
     *        max(1, m) and at least max(1, n) otherwise.
     * lda    leading dimension of two-dimensional array used to store matrix A
     * x      double precision array of length at least (1 + (n - 1) * abs(incx))
     *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx)) 
     *        otherwise.
     * incx   specifies the storage spacing between elements of x. incx must not 
     *        be zero.
     * beta   double precision scalar multiplier applied to vector y. If beta 
     *        is zero, y is not read.
     * y      double precision array of length at least (1 + (m - 1) * abs(incy))
     *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy)) 
     *        otherwise.
     * incy   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * 
     * Output
     * ------
     * y      updated according to alpha * op(A) * x + beta * y
     * 
     * Reference: http://www.netlib.org/blas/dgemv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are &lt; 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDgemv(char trans, int m, int n, double alpha, String A, int offsetA, int lda, String x, int offsetx, int incx, double beta, String y, int offsety, int incy);

    public static void cublasDgemv(char trans, int m, int n, double alpha, String A, int lda, String x, int incx, double beta, String y, int incy) {
        cublasDgemv(trans, m, n, alpha, A, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * cublasDger (int m, int n, double alpha, const double *x, int incx,
     *             const double *y, int incy, double *A, int lda)
     * 
     * performs the symmetric rank 1 operation
     * 
     *    A = alpha * x * transpose(y) + A,
     * 
     * where alpha is a double precision scalar, x is an m element double
     * precision vector, y is an n element double precision vector, and A
     * is an m by n matrix consisting of double precision elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     * 
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at
     *        least zero.
     * alpha  double precision scalar multiplier applied to x * transpose(y)
     * x      double precision array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      double precision array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     * A      double precision array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     * 
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(y) + A
     * 
     * Reference: http://www.netlib.org/blas/dger.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDger(int m, int n, double alpha, String x, int offsetx, int incx, String y, int offsety, int incy, String A, int offsetA, int lda);

    public static void cublasDger(int m, int n, double alpha, String x, int incx, String y, int incy, String A, int lda) {
        cublasDger(m, n, alpha, x, 0, incx, y, 0, incy, A, 0, lda);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDsyr (char uplo, int n, double alpha, const double *x, int incx, 
     *             double *A, int lda)
     * 
     * performs the symmetric rank 1 operation
     * 
     *    A = alpha * x * transpose(x) + A,
     * 
     * where alpha is a double precision scalar, x is an n element double 
     * precision vector and A is an n x n symmetric matrix consisting of 
     * double precision elements. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array 
     * containing A.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or 
     *        the lower triangular part of array A. If uplo = 'U' or 'u',
     *        then only the upper triangular part of A may be referenced.
     *        If uplo = 'L' or 'l', then only the lower triangular part of
     *        A may be referenced.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * alpha  double precision scalar multiplier applied to x * transpose(x)
     * x      double precision array of length at least (1 + (n - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must 
     *        not be zero.
     * A      double precision array of dimensions (lda, n). If uplo = 'U' or 
     *        'u', then A must contain the upper triangular part of a symmetric 
     *        matrix, and the strictly lower triangular part is not referenced. 
     *        If uplo = 'L' or 'l', then A contains the lower triangular part 
     *        of a symmetric matrix, and the strictly upper triangular part is 
     *        not referenced.
     * lda    leading dimension of the two-dimensional array containing A. lda
     *        must be at least max(1, n).
     * 
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(x) + A
     * 
     * Reference: http://www.netlib.org/blas/dsyr.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0, or incx == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDsyr(char uplo, int n, double alpha, String x, int offsetx, int incx, String A, int offsetA, int lda);

    public static void cublasDsyr(char uplo, int n, double alpha, String x, int incx, String A, int lda) {
        cublasDsyr(uplo, n, alpha, x, 0, incx, A, 0, lda);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDtrsv (char uplo, char trans, char diag, int n, const double *A, 
     *              int lda, double *x, int incx)
     * 
     * solves a system of equations op(A) * x = b, where op(A) is either A or 
     * transpose(A). b and x are double precision vectors consisting of n
     * elements, and A is an n x n matrix composed of a unit or non-unit, upper
     * or lower triangular matrix. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-diemnsional array containing
     * A.
     * 
     * No test for singularity or near-singularity is included in this function. 
     * Such tests must be performed before calling this function.
     * 
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the 
     *        lower triangular part of array A. If uplo = 'U' or 'u', then only 
     *        the upper triangular part of A may be referenced. If uplo = 'L' or 
     *        'l', then only the lower triangular part of A may be referenced.
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
     *        'T', 'c', or 'C', op(A) = transpose(A)
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If 
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0. In the current implementation n must be &lt;=
     *        2040.
     * A      is a double precision array of dimensions (lda, n). If uplo = 'U' 
     *        or 'u', then A must contains the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular parts is not referenced. 
     *        If uplo = 'L' or 'l', then A contains the lower triangular part of 
     *        a symmetric matrix, and the strictly upper triangular part is not 
     *        referenced. 
     * lda    is the leading dimension of the two-dimensional array containing A.
     *        lda must be at least max(1, n).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   specifies the storage spacing between elements of x. incx must not 
     *        be zero.
     * 
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     * 
     * Reference: http://www.netlib.org/blas/dtrsv.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n &lt; 0 or n &gt; 2040
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDtrsv(char uplo, char trans, char diag, int n, String A, int offsetA, int lda, String x, int offsetx, int incx);

    public static void cublasDtrsv(char uplo, char trans, char diag, int n, String A, int lda, String x, int incx) {
        cublasDtrsv(uplo, trans, diag, n, A, 0, lda, x, 0, incx);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDgemm (char transa, char transb, int m, int n, int k, double alpha,
     *              const double *A, int lda, const double *B, int ldb, 
     *              double beta, double *C, int ldc)
     * 
     * computes the product of matrix A and matrix B, multiplies the result 
     * by scalar alpha, and adds the sum to the product of matrix C and
     * scalar beta. It performs one of the matrix-matrix operations:
     * 
     * C = alpha * op(A) * op(B) + beta * C,  
     * where op(X) = X or op(X) = transpose(X),
     * 
     * and alpha and beta are double-precision scalars. A, B and C are matrices
     * consisting of double-precision elements, with op(A) an m x k matrix, 
     * op(B) a k x n matrix, and C an m x n matrix. Matrices A, B, and C are 
     * stored in column-major format, and lda, ldb, and ldc are the leading 
     * dimensions of the two-dimensional arrays containing A, B, and C.
     * 
     * Input
     * -----
     * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. 
     *        If transa == 'T', 't', 'C', or 'c', op(A) = transpose(A).
     * transb specifies op(B). If transb == 'N' or 'n', op(B) = B. 
     *        If transb == 'T', 't', 'C', or 'c', op(B) = transpose(B).
     * m      number of rows of matrix op(A) and rows of matrix C; m must be at
     *        least zero.
     * n      number of columns of matrix op(B) and number of columns of C; 
     *        n must be at least zero.
     * k      number of columns of matrix op(A) and number of rows of op(B);
     *        k must be at least zero.
     * alpha  double-precision scalar multiplier applied to op(A) * op(B).
     * A      double-precision array of dimensions (lda, k) if transa == 'N' or 
     *        'n', and of dimensions (lda, m) otherwise. If transa == 'N' or 
     *        'n' lda must be at least max(1, m), otherwise lda must be at
     *        least max(1, k).
     * lda    leading dimension of two-dimensional array used to store matrix A.
     * B      double-precision array of dimensions (ldb, n) if transb == 'N' or
     *        'n', and of dimensions (ldb, k) otherwise. If transb == 'N' or 
     *        'n' ldb must be at least max (1, k), otherwise ldb must be at
     *        least max(1, n).
     * ldb    leading dimension of two-dimensional array used to store matrix B.
     * beta   double-precision scalar multiplier applied to C. If zero, C does not 
     *        have to be a valid input
     * C      double-precision array of dimensions (ldc, n); ldc must be at least
     *        max(1, m).
     * ldc    leading dimension of two-dimensional array used to store matrix C.
     * 
     * Output
     * ------
     * C      updated based on C = alpha * op(A)*op(B) + beta * C.
     * 
     * Reference: http://www.netlib.org/blas/sgemm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS was not initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m &lt; 0, n &lt; 0, or k &lt; 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDgemm(char transa, char transb, int m, int n, int k, double alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, double beta, String C, int offsetC, int ldc);

    public static void cublasDgemm(char transa, char transb, int m, int n, int k, double alpha, String A, int lda, String B, int ldb, double beta, String C, int ldc) {
        cublasDgemm(transa, transb, m, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasDtrsm (char side, char uplo, char transa, char diag, int m, int n,
     *              double alpha, const double *A, int lda, double *B, int ldb)
     * 
     * solves one of the matrix equations
     * 
     *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
     * 
     * where alpha is a double precision scalar, and X and B are m x n matrices
     * that are composed of double precision elements. A is a unit or non-unit,
     * upper or lower triangular matrix, and op(A) is one of
     * 
     *    op(A) = A  or  op(A) = transpose(A)
     * 
     * The result matrix X overwrites input matrix B; that is, on exit the result
     * is stored in B. Matrices A and B are stored in column major format, and
     * lda and ldb are the leading dimensions of the two-dimensonials arrays that
     * contain A and B, respectively.
     * 
     * Input
     * -----
     * side   specifies whether op(A) appears on the left or right of X as
     *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
     *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
     *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
     *        triangular matrix.
     * transa specifies the form of op(A) to be used in matrix multiplication
     *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
     *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * m      specifies the number of rows of B. m must be at least zero.
     * n      specifies the number of columns of B. n must be at least zero.
     * alpha  is a double precision scalar to be multiplied with B. When alpha is
     *        zero, then A is not referenced and B need not be set before entry.
     * A      is a double precision array of dimensions (lda, k), where k is
     *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
     *        uplo = 'U' or 'u', the leading k x k upper triangular part of
     *        the array A must contain the upper triangular matrix and the
     *        strictly lower triangular matrix of A is not referenced. When
     *        uplo = 'L' or 'l', the leading k x k lower triangular part of
     *        the array A must contain the lower triangular matrix and the
     *        strictly upper triangular part of A is not referenced. Note that
     *        when diag = 'U' or 'u', the diagonal elements of A are not
     *        referenced, and are assumed to be unity.
     * lda    is the leading dimension of the two dimensional array containing A.
     *        When side = 'L' or 'l' then lda must be at least max(1, m), when
     *        side = 'R' or 'r' then lda must be at least max(1, n).
     * B      is a double precision array of dimensions (ldb, n). ldb must be
     *        at least max (1,m). The leading m x n part of the array B must
     *        contain the right-hand side matrix B. On exit B is overwritten
     *        by the solution matrix X.
     * ldb    is the leading dimension of the two dimensional array containing B.
     *        ldb must be at least max(1, m).
     * 
     * Output
     * ------
     * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
     *        or X * op(A) = alpha * B
     * 
     * Reference: http://www.netlib.org/blas/dtrsm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n &lt; 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb);

    public static void cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, String A, int lda, String B, int ldb) {
        cublasDtrsm(side, uplo, transa, diag, m, n, alpha, A, 0, lda, B, 0, ldb);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDtrmm (char side, char uplo, char transa, char diag, int m, int n, 
     *              double alpha, const double *A, int lda, const double *B, int ldb)
     * 
     * performs one of the matrix-matrix operations
     * 
     *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
     * 
     * where alpha is a double-precision scalar, B is an m x n matrix composed
     * of double precision elements, and A is a unit or non-unit, upper or lower, 
     * triangular matrix composed of double precision elements. op(A) is one of
     * 
     *   op(A) = A  or  op(A) = transpose(A)
     * 
     * Matrices A and B are stored in column major format, and lda and ldb are 
     * the leading dimensions of the two-dimensonials arrays that contain A and 
     * B, respectively.
     * 
     * Input
     * -----
     * side   specifies whether op(A) multiplies B from the left or right.
     *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
     *        'R' or 'r', then B = alpha * B * op(A).
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', A is a lower triangular matrix.
     * transa specifies the form of op(A) to be used in the matrix 
     *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
     *        transa = 'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
     *        'n', A is not assumed to be unit triangular.
     * m      the number of rows of matrix B. m must be at least zero.
     * n      the number of columns of matrix B. n must be at least zero.
     * alpha  double precision scalar multiplier applied to op(A)*B, or
     *        B*op(A), respectively. If alpha is zero no accesses are made
     *        to matrix A, and no read accesses are made to matrix B.
     * A      double precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     *        triangular part of A is not referenced. When diag = 'U' or 'u'
     *        the diagonal elements of A are no referenced and are assumed
     *        to be unity.
     * lda    leading dimension of A. When side = 'L' or 'l', it must be at
     *        least max(1,m) and at least max(1,n) otherwise
     * B      double precision array of dimensions (ldb, n). On entry, the 
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * 
     * Output
     * ------
     * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
     * 
     * Reference: http://www.netlib.org/blas/dtrmm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n &lt; 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb);

    public static void cublasDtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, String A, int lda, String B, int ldb) {
        cublasDtrmm(side, uplo, transa, diag, m, n, alpha, A, 0, lda, B, 0, ldb);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasDsymm (char side, char uplo, int m, int n, double alpha,
     *              const double *A, int lda, const double *B, int ldb,
     *              double beta, double *C, int ldc);
     * 
     * performs one of the matrix-matrix operations
     * 
     *   C = alpha * A * B + beta * C, or
     *   C = alpha * B * A + beta * C,
     * 
     * where alpha and beta are double precision scalars, A is a symmetric matrix
     * consisting of double precision elements and stored in either lower or upper
     * storage mode, and B and C are m x n matrices consisting of double precision
     * elements.
     * 
     * Input
     * -----
     * side   specifies whether the symmetric matrix A appears on the left side
     *        hand side or right hand side of matrix B, as follows. If side == 'L'
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the symmetric matrix A is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of symmetric matrix A
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of
     *        columns of matrix B. It also specifies the dimensions of symmetric
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  double precision scalar multiplier applied to A * B, or B * A
     * A      double precision array of dimensions (lda, ka), where ka is m when
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
     *        leading m x m part of array A must contain the symmetric matrix,
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the
     *        upper triangular part of the symmetric matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
     *        the leading m x m part stores the lower triangular part of the
     *        symmetric matrix and the strictly upper triangular part is not
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A
     *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the
     *        symmetric matrix and the strictly lower triangular part of A is not
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part
     *        stores the lower triangular part of the symmetric matrix and the
     *        strictly upper triangular part is not referenced.
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * B      double precision array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      double precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     * 
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
     *        B * A + beta * C
     * 
     * Reference: http://www.netlib.org/blas/dsymm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are &lt; 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDsymm(char side, char uplo, int m, int n, double alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, double beta, String C, int offsetC, int ldc);

    public static void cublasDsymm(char side, char uplo, int m, int n, double alpha, String A, int lda, String B, int ldb, double beta, String C, int ldc) {
        cublasDsymm(side, uplo, m, n, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void 
     * cublasDsyrk (char uplo, char trans, int n, int k, double alpha, 
     *              const double *A, int lda, double beta, double *C, int ldc)
     * 
     * performs one of the symmetric rank k operations
     * 
     *   C = alpha * A * transpose(A) + beta * C, or 
     *   C = alpha * transpose(A) * A + beta * C.
     * 
     * Alpha and beta are double precision scalars. C is an n x n symmetric matrix 
     * consisting of double precision elements and stored in either lower or 
     * upper storage mode. A is a matrix consisting of double precision elements
     * with dimension of n x k in the first case, and k x n in the second case.
     * 
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower 
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper 
     *        triangular part of the symmetric matrix is to be referenced, and the 
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the 
     *        lower triangular part of the symmetric matrix is to be referenced, 
     *        and the elements of the strictly upper triangular part are to be 
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C = 
     *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c', 
     *        C = transpose(A) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If 
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If 
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A. 
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A. 
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of 
     *        matrix A. k must be at least zero.
     * alpha  double precision scalar multiplier applied to A * transpose(A) or 
     *        transpose(A) * A.
     * A      double precision array of dimensions (lda, ka), where ka is k when 
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n', 
     *        the leading n x k part of array A must contain the matrix A, 
     *        otherwise the leading k x n part of the array must contains the 
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   double precision scalar multiplier applied to C. If beta izs zero, C
     *        does not have to be a valid input
     * C      double precision array of dimensions (ldc, n). If uplo = 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the 
     *        upper triangular part of the symmetric matrix C and the strictly 
     *        lower triangular part of C is not referenced. On exit, the upper 
     *        triangular part of C is overwritten by the upper trinagular part of 
     *        the updated matrix. If uplo = 'L' or 'l', the leading n x n 
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is 
     *        overwritten by the lower trinagular part of the updated matrix.
     * ldc    leading dimension of C. It must be at least max(1, n).
     * 
     * Output
     * ------
     * C      updated according to C = alpha * A * transpose(A) + beta * C, or C = 
     *        alpha * transpose(A) * A + beta * C
     * 
     * Reference: http://www.netlib.org/blas/dsyrk.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0 or k &lt; 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDsyrk(char uplo, char trans, int n, int k, double alpha, String A, int offsetA, int lda, double beta, String C, int offsetC, int ldc);

    public static void cublasDsyrk(char uplo, char trans, int n, int k, double alpha, String A, int lda, double beta, String C, int ldc) {
        cublasDsyrk(uplo, trans, n, k, alpha, A, 0, lda, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void
     * cublasDsyr2k (char uplo, char trans, int n, int k, double alpha,
     *               const double *A, int lda, const double *B, int ldb,
     *               double beta, double *C, int ldc)
     * 
     * performs one of the symmetric rank 2k operations
     * 
     *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
     *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
     * 
     * Alpha and beta are double precision scalars. C is an n x n symmetric matrix
     * consisting of double precision elements and stored in either lower or upper
     * storage mode. A and B are matrices consisting of double precision elements
     * with dimension of n x k in the first case, and k x n in the second case.
     * 
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be references,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n',
     *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
     *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
     *        alpha * transpose(B) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  double precision scalar multiplier.
     * A      double precision array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      double precision array of dimensions (lda, kb), where kb is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array B must contain the matrix B,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      double precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper trinagular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower trinagular part of the updated matrix.
     * ldc    leading dimension of C. Must be at least max(1, n).
     * 
     * Output
     * ------
     * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
     *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
     * 
     * Reference:   http://www.netlib.org/blas/dsyr2k.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n &lt; 0 or k &lt; 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, double beta, String C, int offsetC, int ldc);

    public static void cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, String A, int lda, String B, int ldb, double beta, String C, int ldc) {
        cublasDsyr2k(uplo, trans, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * Wrapper for CUBLAS function.
     * 
     * <pre>
     * void cublasZgemm (char transa, char transb, int m, int n, int k,
     *                   cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
     *                   const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
     *                   cuDoubleComplex *C, int ldc)
     * 
     * zgemm performs one of the matrix-matrix operations
     * 
     *    C = alpha * op(A) * op(B) + beta*C,
     * 
     * where op(X) is one of
     * 
     *    op(X) = X   or   op(X) = transpose  or  op(X) = conjg(transpose(X))
     * 
     * alpha and beta are double-complex scalars, and A, B and C are matrices
     * consisting of double-complex elements, with op(A) an m x k matrix, op(B)
     * a k x n matrix and C an m x n matrix.
     * 
     * Input
     * -----
     * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. If transa ==
     *        'T' or 't', op(A) = transpose(A). If transa == 'C' or 'c', op(A) =
     *        conjg(transpose(A)).
     * transb specifies op(B). If transa == 'N' or 'n', op(B) = B. If transb ==
     *        'T' or 't', op(B) = transpose(B). If transb == 'C' or 'c', op(B) =
     *        conjg(transpose(B)).
     * m      number of rows of matrix op(A) and rows of matrix C. It must be at
     *        least zero.
     * n      number of columns of matrix op(B) and number of columns of C. It
     *        must be at least zero.
     * k      number of columns of matrix op(A) and number of rows of op(B). It
     *        must be at least zero.
     * alpha  double-complex scalar multiplier applied to op(A)op(B)
     * A      double-complex array of dimensions (lda, k) if transa ==  'N' or
     *        'n'), and of dimensions (lda, m) otherwise.
     * lda    leading dimension of A. When transa == 'N' or 'n', it must be at
     *        least max(1, m) and at least max(1, k) otherwise.
     * B      double-complex array of dimensions (ldb, n) if transb == 'N' or 'n',
     *        and of dimensions (ldb, k) otherwise
     * ldb    leading dimension of B. When transb == 'N' or 'n', it must be at
     *        least max(1, k) and at least max(1, n) otherwise.
     * beta   double-complex scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      double precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m).
     * 
     * Output
     * ------
     * C      updated according to C = alpha*op(A)*op(B) + beta*C
     * 
     * Reference: http://www.netlib.org/blas/zgemm.f
     * 
     * Error status for this function can be retrieved via cublasGetError().
     * 
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are &lt; 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static native void cublasZgemm(char transa, char transb, int m, int n, int k, JCuDoubleComplex alpha, String A, int offsetA, int lda, String B, int offsetB, int ldb, JCuDoubleComplex beta, String C, int offsetC, int ldc);

    public static void cublasZgemm(char transa, char transb, int m, int n, int k, JCuDoubleComplex alpha, String A, int lda, String B, int ldb, JCuDoubleComplex beta, String C, int ldc) {
        cublasZgemm(transa, transb, m, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

}