/* ***** BEGIN LICENSE BLOCK *****
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright 2008
 *
 * Redistribution  and  use  in  source and binary forms, with or without
 * modification,  are  permitted  provided  that the following conditions
 * are met:
 *
 * * Redistributions  of  source  code  must  retain  the above copyright
 *   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * * Redistributions  in  binary  form must reproduce the above copyright
 *   notice,  this list of conditions and the following disclaimer in the
 *   documentation  and/or other materials provided with the distribution.
 * * Neither  the  name of the University of Tennessee, Knoxville nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jplasma.tdouble;

import java.util.concurrent.Future;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.netlib.util.intW;

class Dglobal {

    private Dglobal() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Configuration
     */
    protected static final int CORES_MAX = 1024;

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Action commands
     */
    protected static final int PLASMA_ACT_STAND_BY = 0;
    protected static final int PLASMA_ACT_FINALIZE = 1;

    protected static final int PLASMA_ACT_DGEQRF = 2;
    protected static final int PLASMA_ACT_DORMQR = 3;
    protected static final int PLASMA_ACT_DTRSM = 4;
    protected static final int PLASMA_ACT_DPOTRF = 5;
    protected static final int PLASMA_ACT_DGELQF = 6;
    protected static final int PLASMA_ACT_DGETRF = 7;
    protected static final int PLASMA_ACT_DTRSMPL = 8;

    protected static final int PLASMA_ACT_F77_TO_BDL = 9;
    protected static final int PLASMA_ACT_BDL_TO_F77 = 10;

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Tuned functions
     */
    protected static final int PLASMA_TUNE_DGELS = 1;
    protected static final int PLASMA_TUNE_DPOSV = 2;
    protected static final int PLASMA_TUNE_DGESV = 3;

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Main control data structure
     */
    protected static class Dplasma_cntrl {
        protected Lock action_mutex; // master->workers action communication
        protected Condition action_condt;
        protected int action;
        protected boolean initialized; // initialization flag
        protected int NB_max;
        protected int NB_min;
        protected int IB_max;
        protected int NB;
        protected int IB;
        protected int NBNBSIZE; // tile size padded to cache line size
        protected int IBNBSIZE; // T tile size padded to cache line size
        protected int cores_max;
        protected int cores_num;
        protected int bdl_size_elems;
        protected int progress_size_elems;
        protected int[] core_num;
        protected Future<?>[] workers;

        protected Dplasma_cntrl(int NTHREADS) {
            action_mutex = new ReentrantLock();
            action_condt = action_mutex.newCondition();
            action = PLASMA_ACT_STAND_BY;
            initialized = false;
            NB_max = 256;
            NB_min = 5;
            IB_max = 128;
            NB = 0;
            IB = 0;
            NBNBSIZE = 0;
            IBNBSIZE = 0;
            cores_max = 0;
            cores_num = 0;
            bdl_size_elems = 0;
            progress_size_elems = 0;
            core_num = new int[NTHREADS];
            workers = new Future[NTHREADS];
        }

    };

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Auxiliary storage
     */
    protected static class Dplasma_aux {
        protected double[] bdl_mem; // BDL storage
        protected volatile int[] progress; // progress table
        protected double[][] WORK; // kernel workspace
        protected double[][] TAU; // kernel workspace

        protected Dplasma_aux(int NTHREADS) {
            bdl_mem = null;
            progress = null;
            WORK = new double[NTHREADS][];
            TAU = new double[NTHREADS][];
        }
    };

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Master->workers communication of arguments
     */
    protected static class Plasma_args {
        protected int trans;
        protected int side;
        protected int uplo;
        protected int diag;
        protected int M;
        protected int N;
        protected int NRHS;
        protected int NB;
        protected int NBNBSIZE;
        protected int IBNBSIZE;
        protected int MT;
        protected int MTB;
        protected int NT;
        protected int NTRHS;
        protected int IB;
        protected int LDA;
        protected int LDB;
        protected double[] F77;
        protected int F77_offset;
        protected double[] A;
        protected int A_offset;
        protected double[] B;
        protected int B_offset;
        protected double[] T;
        protected int T_offset;
        protected double[] L;
        protected int L_offset;
        protected int[] IPIV;
        protected int IPIV_offset;
        protected intW INFO = new intW(0);

        protected Plasma_args() {

        }

    };

}
