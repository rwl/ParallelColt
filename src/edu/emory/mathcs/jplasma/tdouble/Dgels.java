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

import java.util.concurrent.locks.Lock;

import edu.emory.mathcs.jplasma.Barrier;

class Dgels {

    private Dgels() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Find the least squares solution of an overdetermined problem using QR factorization
     *  Find minimum norm solution of an underdetermined problem using LQ factorization
     *
     *  Differences with LAPACK:
     *  - if one of the dimensions is zero, B is not set to zero
     *  - A and B are not scaled
     *  - if M < N, B(M+1:N,1:NRHS) is not set to zero
     */
    protected static int plasma_DGELS(int trans, int M, int N, int NRHS, double[] A, int A_offset, int LDA, double[] T,
            int T_offset, double[] B, int B_offset, int LDB) {
        int NB, MT, NT, NTRHS;
        int status;
        double[] Abdl;
        double[] Bbdl;
        double[] Tbdl;
        double[] bdl_mem;
        int size_elems;

        /* Check if initialized */
        if (!Dcommon.plasma_cntrl.initialized) {
            Dauxiliary.plasma_warning("plasma_DGELS", "PLASMA not initialized");
            return Dplasma.PLASMA_ERR_NOT_INITIALIZED;
        }

        /* Check input arguments */
        if (trans != Dplasma.PlasmaNoTrans) {
            Dauxiliary.plasma_error("plasma_DGELS", "only PlasmaNoTrans supported");
            return Dplasma.PLASMA_ERR_NOT_SUPPORTED;
        }
        if (M < 0) {
            Dauxiliary.plasma_error("plasma_DGELS", "illegal value of M");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        if (N < 0) {
            Dauxiliary.plasma_error("plasma_DGELS", "illegal value of N");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        if (NRHS < 0) {
            Dauxiliary.plasma_error("plasma_DGELS", "illegal value of NRHS");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        if (LDA < Math.max(1, M)) {
            Dauxiliary.plasma_error("plasma_DGELS", "illegal value of LDA");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        if (LDB < Math.max(1, Math.max(M, N))) {
            Dauxiliary.plasma_error("plasma_DGELS", "illegal value of LDB");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        /* Quick return - currently NOT equivalent to LAPACK's:
         * CALL DLASET( 'Full', MAX( M, N ), NRHS, ZERO, ZERO, B, LDB ) */
        if (Math.min(M, Math.min(N, NRHS)) == 0)
            return Dplasma.PLASMA_SUCCESS;

        /* Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE */
        status = Dauxiliary.plasma_tune(Dglobal.PLASMA_TUNE_DGELS, M, N, NRHS);
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_DGELS", "plasma_tune() failed");
            return status;
        }

        /* Set MT, NT & NTRHS */
        NB = Dcommon.plasma_cntrl.NB;
        NT = (N % NB == 0) ? (N / NB) : (N / NB + 1);
        MT = (M % NB == 0) ? (M / NB) : (M / NB + 1);
        NTRHS = (NRHS % NB == 0) ? (NRHS / NB) : (NRHS / NB + 1);

        /* If NB larger than NB_max, set NB_max to NB, reallocate WORK & TAU */
        if (Dcommon.plasma_cntrl.NB > Dcommon.plasma_cntrl.NB_max) {
            status = Dallocate.plasma_free_aux_work_tau();
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGELS", "plasma_free_aux_work_tau() failed");
                return status;
            }
            Dcommon.plasma_cntrl.NB_max = Dcommon.plasma_cntrl.NB;
            status = Dallocate.plasma_alloc_aux_work_tau();
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGELS", "plasma_alloc_aux_work_tau() failed");
                return status;
            }
        }

        /* If progress table too small, reallocate */
        size_elems = Math.max(MT, NT) * Math.max(NT, NTRHS);
        if (Dcommon.plasma_cntrl.progress_size_elems < size_elems) {
            status = Dallocate.plasma_free_aux_progress();
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGELS", "plasma_free_aux_progress() failed");
            }
            status = Dallocate.plasma_alloc_aux_progress(size_elems);
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGELS", "plasma_alloc_aux_progress() failed");
                return status;
            }
        }

        /* Assign arrays to BDL storage */
        bdl_mem = Dcommon.plasma_aux.bdl_mem;
        Abdl = bdl_mem;
        int Abdl_offset = 0;
        Tbdl = bdl_mem;
        int Tbdl_offset = MT * NT * Dcommon.plasma_cntrl.NBNBSIZE;
        Bbdl = bdl_mem;
        int Bbdl_offset = Tbdl_offset + MT * NT * Dcommon.plasma_cntrl.IBNBSIZE;
        size_elems = Bbdl_offset + Math.max(MT, NT) * NTRHS * Dcommon.plasma_cntrl.NBNBSIZE;
        /* If BDL storage too small, reallocate & reassign */
        if (Dcommon.plasma_cntrl.bdl_size_elems < size_elems) {
            status = Dallocate.plasma_free_aux_bdl();
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGELS", "plasma_free_aux_bdl() failed");
                return status;
            }
            status = Dallocate.plasma_alloc_aux_bdl(size_elems);
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGELS", "plasma_alloc_aux_bdl() failed");
                return status;
            }
            bdl_mem = Dcommon.plasma_aux.bdl_mem;
            Abdl = bdl_mem;
            Abdl_offset = 0;
            Tbdl = bdl_mem;
            Tbdl_offset = MT * NT * Dcommon.plasma_cntrl.NBNBSIZE;
            Bbdl = bdl_mem;
            Bbdl_offset = Tbdl_offset + MT * NT * Dcommon.plasma_cntrl.IBNBSIZE;
        }

        if (M >= N) {
            /* Convert A from LAPACK to BDL */
            /* Set arguments */
            Dcommon.plasma_args.F77 = A;
            Dcommon.plasma_args.F77_offset = A_offset;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.M = M;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.LDA = LDA;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            /* Signal workers */
            Lock lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_F77_TO_BDL;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            DbdlConvert.plasma_lapack_to_bdl(Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset,
                    Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.M, Dcommon.plasma_args.N,
                    Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT, Dcommon.plasma_args.NT,
                    Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Convert B from LAPACK to BDL */
            /* Set arguments */
            Dcommon.plasma_args.F77 = B;
            Dcommon.plasma_args.F77_offset = B_offset;
            Dcommon.plasma_args.A = Bbdl;
            Dcommon.plasma_args.A_offset = Bbdl_offset;
            Dcommon.plasma_args.M = M;
            Dcommon.plasma_args.N = NRHS;
            Dcommon.plasma_args.LDA = LDB;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.NT = NTRHS;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_F77_TO_BDL;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            DbdlConvert.plasma_lapack_to_bdl(Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset,
                    Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.M, Dcommon.plasma_args.N,
                    Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT, Dcommon.plasma_args.NT,
                    Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Use QR factorization */
            /* Call parallel DGEQRF */
            /* Set arguments */
            Dcommon.plasma_args.M = M;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            Dcommon.plasma_args.IBNBSIZE = Dcommon.plasma_cntrl.IBNBSIZE;
            Dcommon.plasma_args.IB = Dcommon.plasma_cntrl.IB;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.T = Tbdl;
            Dcommon.plasma_args.T_offset = Tbdl_offset;
            /* Clear progress table */
            Dauxiliary.plasma_clear_aux_progress(MT * NT, -1);
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DGEQRF;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            Pdgeqrf.plasma_pDGEQRF(Dcommon.plasma_args.M, Dcommon.plasma_args.N, Dcommon.plasma_args.A,
                    Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB, Dcommon.plasma_args.NBNBSIZE,
                    Dcommon.plasma_args.IBNBSIZE, Dcommon.plasma_args.IB, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.NT, Dcommon.plasma_args.T, Dcommon.plasma_args.T_offset,
                    Dcommon.plasma_args.INFO, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Return T to the user */
            System.arraycopy(Tbdl, Tbdl_offset, T, T_offset, MT * NT * Dcommon.plasma_cntrl.IBNBSIZE);

            /* Call parallel DORMQR */
            /* Set arguments */
            Dcommon.plasma_args.M = M;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.NRHS = NRHS;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            Dcommon.plasma_args.IBNBSIZE = Dcommon.plasma_cntrl.IBNBSIZE;
            Dcommon.plasma_args.IB = Dcommon.plasma_cntrl.IB;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.NTRHS = NTRHS;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.T = Tbdl;
            Dcommon.plasma_args.T_offset = Tbdl_offset;
            Dcommon.plasma_args.B = Bbdl;
            Dcommon.plasma_args.B_offset = Bbdl_offset;
            /* Clear progress table */
            Dauxiliary.plasma_clear_aux_progress(MT * NTRHS, -1);
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DORMQR;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            Pdormqr.plasma_pDORMQR(Dcommon.plasma_args.M, Dcommon.plasma_args.NRHS, Dcommon.plasma_args.N,
                    Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB,
                    Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_args.IBNBSIZE, Dcommon.plasma_args.IB,
                    Dcommon.plasma_args.MT, Dcommon.plasma_args.NTRHS, Dcommon.plasma_args.NT, Dcommon.plasma_args.T,
                    Dcommon.plasma_args.T_offset, Dcommon.plasma_args.B, Dcommon.plasma_args.B_offset,
                    Dcommon.plasma_args.INFO, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Call parallel DTRSM */
            /* Set arguments */
            Dcommon.plasma_args.uplo = Dplasma.PlasmaUpper;
            Dcommon.plasma_args.trans = Dplasma.PlasmaNoTrans;
            Dcommon.plasma_args.diag = Dplasma.PlasmaNonUnit;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.NRHS = NRHS;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.B = Bbdl;
            Dcommon.plasma_args.B_offset = Bbdl_offset;
            Dcommon.plasma_args.MTB = MT;
            Dcommon.plasma_args.NTRHS = NTRHS;
            /* Clear progress table */
            Dauxiliary.plasma_clear_aux_progress(NT * NTRHS, -1);
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DTRSM;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            Pdtrsm.plasma_pDTRSM(Dplasma.PlasmaLeft, Dcommon.plasma_args.uplo, Dcommon.plasma_args.trans,
                    Dcommon.plasma_args.diag, Dcommon.plasma_args.N, Dcommon.plasma_args.NRHS, 1.0,
                    Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB,
                    Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_args.NT, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.B, Dcommon.plasma_args.B_offset, Dcommon.plasma_args.MTB,
                    Dcommon.plasma_args.NTRHS, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Convert A from BDL to LAPACK */
            /* Set arguments */
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.F77 = A;
            Dcommon.plasma_args.F77_offset = A_offset;
            Dcommon.plasma_args.M = M;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.LDA = LDA;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_BDL_TO_F77;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            DbdlConvert.plasma_bdl_to_lapack(Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset,
                    Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset, Dcommon.plasma_args.M,
                    Dcommon.plasma_args.N, Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.NT, Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Convert B from BDL to LAPACK */
            /* Set arguments */
            Dcommon.plasma_args.A = Bbdl;
            Dcommon.plasma_args.A_offset = Bbdl_offset;
            Dcommon.plasma_args.F77 = B;
            Dcommon.plasma_args.F77_offset = B_offset;
            Dcommon.plasma_args.M = M;
            Dcommon.plasma_args.N = NRHS;
            Dcommon.plasma_args.LDA = LDB;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.NT = NTRHS;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_BDL_TO_F77;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            DbdlConvert.plasma_bdl_to_lapack(Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset,
                    Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset, Dcommon.plasma_args.M,
                    Dcommon.plasma_args.N, Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.NT, Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
        } else {
            /* Convert arrays from LAPACK to BDL */
            DbdlConvert.plasma_lapack_to_bdl(A, A_offset, Abdl, Abdl_offset, M, N, LDA, Dcommon.plasma_cntrl.NB, MT,
                    NT, Dcommon.plasma_cntrl.NBNBSIZE, 1, 0);
            DbdlConvert.plasma_lapack_to_bdl(B, B_offset, Bbdl, Bbdl_offset, N, NRHS, LDB, Dcommon.plasma_cntrl.NB, NT,
                    NTRHS, Dcommon.plasma_cntrl.NBNBSIZE, 1, 0);

            /* Use LQ factorization */
            /* Call parallel DGEQRF */
            /* Set arguments */
            Dcommon.plasma_args.M = M;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            Dcommon.plasma_args.IBNBSIZE = Dcommon.plasma_cntrl.IBNBSIZE;
            Dcommon.plasma_args.IB = Dcommon.plasma_cntrl.IB;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.T = Tbdl;
            Dcommon.plasma_args.T_offset = Tbdl_offset;
            /* Clear progress table */
            Dauxiliary.plasma_clear_aux_progress(MT * NT, -1);
            /* Signal workers */
            Lock lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DGELQF;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            Pdgelqf.plasma_pDGELQF(Dcommon.plasma_args.M, Dcommon.plasma_args.N, Dcommon.plasma_args.A,
                    Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB, Dcommon.plasma_args.NBNBSIZE,
                    Dcommon.plasma_args.IBNBSIZE, Dcommon.plasma_args.IB, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.NT, Dcommon.plasma_args.T, Dcommon.plasma_args.T_offset,
                    Dcommon.plasma_args.INFO, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Call parallel DTRSM */
            /* Set arguments */
            Dcommon.plasma_args.uplo = Dplasma.PlasmaLower;
            Dcommon.plasma_args.trans = Dplasma.PlasmaNoTrans;
            Dcommon.plasma_args.diag = Dplasma.PlasmaNonUnit;
            Dcommon.plasma_args.N = M;
            Dcommon.plasma_args.NRHS = NRHS;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            Dcommon.plasma_args.NT = MT;
            Dcommon.plasma_args.MT = MT;
            Dcommon.plasma_args.B = Bbdl;
            Dcommon.plasma_args.B_offset = Bbdl_offset;
            Dcommon.plasma_args.MTB = NT;
            Dcommon.plasma_args.NTRHS = NTRHS;
            /* Clear progress table */
            Dauxiliary.plasma_clear_aux_progress(MT * NTRHS, -1);
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DTRSM;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            Pdtrsm.plasma_pDTRSM(Dplasma.PlasmaLeft, Dcommon.plasma_args.uplo, Dcommon.plasma_args.trans,
                    Dcommon.plasma_args.diag, Dcommon.plasma_args.N, Dcommon.plasma_args.NRHS, 1.0,
                    Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB,
                    Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_args.NT, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.B, Dcommon.plasma_args.B_offset, Dcommon.plasma_args.MTB,
                    Dcommon.plasma_args.NTRHS, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Here LAPACK sets B(M+1:N,1:NRHS) to zero
             */

            /* Call parallel DORMQR */

            /* Convert arrays from BDL to LAPACK */
            DbdlConvert.plasma_bdl_to_lapack(Abdl, Abdl_offset, A, A_offset, M, N, LDA, Dcommon.plasma_cntrl.NB, MT,
                    NT, Dcommon.plasma_cntrl.NBNBSIZE, 1, 0);
            DbdlConvert.plasma_bdl_to_lapack(Bbdl, Bbdl_offset, B, B_offset, N, NRHS, LDB, Dcommon.plasma_cntrl.NB, NT,
                    NTRHS, Dcommon.plasma_cntrl.NBNBSIZE, 1, 0);
        }

        return Dplasma.PLASMA_SUCCESS;
    }
}
