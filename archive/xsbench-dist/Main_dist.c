/*
    This file is subject to the license agreement located in the file ../../../../LICENSE (apps/LICENSE)
    and cannot be distributed without it. This notice cannot be removed or modified.

    This file is also subject to the license agreement located in the file LICENSE in the current directory.
*/

/* Distributed OCR XSBench driver (mimicking the MPI+OpenMP implementation's
 * distribution model): true SPMD over the policy domains, one deterministic
 * table replica per SPMD rank.  Every rank builds its own full copy of the
 * nuclide/energy/material tables from the fixed data-generation RNG, so the
 * replicas are bit-identical with zero data movement and all table reads are
 * local to the rank that built them for the whole run.  The global lookup
 * range is tiled once across ranks and then across each rank's worker
 * threads; per-thread digests return through a two-level gather (thread
 * partials -> rank partial -> global checksum), so the only cross-rank
 * payloads are one u64 per rank. */

#include "XSbench_header.h"

// Lookups per generation-chain link: bounds every task's execution time
// without any communication cost (all table reads are rank-local).
#ifndef XSD_CHUNK
#define XSD_CHUNK 25000
#endif

// Gather fan-in bound of the global finisher (maximum SPMD width).
#define XSD_MAX_RANKS 256

// Task-local replica of the C library's additive-feedback rand() stream
// (TYPE_3: degree-31 table, separation 3, 310 warm-up discards).  The
// data-generation RNG must be private to the building task: rand()'s
// process-global state interleaves nondeterministically when several ranks'
// builders are co-scheduled in one process, and the per-rank replicas must
// be bit-identical for the summed lookup digest to be partition-invariant.
typedef struct { u32 r[31]; int f, rp; } xsdRand_t;

static u32 xsdRandNext( xsdRand_t* s )
{
    s->r[s->f] += s->r[s->rp];
    u32 out = s->r[s->f] >> 1;
    if( ++s->f  >= 31 ) s->f  = 0;
    if( ++s->rp >= 31 ) s->rp = 0;
    return out;
}

static void xsdRandInit( xsdRand_t* s, u32 seed )
{
    if( seed == 0 ) seed = 1;
    s->r[0] = seed;
    for( int i = 1; i < 31; i++ )
        s->r[i] = (u32)((16807ULL * s->r[i-1]) % 2147483647ULL);
    s->f = 3; s->rp = 0;
    for( int i = 0; i < 310; i++ ) (void) xsdRandNext(s);
}

#define XSD_RAND_MAX 2147483647.0

typedef struct
{
    Inputs in;
    s64 nRanks;
    double startTime;
    ocrEVT_t finishEVT[XSD_MAX_RANKS];
} xsdGlobalH_t;

typedef struct
{
    u64 iCur, iEndExcl;       // remaining global lookup range [iCur, iEndExcl)
    u64 vhash;                // running digest carried across the chain
    u64 n_isotopes, n_gridpoints;
    ocrEVT_t doneEVT;         // fires with this thread's partial when done
    ocrDBK_t dbk[6];          // nuclide_grids, uEnergy_grid, xs_grid,
                              // num_nucs, mats_all, concs_all
} PRM_gen_t;

typedef struct
{
    ocrEVT_t finishEVT;       // the rank's slot on the global finisher
    ocrDBK_t tableDbk[9];     // every replica DB, destroyed after the gather
} PRM_rankSum_t;

ocrGuid_t genEdt( EDT_ARGS );

static double xsdNow( void )
{
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_sec + 1.0e-6 * (double) tv.tv_usec;
}

// EDT-affinity-only placement: children are pinned to the policy domain of
// their creator, queried fresh at each create site.
static ocrHint_t xsdEdtHintCurrent( void )
{
    ocrHint_t h;
    ocrHintInit( &h, OCR_HINT_EDT_T );
#ifdef ENABLE_EXTENSION_AFFINITY
    ocrGuid_t aff = NULL_GUID;
    ocrAffinityGetCurrent( &aff );
    ocrSetHintValue( &h, OCR_HINT_EDT_AFFINITY, ocrAffinityToHintValue(aff) );
#endif
    return h;
}

// Architecture-stable digest of one lookup's inputs and result.  Each
// cross-section is rounded to a fixed decimal precision before mixing so
// bit-level floating-point differences across compilers/optimizers cannot
// perturb the checksum; the per-contribution range is bounded so the summed
// total stays well within u64.
static u64 xs_lookup_hash( double p_energy, int mat, const double* v )
{
    u64 h = 1469598103934665603ULL;
    h = (h ^ (u64)(s64)llround(p_energy * 100000.0)) * 1099511628211ULL;
    h = (h ^ (u64)(u32)mat) * 1099511628211ULL;
    for( int k = 0; k < 5; k++ )
        h = (h ^ (u64)(s64)llround(v[k] * 100000.0)) * 1099511628211ULL;
    return h % 10000ULL;
}

// Macroscopic lookup over the linear table DBs: identical arithmetic to
// calculate_macro_xs, but the unionized-grid cross-links are read as plain
// indices out of the linear xs-index table instead of through embedded
// pointers, so read-only consumers never need per-address pointer arrays
// (nor writes into a read-only mapping) to walk the grids.
static void calculate_macro_xs_dist( double p_energy, int mat, long n_isotopes,
                                     long n_gridpoints,
                                     const int* num_nucs, const int* mat_off,
                                     const double* concs_all,
                                     GridPoint* uEnergy_grid,
                                     const int* xs_grid,
                                     NuclideGridPoint* nuclide_grids_linear,
                                     const int* mats_all,
                                     double* macro_xs_vector )
{
    double xs_vector[5];

    for( int k = 0; k < 5; k++ )
        macro_xs_vector[k] = 0;

    long idx = grid_search( n_isotopes * n_gridpoints, p_energy, uEnergy_grid );

    for( int j = 0; j < num_nucs[mat]; j++ )
    {
        int p_nuc = mats_all[mat_off[mat] + j];
        double conc = concs_all[mat_off[mat] + j];
        calculate_micro_xs_new( p_energy, n_gridpoints,
                                xs_grid[idx * n_isotopes + p_nuc],
                                nuclide_grids_linear + (long)p_nuc * n_gridpoints,
                                xs_vector );
        for( int k = 0; k < 5; k++ )
            macro_xs_vector[k] += xs_vector[k] * conc;
    }
}

// Builds this rank's table replica.  The fixed data-generation seed makes
// every rank's replica bit-identical, which is what makes the summed lookup
// digest invariant to how the lookups are partitioned; the build's writes
// also leave the payloads resident on the building rank.
// generate_grids with the draw source swapped from the process-global rand()
// to the task-local stream; the fill order and arithmetic are unchanged, so
// for the same seed the tables are bit-identical to the rand()-built ones.
static void generate_grids_dist( NuclideGridPoint ** nuclide_grids,
                                 long n_isotopes, long n_gridpoints, xsdRand_t* s )
{
    for( long i = 0; i < n_isotopes; i++ )
        for( long j = 0; j < n_gridpoints; j++ )
        {
            nuclide_grids[i][j].energy        = ((double)xsdRandNext(s)/XSD_RAND_MAX);
            nuclide_grids[i][j].total_xs      = ((double)xsdRandNext(s)/XSD_RAND_MAX);
            nuclide_grids[i][j].elastic_xs    = ((double)xsdRandNext(s)/XSD_RAND_MAX);
            nuclide_grids[i][j].absorbtion_xs = ((double)xsdRandNext(s)/XSD_RAND_MAX);
            nuclide_grids[i][j].fission_xs    = ((double)xsdRandNext(s)/XSD_RAND_MAX);
            nuclide_grids[i][j].nu_fission_xs = ((double)xsdRandNext(s)/XSD_RAND_MAX);
        }
}

// load_concs with the same rand() -> task-local stream swap (DB layout and
// fill order unchanged).
static void load_concs_dist( rankDataH_t* PTR_rankDataH, int * num_nucs, xsdRand_t* s )
{
    double ** concs;
    ocrDbCreate( &PTR_rankDataH->DBK_conc_ptrs, (void **) &concs, 12*sizeof(double *),
                 0, NULL_HINT, NO_ALLOC );

    int num_nucs_all = 0;
    for( int i = 0; i < 12; i++ )
        num_nucs_all += num_nucs[i];

    double* concs_all;
    ocrDbCreate( &PTR_rankDataH->DBK_concs_all, (void **) &concs_all, num_nucs_all*sizeof(double),
                 0, NULL_HINT, NO_ALLOC );

    assign_concs_ptrs( num_nucs, concs_all, concs );

    for( int i = 0; i < 12; i++ )
        for( int j = 0; j < num_nucs[i]; j++ )
            concs[i][j] = (double) xsdRandNext(s) / XSD_RAND_MAX;
}

static void buildReplica( rankDataH_t* PTR_rankDataH, const Inputs* PTR_in, int mype )
{
    xsdRand_t dataRNG;
    xsdRandInit( &dataRNG, 42 );

    if( mype == 0 ) ocrPrintf("Generating Nuclide Energy Grids...\n");
    NuclideGridPoint ** nuclide_grids = gpmatrix( &PTR_rankDataH->DBK_nuclide_grids,
                                                  &PTR_rankDataH->DBK_nuclide_grid_ptrs,
                                                  PTR_in->n_isotopes, PTR_in->n_gridpoints );
    generate_grids_dist( nuclide_grids, PTR_in->n_isotopes, PTR_in->n_gridpoints, &dataRNG );

    if( mype == 0 ) ocrPrintf("Sorting Nuclide Energy Grids...\n");
    sort_nuclide_grids( nuclide_grids, PTR_in->n_isotopes, PTR_in->n_gridpoints );

    GridPoint * energy_grid = generate_energy_grid( &PTR_rankDataH->DBK_uEnergy_grid,
                                                    &PTR_rankDataH->DBK_xs_grid,
                                                    PTR_in->n_isotopes, PTR_in->n_gridpoints,
                                                    nuclide_grids, mype );
    set_grid_ptrs( energy_grid, nuclide_grids, PTR_in->n_isotopes, PTR_in->n_gridpoints, mype );

    if( mype == 0 ) ocrPrintf("Loading Mats...\n");
    int *num_nucs = load_num_nucs( &PTR_rankDataH->DBK_num_nucs, PTR_in->n_isotopes );
    load_mats( PTR_rankDataH, num_nucs, PTR_in->n_isotopes );
    load_concs_dist( PTR_rankDataH, num_nucs, &dataRNG );

    ocrDbRelease( PTR_rankDataH->DBK_nuclide_grids );
    ocrDbRelease( PTR_rankDataH->DBK_nuclide_grid_ptrs );
    ocrDbRelease( PTR_rankDataH->DBK_uEnergy_grid );
    ocrDbRelease( PTR_rankDataH->DBK_xs_grid );
    ocrDbRelease( PTR_rankDataH->DBK_num_nucs );
    ocrDbRelease( PTR_rankDataH->DBK_mat_ptrs );
    ocrDbRelease( PTR_rankDataH->DBK_conc_ptrs );
    ocrDbRelease( PTR_rankDataH->DBK_mats_all );
    ocrDbRelease( PTR_rankDataH->DBK_concs_all );
}

// One generation-chain link: runs up to XSD_CHUNK lookups of its thread's
// range, then either chains the next link (digest carried in paramv) or
// publishes the thread's partial (release-before-satisfy handoff).
ocrGuid_t genEdt( EDT_ARGS )
{
    s32 _idep;

    PRM_gen_t* p = (PRM_gen_t*) paramv;

    _idep = 0;
    NuclideGridPoint * nuclide_grids_linear = depv[_idep++].ptr;
    GridPoint * uEnergy_grid = depv[_idep++].ptr;
    int * xs_grid = depv[_idep++].ptr;
    int * num_nucs = depv[_idep++].ptr;
    int * mats_all = depv[_idep++].ptr;
    double * concs_all = depv[_idep++].ptr;

    int mat_off[12];
    int off = 0;
    for( int m = 0; m < 12; m++ )
    {
        mat_off[m] = off;
        off += num_nucs[m];
    }

    u64 span = p->iEndExcl - p->iCur;
    if( span > XSD_CHUNK ) span = XSD_CHUNK;

    long n_isotopes = (long) p->n_isotopes;
    long n_gridpoints = (long) p->n_gridpoints;

    double macro_xs_vector[5];
    u64 vhash = p->vhash;

    // Global-index-pure seeding: the draw for lookup i depends only on i (not
    // on the rank/thread that runs it), so any exact tiling of [0,lookups)
    // yields the same per-lookup contribution and the summed checksum is
    // invariant to rank count, thread count, and placement.
    for( u64 i = p->iCur; i < p->iCur + span; i++ )
    {
        u64 seed = ((i + 1) * 19 + 17) % 2147483646ULL + 1;
        double p_energy = rn(&seed);
        int mat         = pick_mat(&seed);

        calculate_macro_xs_dist( p_energy, mat, n_isotopes, n_gridpoints,
                                 num_nucs, mat_off, concs_all, uEnergy_grid,
                                 xs_grid, nuclide_grids_linear, mats_all,
                                 macro_xs_vector );

        vhash += xs_lookup_hash( p_energy, mat, macro_xs_vector );
    }

    if( p->iCur + span < p->iEndExcl )
    {
        PRM_gen_t next = *p;
        next.iCur = p->iCur + span;
        next.vhash = vhash;

        ocrHint_t myEdtAffinityHNT = xsdEdtHintCurrent();

        ocrGuid_t genTML, genEDT;
        ocrEdtTemplateCreate( &genTML, genEdt, PARAMC_U64(PRM_gen_t), 6 );
        ocrEdtCreate( &genEDT, genTML, EDT_PARAM_DEF, (u64*)&next, EDT_PARAM_DEF, NULL,
                      EDT_PROP_NONE, &myEdtAffinityHNT, NULL );
        ocrEdtTemplateDestroy( genTML );

        for( u32 d = 0; d < 6; d++ )
            ocrAddDependence( p->dbk[d], genEDT, d, DB_MODE_RO );
    }
    else
    {
        for( u32 d = 0; d < 6; d++ )
            ocrDbRelease( depv[d].guid );

        ocrDBK_t DBK_partial;
        u64* partial;
        ocrDbCreate( &DBK_partial, (void**) &partial, sizeof(u64), 0, NULL_HINT, NO_ALLOC );
        partial[0] = vhash;
        ocrDbRelease( DBK_partial );
        ocrEventSatisfy( p->doneEVT, DBK_partial );
    }

    return NULL_GUID;
}

// Gathers the rank's per-thread partials, retires the replica, and forwards
// the rank partial to the global finisher.
ocrGuid_t rankSummaryEdt( EDT_ARGS )
{
    PRM_rankSum_t* p = (PRM_rankSum_t*) paramv;

    u64 sum = 0;
    for( u32 t = 0; t < depc; t++ )
    {
        sum += ((u64*) depv[t].ptr)[0];
        ocrDbDestroy( depv[t].guid );
    }

    for( int d = 0; d < 9; d++ )
        ocrDbDestroy( p->tableDbk[d] );

    ocrDBK_t DBK_partial;
    u64* partial;
    ocrDbCreate( &DBK_partial, (void**) &partial, sizeof(u64), 0, NULL_HINT, NO_ALLOC );
    partial[0] = sum;
    ocrDbRelease( DBK_partial );
    ocrEventSatisfy( p->finishEVT, DBK_partial );

    return NULL_GUID;
}

// paramv: {nRanks}; depv 0..nRanks-1: rank partials (RO), nRanks: global
// params (RO).  Sums the per-rank digests, prints results, and shuts down.
ocrGuid_t globalFinishEdt( EDT_ARGS )
{
    u64 nRanks = paramv[0];

    u64 vhash = 0;
    for( u64 g = 0; g < nRanks; g++ )
    {
        vhash += ((u64*) depv[g].ptr)[0];
        ocrDbDestroy( depv[g].guid );
    }

    xsdGlobalH_t* PTR_globalH = (xsdGlobalH_t*) depv[nRanks].ptr;
    double runtime = xsdNow() - PTR_globalH->startTime;

    ocrPrintf("\n");
    ocrPrintf("Simulation complete.\n");
    print_results( PTR_globalH->in, 0, runtime, (int) nRanks, vhash );

    // Deterministic correctness checksum: a pure function of the fixed
    // data-generation and index-derived lookup seeds, so it must be
    // identical across runs, thread counts, node counts, and runtimes.
    ocrPrintf("XS_CHECKSUM = %llu\n", (unsigned long long) vhash);

    ocrDbDestroy( depv[nRanks].guid );
    ocrShutdown();

    return NULL_GUID;
}

// One SPMD rank: builds the rank's replica, then launches this rank's
// worker-thread generation chains over its share of the lookup range and the
// rank gather that joins them.
ocrGuid_t initEdt( EDT_ARGS )
{
    PRM_init1dEdt_t* PTR_PRM_initEdt = (PRM_init1dEdt_t*) paramv;

    u64 myRank = PTR_PRM_initEdt->id;
    u64 nRanks = PTR_PRM_initEdt->edtGridDims[0];

    const xsdGlobalH_t* PTR_globalH = (const xsdGlobalH_t*) depv[1].ptr;
    Inputs in = PTR_globalH->in;

    u64 K = (in.nthreads < 1) ? 1 : (u64) in.nthreads;
    u64 L = (u64) in.lookups;

    rankDataH_t rankDataH;
    buildReplica( &rankDataH, &in, (int) myRank );

    ocrDBK_t tableDbk6[6] = {
        rankDataH.DBK_nuclide_grids, rankDataH.DBK_uEnergy_grid,
        rankDataH.DBK_xs_grid, rankDataH.DBK_num_nucs,
        rankDataH.DBK_mats_all, rankDataH.DBK_concs_all
    };

    ocrHint_t myEdtAffinityHNT = xsdEdtHintCurrent();

    // Thread-done events are fully wired to the rank gather BEFORE any chain
    // can fire them (register-then-satisfy ordering).
    ocrEVT_t* doneEVT = (ocrEVT_t*) malloc( K * sizeof(ocrEVT_t) );
    for( u64 t = 0; t < K; t++ )
        ocrEventCreate( &doneEVT[t], OCR_EVENT_ONCE_T, EVT_PROP_TAKES_ARG );

    PRM_rankSum_t PRM_rankSum;
    PRM_rankSum.finishEVT = PTR_globalH->finishEVT[myRank];
    PRM_rankSum.tableDbk[0] = rankDataH.DBK_nuclide_grids;
    PRM_rankSum.tableDbk[1] = rankDataH.DBK_nuclide_grid_ptrs;
    PRM_rankSum.tableDbk[2] = rankDataH.DBK_uEnergy_grid;
    PRM_rankSum.tableDbk[3] = rankDataH.DBK_xs_grid;
    PRM_rankSum.tableDbk[4] = rankDataH.DBK_num_nucs;
    PRM_rankSum.tableDbk[5] = rankDataH.DBK_mat_ptrs;
    PRM_rankSum.tableDbk[6] = rankDataH.DBK_conc_ptrs;
    PRM_rankSum.tableDbk[7] = rankDataH.DBK_mats_all;
    PRM_rankSum.tableDbk[8] = rankDataH.DBK_concs_all;

    ocrGuid_t rankSumTML, rankSumEDT;
    ocrEdtTemplateCreate( &rankSumTML, rankSummaryEdt, PARAMC_U64(PRM_rankSum_t), K );
    ocrEdtCreate( &rankSumEDT, rankSumTML, EDT_PARAM_DEF, (u64*)&PRM_rankSum,
                  EDT_PARAM_DEF, NULL, EDT_PROP_NONE, &myEdtAffinityHNT, NULL );
    ocrEdtTemplateDestroy( rankSumTML );

    for( u64 t = 0; t < K; t++ )
        ocrAddDependence( doneEVT[t], rankSumEDT, (u32) t, DB_MODE_RO );

    // Rank g's share of [0,L) is [g*L/P, (g+1)*L/P); each thread tiles the
    // rank share the same way.  Exclusive bounds keep empty shares
    // well-defined; every index in [0,L) is covered exactly once.
    u64 rankBeg = myRank * L / nRanks;
    u64 rankEnd = (myRank + 1) * L / nRanks;
    u64 N = rankEnd - rankBeg;

    ocrGuid_t genTML;
    ocrEdtTemplateCreate( &genTML, genEdt, PARAMC_U64(PRM_gen_t), 6 );

    for( u64 t = 0; t < K; t++ )
    {
        PRM_gen_t PRM_gen;
        PRM_gen.iCur     = rankBeg + t * N / K;
        PRM_gen.iEndExcl = rankBeg + (t + 1) * N / K;
        PRM_gen.vhash = 0;
        PRM_gen.n_isotopes = (u64) in.n_isotopes;
        PRM_gen.n_gridpoints = (u64) in.n_gridpoints;
        PRM_gen.doneEVT = doneEVT[t];
        for( int d = 0; d < 6; d++ )
            PRM_gen.dbk[d] = tableDbk6[d];

        ocrGuid_t genEDT;
        ocrEdtCreate( &genEDT, genTML, EDT_PARAM_DEF, (u64*)&PRM_gen,
                      EDT_PARAM_DEF, NULL, EDT_PROP_NONE, &myEdtAffinityHNT, NULL );

        for( u32 d = 0; d < 6; d++ )
            ocrAddDependence( tableDbk6[d], genEDT, d, DB_MODE_RO );
    }

    ocrEdtTemplateDestroy( genTML );
    free( doneEVT );

    return NULL_GUID;
}

ocrGuid_t mainEdt( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[] )
{
    ocrGuid_t DBK_cmdLineArgs = depv[0].guid;
    void * PTR_cmdLineArgs = depv[0].ptr;
    u32 argc = ocrGetArgc( PTR_cmdLineArgs );

    ocrGuid_t argv_g;
    char** argv;
    ocrDbCreate( &argv_g, (void**)&argv, sizeof(char*) * (argc ? argc : 1),
                 DB_PROP_NONE, NULL_HINT, NO_ALLOC );
    for( u32 a = 0; a < argc; ++a )
        argv[a] = ocrGetArgv( PTR_cmdLineArgs, a );

    int version = 13;
    Inputs in = read_CLI( argc, argv );
    ocrDbDestroy( argv_g );

    // SPMD width defaults to the policy-domain count (one rank per node);
    // an explicit -p overrides it, e.g. to probe partition invariance.
    u64 nRanks = 1;
#ifdef ENABLE_EXTENSION_AFFINITY
    ocrAffinityCount( AFFINITY_PD, &nRanks );
#endif
    if( in.nprocs > 1 ) nRanks = (u64) in.nprocs;
    if( nRanks > XSD_MAX_RANKS ) nRanks = XSD_MAX_RANKS;
    in.nprocs = (int) nRanks;

    print_inputs( in, (int) nRanks, version );

    ocrGuid_t DBK_globalH;
    xsdGlobalH_t* PTR_globalH;
    ocrDbCreate( &DBK_globalH, (void**) &PTR_globalH, sizeof(xsdGlobalH_t),
                 DB_PROP_NONE, NULL_HINT, NO_ALLOC );
    PTR_globalH->in = in;
    PTR_globalH->nRanks = (s64) nRanks;
    PTR_globalH->startTime = xsdNow();

    ocrHint_t myEdtAffinityHNT = xsdEdtHintCurrent();

    ocrGuid_t finishTML, finishEDT;
    u64 finishPRM[1] = { nRanks };
    ocrEdtTemplateCreate( &finishTML, globalFinishEdt, 1, (u32)(nRanks + 1) );
    ocrEdtCreate( &finishEDT, finishTML, EDT_PARAM_DEF, finishPRM, EDT_PARAM_DEF, NULL,
                  EDT_PROP_NONE, &myEdtAffinityHNT, NULL );
    ocrEdtTemplateDestroy( finishTML );

    for( u64 g = 0; g < nRanks; g++ )
    {
        ocrEventCreate( &PTR_globalH->finishEVT[g], OCR_EVENT_ONCE_T, EVT_PROP_TAKES_ARG );
        ocrAddDependence( PTR_globalH->finishEVT[g], finishEDT, (u32) g, DB_MODE_RO );
    }

    ocrDbRelease( DBK_globalH );
    ocrAddDependence( DBK_globalH, finishEDT, (u32) nRanks, DB_MODE_RO );

    u64 edtGridDims[1] = { nRanks };
    ocrGuid_t spmdDepv[2] = { DBK_cmdLineArgs, DBK_globalH };
    forkSpmdEdts_Cart1D( initEdt, edtGridDims, spmdDepv );

    return NULL_GUID;
}
