#!/usr/bin/env bash
# CPU envelope for a reference-runtime rank.
#
#   bash envelope.sh <width> <ranks> <mode> -- <binary> <args...>
#
#   mode=fixed : the rank owns its host; block = 0..width-1
#   mode=rank  : colocated local ranks; block = r*width..(r+1)*width-1
#
# The wrapper verifies the block is made of per-core first SMT threads, sets
# it as its own affinity mask, reads the mask back and requires exact
# equality, then execs the command — children inherit the verified mask.
# The readback is a POSTCONDITION on purpose: sched_setaffinity intersects
# with the cpuset (not with the current mask), so a partial overlap narrows
# silently and only the readback can tell; and a launcher may have installed
# any mask beforehand, which a precondition check would wrongly reject.
#
# Every failure prints one ARTSRUN-ENVELOPE-FAIL line and exits 90; the
# result checker keys on the line, not the exit code, because a partial rank
# failure can surface to the launcher as a timeout instead.
#
# Test seams (default to the real paths): ENVELOPE_STATUS_FILE,
# ENVELOPE_SYSFS_ROOT, ENVELOPE_TASKSET.

set -u

fail() { printf 'ARTSRUN-ENVELOPE-FAIL: %s\n' "$*" >&2; exit 90; }

STATUS_FILE=${ENVELOPE_STATUS_FILE:-/proc/self/status}
SYSFS=${ENVELOPE_SYSFS_ROOT:-/sys/devices/system/cpu}
TASKSET=${ENVELOPE_TASKSET:-taskset}

[ "$#" -ge 5 ] || fail "usage: expected <width> <ranks> <mode> -- <cmd...>"
width=$1; ranks=$2; mode=$3; sep=$4; shift 4
[ "$sep" = "--" ] || fail "usage: fourth argument must be --, got '$sep'"
case $width in ''|*[!0-9]*) fail "usage: width '$width' is not a number";; esac
case $ranks in ''|*[!0-9]*) fail "usage: ranks '$ranks' is not a number";; esac
[ "$width" -ge 1 ] || fail "usage: width must be >= 1"

case $mode in
rank)
    # No fallback to 0: a silent r=0 stacks every rank onto block 0, and a
    # runtime that also binds per-core from its own configuration would then
    # mask the launcher bug while one that does not suffers it — the two
    # arms disagreeing for launcher reasons, not runtime reasons.
    r=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-}}
    [ -n "$r" ] || fail "rank: neither PMI_RANK nor OMPI_COMM_WORLD_RANK is set"
    case $r in ''|*[!0-9]*) fail "rank: '$r' is not a number";; esac
    [ "$r" -lt "$ranks" ] || fail "rank: rank $r is outside 0..$((ranks - 1))"
    start=$((r * width))
    ;;
fixed)
    # One rank per host is the remote contract; a second rank here would
    # claim the same block and the overcommit is invisible to either rank.
    for v in SLURM_LOCALID OMPI_COMM_WORLD_LOCAL_RANK MPI_LOCALRANKID \
             PMI_LOCAL_RANK; do
        val=${!v-}
        if [ -n "$val" ] && [ "$val" != 0 ]; then
            fail "colocation: $v=$val — a second rank landed on this host"
        fi
    done
    # A launch whose process manager never formed the world degrades MPI to
    # singleton init: n independent size-1 worlds that each "succeed".  The
    # launcher-side task count is cross-checked where one is visible.
    for v in SLURM_NTASKS PMI_SIZE; do
        val=${!v-}
        if [ -n "$val" ] && [ "$val" != "$ranks" ]; then
            fail "world: $v=$val but this cell runs $ranks rank(s)"
        fi
    done
    start=0
    ;;
*)
    fail "usage: mode '$mode' is not fixed|rank"
    ;;
esac
end=$((start + width - 1))

# Every block CPU must be the first thread of its own core: the whole point
# of the envelope is one thread per physical core, and on a sibling-adjacent
# enumeration a plain 0..width-1 block would be half cores, half SMT twins.
# Fail-closed on unreadable topology — a hidden sysfs must not silently
# disable the check.
for c in $(seq "$start" "$end"); do
    f=$SYSFS/cpu$c/topology/thread_siblings_list
    [ -r "$f" ] || f=$SYSFS/cpu$c/topology/core_cpus_list
    [ -r "$f" ] || fail "topology: no readable sibling list for cpu$c"
    first=$(head -n1 "$f" | sed 's/[,-].*//')
    [ "$first" = "$c" ] || fail \
        "topology: cpu$c is an SMT sibling of cpu$first, not a first thread"
done

if [ "$width" -eq 1 ]; then block=$start; else block=$start-$end; fi
"$TASKSET" -p -c "$block" $$ >/dev/null 2>&1 \
    || fail "bind: taskset -c $block failed (cpus offline or outside the cpuset)"
got=$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' "$STATUS_FILE")
[ "$got" = "$block" ] \
    || fail "bind: mask readback is '$got', wanted '$block' (cpuset narrowed the block)"

exec "$@"
