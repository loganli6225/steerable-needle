"""What does the planner actually return at the moments it replans?

The idealised test (aligned pose, heading straight at the goal) showed no
looping at either curvature -- every path came back 1.0x direct. But it also
showed a 500x iteration cost at kappa=1/25, d=20mm (13,002 vs 23 iterations),
which says the tighter-curvature planner is struggling in that regime even
when it eventually succeeds.

Real replans do not happen from aligned poses. They happen from wherever the
needle has drifted to, pointing wherever it happens to point, under a finite
budget. This script pulls the ACTUAL replan poses out of a failing run and
asks what the planner returned from each -- rather than guessing with a proxy.

Produced the terminal-approach diagnosis: a replan 15.6mm from the goal at
0.89 rad of heading offset returned a 399mm plan (25x direct), whose loop drove
the needle off the workspace. That motivated the terminal-approach guard now in
closed_loop.py. NOTE: the guard is unconditionally active in run_closed_loop, so
rerunning this script no longer reproduces those pre-guard numbers -- it now
shows the guard WORKING: on `open`, seed 1, model 1/25 the terminal replan is
suppressed and only a single benign replan (~80mm out, ~1.1x direct) fires, the
run ending near the goal via plan_exhausted rather than off_map. That is the
confirmation, not a regression.

Run:  python scripts/diagnose_replans.py
"""

import math

from needlesim.benchmark.scenarios import OPEN, build_env
from needlesim.control.closed_loop import ClosedLoopConfig, run_closed_loop
from needlesim.estimation.ekf import EKFConfig
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

SCENARIO = OPEN
TRUE_KAPPA = 1.0 / 50.0
MODEL_KAPPA = 1.0 / 25.0
SEED = 1  # this seed went off_map at 1/25 on `open` PRE-GUARD; with the
# guard active it now ends near the goal (see module docstring)


def make_planner(env, kappa, seed):
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, NeedleParams(kappa=kappa), cfg)


def path_length(path):
    return sum(
        math.hypot(path[i + 1].x - path[i].x, path[i + 1].y - path[i].y)
        for i in range(len(path) - 1)
    )


def main():
    env = build_env(SCENARIO)
    goal = SCENARIO.goal

    cl = run_closed_loop(
        planner=make_planner(env, MODEL_KAPPA, SEED),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=MODEL_KAPPA),
        start=SCENARIO.start,
        goal=goal,
        ekf_config=EKFConfig(seed=SEED),
        loop_config=ClosedLoopConfig(seed=SEED),
    )

    print(
        f"run: {cl.final_error_mm:.1f}mm  end={cl.termination_reason}  "
        f"replans at {cl.replan_steps}  fails={cl.n_replan_failures}"
    )
    print()

    # The pose the planner was given at each replan is the ESTIMATE at that
    # step -- that is what run_closed_loop passes to plan().
    print("Poses the planner was actually asked to replan from:")
    print(
        f"{'step':>6} {'est pose (x, y, theta)':>30} {'dist to goal':>13} "
        f"{'heading off':>12} {'plan len':>9} {'x direct':>9}"
    )
    print("-" * 90)

    for i, k in enumerate(cl.replan_steps):
        est = cl.estimated_states[k - 1]  # state at the step the replan fired
        d = math.hypot(est.x - goal.x, est.y - goal.y)

        # How far off is the heading from pointing straight at the goal? This
        # is the quantity the idealised test held at zero.
        bearing = math.atan2(goal.y - est.y, goal.x - est.x)
        off = abs((est.theta - bearing + math.pi) % (2 * math.pi) - math.pi)

        # plans[i+1] is the plan produced BY this replan (plans[0] is initial)
        plan = cl.plans[i + 1] if i + 1 < len(cl.plans) else None
        if plan is None:
            print(f"{k:6d} {'(no plan recorded)':>30}")
            continue

        L = path_length(plan)
        print(
            f"{k:6d} ({est.x:7.1f},{est.y:7.1f},{est.theta:6.2f}) "
            f"{d:13.1f} {off:12.2f} {L:9.1f} {L/max(d, 1e-9):9.1f}"
        )

    print()
    print("Then: re-plan from each of those poses at BOTH curvatures, to")
    print("separate 'this pose is hard' from 'this curvature makes it hard'.")
    print(
        f"{'step':>6} {'k=1/50 len':>11} {'x direct':>9} {'iters':>8}   "
        f"{'k=1/25 len':>11} {'x direct':>9} {'iters':>8}"
    )
    print("-" * 90)

    for k in cl.replan_steps:
        est = cl.estimated_states[k - 1]
        d = math.hypot(est.x - goal.x, est.y - goal.y)
        row = [f"{k:6d}"]
        for kappa in (TRUE_KAPPA, MODEL_KAPPA):
            env_i = build_env(SCENARIO)
            r = make_planner(env_i, kappa, SEED).plan(est, goal)
            if not r.success:
                row.append(f"{'FAILED':>11} {'-':>9} {r.n_iterations:8d}")
            else:
                L = path_length(r.path)
                row.append(f"{L:11.1f} {L/max(d,1e-9):9.1f} {r.n_iterations:8d}")
        print("   ".join(row))

    print()
    print("Reading it:")
    print("  'x direct' near 1.0 means the planner found a direct route from")
    print("  that pose. Much larger means it looped. If 1/25 loops where 1/50")
    print("  does not, curvature is the cause. If BOTH are near 1.0, the plans")
    print("  are fine and the off_map exits come from something else -- most")
    print("  likely the needle being unable to FOLLOW an adequate plan, rather")
    print("  than the plan being bad.")
    print("  'heading off' is the quantity the idealised test held at zero;")
    print("  if it is large at the replan moments, that test was not")
    print("  representative of the real problem.")


if __name__ == "__main__":
    main()
