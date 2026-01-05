def convert_franka_flat_to_ur5e_flat(franka_env, ur5e_env, franka_flat):
    # 1) Franka에 state 적용
    franka_env.sim.set_state_from_flattened(franka_flat)
    franka_env.sim.forward()

    # 2) 목표 EEF pose (world)
    tgt_pos, tgt_xmat = get_eef_world_pose(franka_env)

    # 3) UR5e reset
    ur5e_env.reset()
    ur5e_env.sim.forward()

    # 4) qpos/qvel split
    fr_t, fr_qpos, fr_qvel = split_state(franka_env.sim, franka_flat)
    ur_flat0 = ur5e_env.get_sim_state()
    ur_t, ur_qpos, ur_qvel = split_state(ur5e_env.sim, ur_flat0)

    # 5) 비로봇만 복사 (mask 기반, 빠르고 간단)
    _, _, fr_non_robot_qpos_mask, _ = robot_index_masks(franka_env)
    _, _, ur_non_robot_qpos_mask, _ = robot_index_masks(ur5e_env)

    # ⚠️ 주의: mask 방식은 “비로봇 qpos 배열의 순서/구성”이 동일하다는 가정이 들어감.
    # 지금 LIBERO task 구조가 거의 같아 보이지만, 더 안전하게 하려면 joint-name 복사를 쓰는 걸 추천.
    ur_qpos[ur_non_robot_qpos_mask] = fr_qpos[fr_non_robot_qpos_mask]

    # 6) UR5e에 비로봇 적용 + qvel 0
    ur5e_env.sim.data.qpos[:] = ur_qpos
    ur5e_env.sim.data.qvel[:] = 0.0
    ur5e_env.sim.forward()

    # 7) IK로 로봇 관절 맞추기 (처음엔 pos만)
    solve_ik_ur5e_inplace(ur5e_env, tgt_pos, tgt_xmat, match_ori=False)

    # 8) qvel 0 유지 + state rebuild (tail은 UR5e tail 유지)
    ur5e_env.sim.data.qvel[:] = 0.0
    ur5e_env.sim.forward()

    new_flat = merge_state(
        fr_t,
        ur5e_env.sim.data.qpos.copy(),
        ur5e_env.sim.data.qvel.copy(),
    )
    return new_flat

import mujoco


def mj_jacSite_compat(sim, site_id: int):
    # jacp/jacr는 float64 & C-contiguous 여야 안전
    jacp = np.zeros((3, sim.model.nv), dtype=np.float64)
    jacr = np.zeros((3, sim.model.nv), dtype=np.float64)

    m = sim.model
    d = sim.data

    # robosuite binding_utils 래퍼 언랩 (이름은 버전에 따라 다를 수 있어서 여러 후보를 체크)
    for attr in ["_model", "model", "_mjmodel", "_m"]:
        if hasattr(m, attr):
            m = getattr(m, attr)
            break
    for attr in ["_data", "data", "_mjdata", "_d"]:
        if hasattr(d, attr):
            d = getattr(d, attr)
            break

    mujoco.mj_jacSite(m, d, jacp, jacr, int(site_id))
    return jacp, jacr

def rot_to_omega(R_err):
    """R_err = R_target @ R_current^T. returns axis-angle vector (3,)"""
    tr = np.trace(R_err)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos)
    if ang < 1e-8:
        return np.zeros(3)
    w = np.array([
        R_err[2,1] - R_err[1,2],
        R_err[0,2] - R_err[2,0],
        R_err[1,0] - R_err[0,1],
    ]) / (2.0 * np.sin(ang))
    return w * ang

def solve_ik_ur5e_inplace(
    ur5e_env,
    target_pos,
    target_xmat,
    match_ori=False,
    n_iters=200,
    pos_tol=1e-4,
    ori_tol=1e-3,
    damping=1e-2,
    step=0.5,
):
    sim = ur5e_env.sim
    m, d = sim.model, sim.data
    robot = ur5e_env.robots[0]
    sid = robot.eef_site_id

    # 로봇 dof 인덱스 (qvel 기준) / qpos 기준
    jpos_idx = np.array(robot._ref_joint_pos_indexes, dtype=np.int64)  # len 6
    jvel_idx = np.array(robot._ref_joint_vel_indexes, dtype=np.int64)  # len 6

    for _ in range(n_iters):
        sim.forward()

        cur_pos = d.site_xpos[sid].copy()
        cur_xmat = d.site_xmat[sid].reshape(3,3).copy()

        pos_err = (target_pos - cur_pos)

        if match_ori:
            R_err = target_xmat @ cur_xmat.T
            ori_err = rot_to_omega(R_err)
        else:
            ori_err = np.zeros(3)

        if np.linalg.norm(pos_err) < pos_tol and (not match_ori or np.linalg.norm(ori_err) < ori_tol):
            break

        # Jacobian (3 x nv)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        jacp, jacr = mj_jacSite_compat(sim, sid)


        # 로봇 관절에 해당하는 column만 뽑기: (3 or 6) x 6
        Jp = jacp[:, jvel_idx]
        Jr = jacr[:, jvel_idx]
        if match_ori:
            Jr = jacr[:, jvel_idx]
            J = np.vstack([Jp, Jr])        # (6,6)
            err = np.concatenate([pos_err, ori_err])
        else:
            J = Jp                         # (3,6)
            err = pos_err                  # (3,)

        # DLS: dq = J^T (J J^T + λI)^-1 err
        A = J @ J.T + (damping**2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(A, err)

        # 업데이트
        q = d.qpos[jpos_idx].copy()
        q = q + step * dq
        d.qpos[jpos_idx] = q

    sim.forward()

def get_eef_world_pose(env):
    sim = env.sim
    r = env.robots[0]
    sid = r.eef_site_id
    pos = sim.data.site_xpos[sid].copy()                          # (3,)
    xmat = sim.data.site_xmat[sid].reshape(3, 3).copy()           # (3,3)
    return pos, xmat

def robot_index_masks(env):
    m = env.sim.model
    robot = env.robots[0]

    robot_qpos_idx = np.array(robot._ref_joint_pos_indexes, dtype=np.int64)
    robot_qvel_idx = np.array(robot._ref_joint_vel_indexes, dtype=np.int64)

    # 그리퍼까지 로봇으로 볼지 선택
    grip_qpos_idx = np.array(getattr(robot, "_ref_gripper_joint_pos_indexes", []), dtype=np.int64)
    grip_qvel_idx = np.array(getattr(robot, "_ref_gripper_joint_vel_indexes", []), dtype=np.int64)

    all_robot_qpos_idx = np.unique(np.concatenate([robot_qpos_idx, grip_qpos_idx]))
    all_robot_qvel_idx = np.unique(np.concatenate([robot_qvel_idx, grip_qvel_idx]))

    non_robot_qpos_mask = np.ones(m.nq, dtype=bool)
    non_robot_qpos_mask[all_robot_qpos_idx] = False

    non_robot_qvel_mask = np.ones(m.nv, dtype=bool)
    non_robot_qvel_mask[all_robot_qvel_idx] = False

    return robot_qpos_idx, robot_qvel_idx, non_robot_qpos_mask, non_robot_qvel_mask

import numpy as np

def split_state(sim, flat):
    m = sim.model
    nq, nv = m.nq, m.nv
    t = flat[0:1].copy()                    # (1,)
    qpos = flat[1:1+nq].copy()              # (nq,)
    qvel = flat[1+nq:1+nq+nq*0+nv].copy()   # (nv,)
    return t, qpos, qvel

def merge_state(t, qpos, qvel):
    return np.concatenate([t, qpos, qvel], axis=0)

import numpy as np

def get_eef_world_pos(env):
    sid = env.robots[0].eef_site_id
    return env.sim.data.site_xpos[sid].copy()  # (3,)

def get_base_world_pose(env, base_body_name="robot0_base"):
    bid = env.sim.model.body_name2id(base_body_name)
    p = env.sim.data.body_xpos[bid].copy()                # (3,)
    R = env.sim.data.body_xmat[bid].reshape(3, 3).copy()  # (3,3)
    return p, R

def world_to_base(p_world, base_p_world, base_R_world):
    # base_R_world: columns are base axes in world
    # transform: p_base = R^T (p_world - base_p)
    return base_R_world.T @ (p_world - base_p_world)

def compare_eef_pos(franka_env, ur5e_env, base_body_name="robot0_base"):
    # world pos
    pf_w = get_eef_world_pos(franka_env)
    pu_w = get_eef_world_pos(ur5e_env)

    err_w = pu_w - pf_w

    # base-frame pos
    bf_p, bf_R = get_base_world_pose(franka_env, base_body_name)
    bu_p, bu_R = get_base_world_pose(ur5e_env, base_body_name)

    pf_b = world_to_base(pf_w, bf_p, bf_R)
    pu_b = world_to_base(pu_w, bu_p, bu_R)

    err_b = pu_b - pf_b

    print("\n[EEF pos] World frame")
    print("  Franka:", pf_w)
    print("  UR5e  :", pu_w)
    print("  Err   :", err_w, " |norm| =", np.linalg.norm(err_w))

    print("\n[EEF pos] Robot base frame")
    print("  Franka:", pf_b)
    print("  UR5e  :", pu_b)
    print("  Err   :", err_b, " |norm| =", np.linalg.norm(err_b))

    return {
        "pf_w": pf_w, "pu_w": pu_w, "err_w": err_w,
        "pf_b": pf_b, "pu_b": pu_b, "err_b": err_b,
    }