# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_envs.bullet.mmKukaHuskyGymEnv import MMKukaHuskyGymEnv
import math


def main():
    env = MMKukaHuskyGymEnv(renders=True, isDiscrete=False, maxSteps=10000000, action_dim = 9)

    motorsIds = []
    dv = 0.01
    if(env._action_dim == 5):
        motorsIds.append(env._p.addUserDebugParameter("kuka_ee_dx", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_ee_dy", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_ee_dz", -dv, dv, 0))
    elif(env._action_dim == 9):
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_0", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_1", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_2", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_3", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_4", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_5", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_6", -dv, dv, 0))

    bs = 1
    motorsIds.append(env._p.addUserDebugParameter("base_linear_speed", -bs, bs, 0))
    motorsIds.append(env._p.addUserDebugParameter("base_angular_speed", -2*bs, 2*bs, 0))

    done = False
    while (not done):

        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))
        # print('action。 ', action)

        state, reward, done, info = env.step(action)
        obs = env.getExtendedObservation()

if __name__ == "__main__":
    main()
