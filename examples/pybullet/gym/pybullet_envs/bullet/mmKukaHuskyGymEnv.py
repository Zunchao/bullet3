import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import mmKukaHusky
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960



class MMKukaHuskyGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=1000,
                 action_dim = 5):
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._action_dim = action_dim

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        self._seed()
        self.reset()
        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(9)
        else:
            self._action_bound = math.pi/2
            # action_high = np.array([self._action_bound] * action_dim)
            if (self._action_dim == 5):
                # husky twist limits is from https://github.com/husky/husky/blob/kinetic-devel/husky_control/config/control.yaml
                action_high = np.array([0.01, 0.01, 0.01, 1, 2])
            elif (self._action_dim == 9):
                action_high = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 2])
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def _reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)

        p.setGravity(0, 0, -9.8)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, 0])

        xpos = random.uniform(-1, 1)
        ypos = random.uniform(-1, 1)
        zpos = random.uniform(0.5, 1.4)
        self.goal = [xpos, ypos, zpos]
        self.goalUid = p.loadURDF(os.path.join(self._urdfRoot, "spheregoal.urdf"), xpos, ypos, zpos)

        self._mmkukahusky = mmKukaHusky.MMKukaHusky(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = self._mmkukahusky.getObservation()
        return self._observation

    def _step(self, action):
        for i in range(self._actionRepeat):
            self._mmkukahusky.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        done = self._termination()
        reward = self._reward()

        return np.array(self._observation), reward, done, {}

    def _render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._mmkukahusky.huskyUid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        state = p.getLinkState(self._mmkukahusky.kukaUid, self._mmkukahusky.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]

        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True

        disvec = [x-y for x, y in zip(actualEndEffectorPos, self.goal)]
        dis = np.linalg.norm(disvec)

        if (dis<0.1):  # (actualEndEffectorPos[2] <= -0.43):
            self.terminated = 1
            self._observation = self.getExtendedObservation()
            print('terminate:', self._observation, dis,self.goal)
            return True
        return False

    def _reward(self):
        state = p.getLinkState(self._mmkukahusky.kukaUid, self._mmkukahusky.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        disvec = [x-y for x, y in zip(actualEndEffectorPos, self.goal)]
        dis = np.linalg.norm(disvec)
        reward = -dis
        return reward

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        render = _render
        reset = _reset
        seed = _seed
        step = _step
