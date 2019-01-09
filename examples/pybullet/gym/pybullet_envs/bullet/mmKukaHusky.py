import os
import sys
sys.path.append('/home/zheng/ws_ros/src/bullet3/examples/pybullet/gym/pybullet_envs/')
import pybullet as p
import pybullet_data
import math


class MMKukaHusky:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 200.
        self.useSimulation = 1
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.reset()
        self.useNullSpace = 0
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def reset(self):
        #p.setGravity(0, 0, -9.8)
        self.huskyUid = p.loadURDF(os.path.join(self.urdfRootPath, "husky/husky.urdf"), [0.290388,0.329902,-0.010270],
                                   [0.002328,-0.000984,0.996491,0.083659])
        self.kukaUid = p.loadURDF(os.path.join(self.urdfRootPath, "kuka_iiwa/model_free_base.urdf"),
                                  [0.193749,0.345564,0.420208], [0.002327,-0.000988,0.996491,0.083659])
        '''
        for i in range (p.getNumJoints(self.huskyUid)):
            print(p.getJointInfo(self.huskyUid,i))
        for i in range (p.getNumJoints(self.kukaUid)):
            print(p.getJointInfo(self.kukaUid,i))
        '''
        p.createConstraint(self.huskyUid, -1, self.kukaUid, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0., 0., -.5], [0, 0, 0, 1])


        #self.wheelVelR = 0.0
        #self.wheelVelL = 0.0
        #self.wheelVel = [self.wheelVelR, self.wheelVelL, self.wheelVelR, self.wheelVelL]
        initial_wheelVel = [0, 0, 0, 0]
        self.wheels = [2, 3, 4, 5]
        p.resetBaseVelocity(self.huskyUid, initial_wheelVel, initial_wheelVel)
        for wheelIndex in range(len(self.wheels)):
            # reset no-zero base velocities
            # not necessary
            # p.resetJointState(self.huskyUid, wheelIndex, self.wheelVel[wheelIndex], self.wheelVel[wheelIndex])
            p.setJointMotorControl2(self.huskyUid, wheelIndex, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=initial_wheelVel[wheelIndex], force=self.maxForce)

        # reset arm joint positions and controllers
        initial_jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.jointstates = initial_jointPositions
        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, initial_jointPositions[jointIndex])
            p.setJointMotorControl2(self.kukaUid, jointIndex, controlMode=p.POSITION_CONTROL,
                                    targetPosition=initial_jointPositions[jointIndex], force=self.maxForce)

        self.wheelDeltasTurn = [1, -1, 1, -1]
        self.wheelDeltasFwd = [1, 1, 1, 1]

        self.motorNames = []
        self.motorIndices = []
        self.wheelNames = []
        self.wheelIndices = []

        initial_kukastate = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
        self.kukastate = [initial_kukastate[0][0],initial_kukastate[0][1], initial_kukastate[0][2]]

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.kukaUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

        for i in range (p.getNumJoints(self.huskyUid)):
            wheelInfo = p.getJointInfo(self.huskyUid, i)
            qIndex = wheelInfo[3]
            if qIndex > -1:
                self.wheelNames.append(str(wheelInfo[1]))
                self.wheelIndices.append(i)


    def getActionDimension(self):
        return len(self.motorIndices)+len(self.wheelIndices)
        # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        #huskystate = p.getLinkState(self.huskyUid, 0)
        kukastate = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
        state = kukastate
        #print('state: ', state)
        pos = state[0]
        orn = state[1]
        if (len(state)>6):
            vel = state[6]
        else:
            vel = [0,0,0]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(vel))
        # observation.extend(list(euler))
        #print('o', observation, list(pos))

        return observation

    def accurateCalculateInverseKinematics(self, kukaId, endEffectorId, targetPos, threshold, maxIter):
        closeEnough = False
        iter = 0
        while (not closeEnough and iter < maxIter):
            jointPoses = p.calculateInverseKinematics(kukaId, endEffectorId, targetPos)
            ls = p.getLinkState(kukaId, endEffectorId)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1
        return jointPoses

    def check_jointstates(self, joint_state, delta_j):
        if (abs(joint_state[0]) > 2.967):
            joint_state[0] = joint_state[0] - delta_j[0]

        if abs(joint_state[1]) > 2.094:
            joint_state[1] = joint_state[1] - delta_j[1]

        if abs(joint_state[2]) > 2.967:
            joint_state[2] = joint_state[2] - delta_j[2]

        if abs(joint_state[3]) > 2.094:
            joint_state[3] = joint_state[3] - delta_j[3]

        if abs(joint_state[4]) > 2.967:
            joint_state[4] = joint_state[4] - delta_j[4]

        if abs(joint_state[5]) > 2.094:
            joint_state[5] = joint_state[5] - delta_j[5]

        if abs(joint_state[6]) > 3.054:
            joint_state[6] = joint_state[6] - delta_j[6]

        return joint_state

    def applyAction(self, motorCommands):
        # action of arm joint states changes
        if (len(motorCommands)==5):
            dp = motorCommands[0:3]
            eeposx = self.kukastate[0] + dp[0]
            eeposy = self.kukastate[1] + dp[1]
            eeposz = self.kukastate[2] + dp[2]

            eepos = [eeposx, eeposy, eeposz]
            self.kukastate = eepos
            # baseVel is the translational speed of husky
            self.baseVel = motorCommands[3]
            # baseAng is the rotational speed of husky
            self.baseAng = motorCommands[4]

            if (self.useNullSpace == 1):
                jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, eepos, lowerLimits=self.ll,
                                                              upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
            else:
                threshold = 0.001
                maxIter = 100
                jointPoses = self.accurateCalculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, eepos, threshold, maxIter)

        # action of arm ee position changes
        elif (len(motorCommands) == 9):
            dp = motorCommands[0:7]
            # baseVel is the translational speed of husky
            self.baseVel = motorCommands[7]
            # baseAng is the rotational speed of husky
            self.baseAng = motorCommands[8]

            self.jointstates = [x+y for x, y in zip(self.jointstates, dp)]
            self.jointstates = self.check_jointstates(self.jointstates, dp)
            # print('jointstate : ', self.jointstates, abs(-9))
            jointPoses = self.jointstates

        self.wheelVelR = (2 * self.baseVel + 0.555 * self.baseAng) / 2
        self.wheelVelL = (2 * self.baseVel - 0.555 * self.baseAng) / 2
        wheelVel = [self.wheelVelL, self.wheelVelR, self.wheelVelL, self.wheelVelR]

        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyUniqueId = self.kukaUid, jointIndex = i, controlMode = p.POSITION_CONTROL,
                                            targetPosition = jointPoses[i], force = 1000, positionGain = 1, velocityGain = 0.1)
        for i in range(len(self.wheels)):
            p.setJointMotorControl2(self.huskyUid, self.wheels[i], p.VELOCITY_CONTROL, targetVelocity=wheelVel[i], force=1000)