#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import torch
import numpy as np
from threading import Lock

class ROSPolicyBridge:
    def __init__(self, policy_checkpoint):
        # Load trained policy
        try:
            self.policy = torch.jit.load(policy_checkpoint)
            self.policy.eval()
        except Exception as e:
            rospy.logerr(f"Failed to load policy: {e}")
            raise
        
        # Safety parameters
        self.max_joint_velocity = 1.0  # rad/s
        self.joint_limits = {
            'upper': np.array([2.5] * 7),  # rad
            'lower': np.array([-2.5] * 7)
        }
        
        # Thread safety
        self.lock = Lock()
        
        # ROS Setup
        rospy.init_node('rl_policy')
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        self.cmd_pub = rospy.Publisher('/joint_commands', JointState, queue_size=1)
        
        # State monitoring
        self.last_cmd = None
        self.last_cmd_time = None
        self.obs_buffer = None
        
        rospy.loginfo("ROS Policy Bridge initialized successfully")

    def _check_safety(self, action):
        if self.last_cmd is not None:
            dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
            velocity = (action - self.last_cmd) / dt
            if np.any(np.abs(velocity) > self.max_joint_velocity):
                rospy.logwarn("Command exceeds velocity limits!")
                return False
        
        if np.any(action > self.joint_limits['upper']) or np.any(action < self.joint_limits['lower']):
            rospy.logwarn("Command exceeds position limits!")
            return False
            
        return True

    def joint_callback(self, msg):
        with self.lock:
            try:
                # Convert ROS message to observation tensor
                obs = {
                    'qpos': np.array(msg.position[:7]),
                    'qvel': np.array(msg.velocity[:7]),
                    'eef_pos': self._get_eef_pose()
                }
                
                # Run policy
                with torch.no_grad():
                    obs_tensor = {k: torch.FloatTensor(v) for k, v in obs.items()}
                    action = self.policy(obs_tensor).cpu().numpy()
                
                # Safety check
                if not self._check_safety(action):
                    return
                
                # Publish commands
                cmd_msg = JointState()
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.position = action.tolist()
                self.cmd_pub.publish(cmd_msg)
                
                # Update state
                self.last_cmd = action
                self.last_cmd_time = rospy.Time.now()
                
            except Exception as e:
                rospy.logerr(f"Error in joint callback: {e}")

    def _get_eef_pose(self):
        try:
            # Wait for transform
            listener = tf.TransformListener()
            listener.waitForTransform('/base_link', '/ee_link', rospy.Time(), rospy.Duration(1.0))
            
            # Get transform
            (trans, rot) = listener.lookupTransform('/base_link', '/ee_link', rospy.Time(0))
            return np.array(trans)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return np.zeros(3)

if __name__ == '__main__':
    try:
        policy_path = rospy.get_param('~policy_path', 'path/to/policy.pt')
        bridge = ROSPolicyBridge(policy_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Bridge failed: {e}")