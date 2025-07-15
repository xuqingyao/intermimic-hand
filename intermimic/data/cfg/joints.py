
datajoints = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_hand_thumb_0_link', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link', 'left_hand_middle_0_link', 'left_hand_middle_1_link', 'left_hand_index_0_link', 'left_hand_index_1_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link', 'right_hand_index_0_link', 'right_hand_index_1_link']
g1joints = ['pelvis', 'imu_in_pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'd435_link', 'head_link', 'imu_in_torso', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_hand_palm_link', 'left_hand_index_0_link', 'left_hand_index_1_link', 'left_hand_middle_0_link', 'left_hand_middle_1_link', 'left_hand_thumb_0_link', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link', 'logo_link', 'mid360_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_hand_palm_link', 'right_hand_index_0_link', 'right_hand_index_1_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link']
contactjoints = ['left_hip_yaw_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link', 'left_hand_index_1_link', 'left_hand_middle_1_link', 'left_hand_thumb_2_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link', 'right_hand_index_1_link', 'right_hand_middle_1_link', 'right_hand_thumb_2_link']


mapindex = []
for joint in datajoints:
    mapindex.append(g1joints.index(joint))
print("MAP", mapindex)

contactindex = []
for joint in contactjoints:
    contactindex.append(datajoints.index(joint))
print("Contact", contactindex)

