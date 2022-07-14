
from mathutils import Matrix, Vector, Quaternion
from math import radians
import math
import numpy as np
import bpy
import cv2
import mediapipe as mp

bl_info = {
    "name": "Character Driven: Live or Offline",
    "author": "yanch2116",
    "blender": (2, 80, 0),
    "version": (1, 0, 0),
}


class CharacterDriven(bpy.types.Operator):
    bl_idname = 'yanch.characterdriven'
    bl_label = 'characterdriven'

    def execute(self, ctx):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.factor = 10
        self.fram=0
        self.preresult=None
        fram = 0
        self.cap = cv2.VideoCapture(0)
        self.thumbl_Importer_ = SMPL_Importer(ctx)

        ctx.window_manager.modal_handler_add(self)
        mocap_timer = ctx.window_manager.event_timer_add(
            1 / 120, window=ctx.window)

        return {'RUNNING_MODAL'}

    # @time_fun
    def modal(self, ctx, evt):

        if evt.type == 'TIMER':
            with self.mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.6) as hands:
                flag, image = self.cap.read()
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.multi_handedness==None:
                    return {'RUNNING_MODAL'}
                for classfid in results.multi_handedness:
                    for j,name in enumerate(classfid.classification):
                        if name.label!="Right":
                            return {'RUNNING_MODAL'}
                if self.fram==0:
                    self.preresult=results.multi_hand_world_landmarks
                    self.fram+=1
                    return {'RUNNING_MODAL'}
                if results.multi_hand_world_landmarks:
                    for hand_landmarks,pre_handmaks in zip(results.multi_hand_world_landmarks,self.preresult):
                        for i, (lm0,lm1) in enumerate(zip(pre_handmaks.landmark,hand_landmarks.landmark)):
                            matrix0 = Vector((lm0.x*10, lm0.y*10, lm0.z*10))
                            #matrix0 = Vector((0.000000001, 0.000000001, 0.000000001))
                            matrix1=[-lm1.x*10,-lm1.y*10,-lm1.z*10]
                            if i == 1:
                                continue
                            self.thumbl_Importer_.process_poses(self.fram, i, matrix0,matrix1)
                self.fram=self.fram+1
                self.preresult=results.multi_hand_landmarks
            if cv2.waitKey(5):
                cv2.imshow("nihao", image)
            #print("begain")

        if evt.type == 'R':
            self.cap.release()
            return {'FINISHED'}

        return {'RUNNING_MODAL'}


class SMPL_Importer:

    def __init__(self, context):
        self.bone_name_from_index = {
            0: 'hand.L',
            1: 'L_Hip',
            2: 'thumb.01.L',
            3: 'thumb.02.L',
            4: 'thumb.03.L',
            5: 'palm.01.L',
            6: 'f_index.01.L',
            7: 'f_index.02.L',
            8: 'f_index.03.L',
            9: 'palm.02.L',
            10: 'f_middle.01.L',
            11: 'f_middle.02.L',
            12: 'f_middle.03.L',
            13: 'palm.03.L',
            14: 'f_ring.01.L',
            15: 'f_ring.02.L',
            16: 'f_ring.03.L',
            17: 'palm.04.L',
            18: 'f_pinky.01.L',
            19: 'f_pinky.02.L',
            20: 'f_pinky.03.L',
        }

    #convert it to a quaternion
    def Rodmatrix(self,srcmatrix,dstmatrix):
        # 代码是在blender环境中跑的，只需将Vector和Matrix转换到numpy版本即可在普通python环境中跑

        #T_location是目标向量
        T_location = dstmatrix

        T_location_norm = T_location.copy()
        T_location_norm.normalize()
        originVector = srcmatrix

        print(T_location_norm)

        sita = math.acos(T_location_norm @ originVector)
        n_vector = T_location_norm.cross(originVector)

        n_vector.normalize()

        n_vector_invert = Matrix((
            [0, -n_vector[2], n_vector[1]],
            [n_vector[2], 0, -n_vector[0]],
            [-n_vector[1], n_vector[0], 0]
        ))

        print(sita)
        print(n_vector_invert)

        I = Matrix((
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ))
        #核心公式：见上图
        R_w2c = I + math.sin(sita) * n_vector_invert + n_vector_invert @ (n_vector_invert) * (1 - math.cos(sita))
        return R_w2c
    #convert it to a quaternion
    def rotation_matrix_from_vectors(self,vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    #convert it to a quaternion
    def Rodrigues(self, rotvec):
            theta = np.linalg.norm(rotvec)
            r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
            cost = np.cos(theta)
            mat = np.asarray([[0, -r[2], r[1]],
                              [r[2], 0, -r[0]],
                              [-r[1], r[0], 0]])
            return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)

    def process_poses(self, framid, indx, matr0,matr1):
        matrix = self.Rodrigues(matr1)
        #print("matrix",matrix)
        armature = bpy.data.objects['metarig']
        bones = armature.pose.bones
        bone = bones[self.bone_name_from_index[indx]]
        bone_rotation = Matrix(matrix).to_quaternion()
        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))
        quat_x_n135_cw = Quaternion((1.0, 0.0, 0.0), radians(-135))
        quat_x_p45_cw = Quaternion((1.0, 0.0, 0.0), radians(45))
        quat_y_90_cw = Quaternion((0.0, 1.0, 0.0), radians(-90))
        quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))

        if indx == 0:
            # Rotate pelvis so that avatar stands upright and looks along negative Y avis
            bone.rotation_quaternion = (
                                               quat_x_90_cw @ quat_z_90_cw) @ bone_rotation

        #print("bone_rotation", bone_rotation)
        else:
            bone.rotation_quaternion = bone_rotation
        print("bone.rotation_quaternion+"+self.bone_name_from_index[indx],bone.rotation_quaternion)
        bpy.context.scene.frame_end = framid
        return


addon_keymaps = []


def register():
    bpy.utils.register_class(CharacterDriven)
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = wm.keyconfigs.addon.keymaps.new(
            name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new(
            CharacterDriven.bl_idname, type='E', value='PRESS', ctrl=True)
        addon_keymaps.append((km, kmi))


def unregister():
    bpy.utils.unregister_class(CharacterDriven)
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()


if __name__ == "__main__":
    register()