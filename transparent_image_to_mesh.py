import bpy
import numpy as np
from bpy.types import Panel, Operator
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from PIL import Image
import os

bl_info = {
    "name": "透明PNG转模型",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (4, 2, 3),
    "location": "View3D > Sidebar > 透明PNG",
    "description": "将透明PNG图片转换为裁剪后的平面模型",
    "category": "Import-Export",
}

class TransparentImagePanel(Panel):
    bl_label = "透明PNG"
    bl_idname = "VIEW3D_PT_transparent_png"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '透明PNG'

    def draw(self, context):
        layout = self.layout
        layout.operator("import.transparent_png", text="导入PNG")

class ImportTransparentPNG(Operator, ImportHelper):
    bl_idname = "import.transparent_png"
    bl_label = "导入透明PNG"
    
    filter_glob: StringProperty(
        default='*.png',
        options={'HIDDEN'}
    )

    def execute(self, context):
        return self.import_transparent_image(context, self.filepath)
    
    def import_transparent_image(self, context, filepath):
        try:
            # 打开图片
            img = Image.open(filepath)
            if not img.mode == 'RGBA':
                self.report({'ERROR'}, "图片必须包含透明通道(RGBA)")
                return {'CANCELLED'}
            
            # 获取alpha通道
            alpha = np.array(img.getchannel('A'))
            
            # 检查是否完全透明
            if np.all(alpha == 0):
                self.report({'ERROR'}, "图片完全透明")
                return {'CANCELLED'}
            
            # 创建平面
            bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD')
            plane = context.active_object
            
            # 创建材质
            mat = bpy.data.materials.new(name=f"Mat_{os.path.basename(filepath)}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            
            # 设置材质节点
            principled = nodes.get('Principled BSDF')
            tex_image = nodes.new('ShaderNodeTexImage')
            tex_image.image = bpy.data.images.load(filepath)
            
            # 连接节点
            mat.node_tree.links.new(
                tex_image.outputs['Color'],
                principled.inputs['Base Color']
            )
            mat.node_tree.links.new(
                tex_image.outputs['Alpha'],
                principled.inputs['Alpha']
            )
            
            # 设置材质属性
            mat.blend_method = 'HASHED'
            mat.shadow_method = 'HASHED'
            
            # 应用材质到平面
            plane.data.materials.append(mat)
            
            # 调整UV
            plane.scale = (img.width/img.height, 1, 1)
            
            self.report({'INFO'}, "成功导入透明PNG")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"导入失败: {str(e)}")
            return {'CANCELLED'}

classes = (
    TransparentImagePanel,
    ImportTransparentPNG,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register() 