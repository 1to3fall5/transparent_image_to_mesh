bl_info = {
    "name": "透明PNG转模型",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (4, 2, 3),
    "location": "View3D > Sidebar > 透明PNG",
    "description": "将透明PNG图片转换为裁剪后的平面模型",
    "category": "Import-Export",
}

# 检查必要的依赖
try:
    import numpy as np
    from PIL import Image
    import cv2
except ImportError as e:
    raise ImportError(
        "\n必须安装以下Python包才能使用此插件："
        "\n - numpy"
        "\n - pillow (PIL)"
        "\n - opencv-python"
        "\n\n请使用以下命令安装："
        "\npip install numpy pillow opencv-python"
    ) from e

import bpy
from bpy.types import Panel, Operator
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
import os
import bmesh

class TransparentImagePanel(Panel):
    bl_label = "透明PNG"
    bl_idname = "VIEW3D_PT_transparent_png"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '透明PNG'

    def draw(self, context):
        layout = self.layout
        
        # 导入相关的box
        import_box = layout.box()
        import_box.operator("import.transparent_png", text="导入PNG")
        
        # 创建裁剪设置的box
        settings_box = layout.box()
        col = settings_box.column(align=True)
        
        # 裁剪设置
        scene = context.scene
        # 添加属性到场景中
        if not hasattr(scene, "transparent_threshold"):
            bpy.types.Scene.transparent_threshold = bpy.props.FloatProperty(
                name="透明度阈值",
                description="小于此值的透明度将被裁剪（0-1）",
                default=0.1,
                min=0.0,
                max=1.0
            )
        if not hasattr(scene, "crop_precision"):
            bpy.types.Scene.crop_precision = bpy.props.IntProperty(
                name="裁剪精度",
                description="裁剪边缘的精度（点数）",
                default=32,
                min=8,
                max=256
            )
        # 在面板中显示设置
        col.prop(scene, "transparent_threshold")
        col.prop(scene, "crop_precision")
        
        # 裁剪按钮单独放在最下方的box中
        crop_box = layout.box()
        crop_box.operator("object.crop_transparent", text="裁剪透明区域")

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
            pil_img = Image.open(filepath)
            
            # 如果不是RGBA模式，转换为RGBA
            if pil_img.mode != 'RGBA':
                pil_img = pil_img.convert('RGBA')
                self.report({'INFO'}, "图片没有透明通道，将使用完全不透明的效果")
            
            # 获取alpha通道
            alpha = np.array(pil_img.getchannel('A'))
            
            # 检查是否完全透明
            if np.all(alpha == 0):
                self.report({'ERROR'}, "图片完全透明")
                return {'CANCELLED'}
            
            # 获取非透明区域的边界
            rows = np.any(alpha > 0, axis=1)
            cols = np.any(alpha > 0, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # 裁剪图片
            cropped_img = pil_img.crop((xmin, ymin, xmax + 1, ymax + 1))
            
            # 保存裁剪后的图片到临时文件
            temp_filepath = os.path.join(os.path.dirname(filepath), 
                                       f"temp_{os.path.basename(filepath)}")
            cropped_img.save(temp_filepath)
            
            # 创建平面
            bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD')
            plane = context.active_object
            
            # 生成唯一的材质名称
            base_name = os.path.basename(filepath)
            mat_name = f"Mat_{base_name}"
            counter = 1
            while mat_name in bpy.data.materials:
                mat_name = f"Mat_{base_name}_{counter}"
                counter += 1
            
            # 创建新材质
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            
            # 设置材质节点
            principled = nodes.get('Principled BSDF')
            tex_image = nodes.new('ShaderNodeTexImage')
            
            # 检查图片是否已经加载
            img_name = os.path.basename(filepath)
            if img_name in bpy.data.images:
                # 如果图片已存在，创建一个新的副本
                blender_img = bpy.data.images[img_name].copy()
                blender_img.filepath = filepath
                blender_img.reload()
            else:
                blender_img = bpy.data.images.load(filepath)
            
            tex_image.image = blender_img
            
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
            
            # 应用材质到平面
            plane.data.materials.append(mat)
            
            # 调整UV以匹配裁剪后的图片比例
            cropped_width = xmax - xmin + 1
            cropped_height = ymax - ymin + 1
            plane.scale = (cropped_width/cropped_height, 1, 1)
            
            # 删除临时文件
            os.remove(temp_filepath)
            
            # 计算原始图片中心点的偏移
            center_x = (xmin + xmax) / 2 - pil_img.size[0] / 2
            center_y = (ymin + ymax) / 2 - pil_img.size[1] / 2
            
            # 调整平面位置以匹配原始图片中的位置
            scale_factor = 1 / pil_img.size[1]
            plane.location.x = center_x * scale_factor
            plane.location.y = center_y * scale_factor
            
            self.report({'INFO'}, "成功导入并裁剪透明PNG")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"导入失败: {str(e)}")
            return {'CANCELLED'}

class CropTransparentOperator(Operator):
    bl_idname = "object.crop_transparent"
    bl_label = "裁剪透明区域"
    bl_options = {'REGISTER', 'UNDO'}
    
    # 只保留需要在Redo面板中显示的属性
    threshold: bpy.props.FloatProperty(
        name="透明度阈值",
        description="小于此值的透明度将被裁剪（0-1）",
        default=0.1,
        min=0.0,
        max=1.0
    )
    
    precision: bpy.props.IntProperty(
        name="裁剪精度",
        description="裁剪边缘的精度（点数）",
        default=32,
        min=8,
        max=256
    )
    
    # 将内部属性改名，移除下划线前缀
    original_dimension_x: bpy.props.FloatProperty(default=0.0, options={'HIDDEN'})
    original_dimension_y: bpy.props.FloatProperty(default=0.0, options={'HIDDEN'})
    original_dimension_z: bpy.props.FloatProperty(default=0.0, options={'HIDDEN'})
    original_scale_x: bpy.props.FloatProperty(default=0.0, options={'HIDDEN'})
    original_scale_y: bpy.props.FloatProperty(default=0.0, options={'HIDDEN'})
    original_scale_z: bpy.props.FloatProperty(default=0.0, options={'HIDDEN'})
    first_run: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        self.threshold = context.scene.transparent_threshold
        self.precision = context.scene.crop_precision
        self.first_run = True
        return self.execute(context)

    def execute(self, context):
        # 更新场景设置
        context.scene.transparent_threshold = self.threshold
        context.scene.crop_precision = self.precision
        
        obj = context.active_object
        if not obj or not obj.type == 'MESH' or not obj.data.materials:
            self.report({'ERROR'}, "请先选择一个带材质的平面物体")
            return {'CANCELLED'}

        mat = obj.data.materials[0]
        if not mat or not mat.use_nodes:
            self.report({'ERROR'}, "物体没有正确的材质设置")
            return {'CANCELLED'}

        # 查找图片节点
        image_node = None
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                image_node = node
                break

        if not image_node:
            self.report({'ERROR'}, "未找到图片节点")
            return {'CANCELLED'}

        try:
            # 如果是第一次运行，保存原尺寸
            if self.first_run:
                self.original_dimension_x = obj.dimensions[0]
                self.original_dimension_y = obj.dimensions[1]
                self.original_dimension_z = obj.dimensions[2]
                self.original_scale_x = obj.scale[0]
                self.original_scale_y = obj.scale[1]
                self.original_scale_z = obj.scale[2]
                self.first_run = False
            
            # 获取图片数据
            img = image_node.image
            if not img:
                self.report({'ERROR'}, "未找到有效图片")
                return {'CANCELLED'}

            # 获取图片像素数据
            pixels = np.array(img.pixels[:])
            width = img.size[0]
            height = img.size[1]
            
            # 重塑数组为RGBA格式
            pixels = pixels.reshape(height, width, 4)
            
            # 获取alpha通道
            alpha = pixels[:, :, 3]
            
            # 根据阈值创建mask - 使用操作器的阈值
            mask = alpha > self.threshold
            
            # 找到非透明区域的轮廓
            import cv2
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.report({'ERROR'}, "未找到有效轮廓")
                return {'CANCELLED'}
            
            # 获取最大轮廓
            contour = max(contours, key=cv2.contourArea)
            
            # 简化轮廓点 - 使用操作器的精度
            epsilon = cv2.arcLength(contour, True) * (1.0 / self.precision)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 创建新的网格数据
            mesh = obj.data
            bm = bmesh.new()
            
            # 创建UV层
            uv_layer = bm.loops.layers.uv.new()
            
            # 添加顶点
            vertices = []
            for point in approx:
                u = point[0][0] / width
                v = point[0][1] / height
                
                # 使用保存的原始尺寸
                x = (u - 0.5) * self.original_dimension_x / self.original_scale_x
                y = (v - 0.5) * self.original_dimension_y / self.original_scale_y
                vert = bm.verts.new((x, y, 0))
                vertices.append(vert)
            
            # 确保顶点顺序正确（逆时针）
            if len(vertices) > 2:
                face = bm.faces.new(vertices)
                
                # 设置UV坐标时也使用保存的尺寸
                for loop in face.loops:
                    v = loop.vert
                    u = (v.co.x * self.original_scale_x / self.original_dimension_x + 0.5)
                    v = (v.co.y * self.original_scale_y / self.original_dimension_y + 0.5)
                    loop[uv_layer].uv = (u, v)
            
            # 保存当前模式
            current_mode = obj.mode
            
            # 更新网格
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()
            
            # 只在第一次运行时重新计算法线
            if self.first_run:
                # 临时切换到编辑模式
                if current_mode != 'EDIT':
                    bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.normals_make_consistent(inside=False)
                # 恢复原始模式
                if current_mode != 'EDIT':
                    bpy.ops.object.mode_set(mode=current_mode)

            self.report({'INFO'}, "裁剪完成")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"裁剪失败: {str(e)}")
            return {'CANCELLED'}

classes = (
    TransparentImagePanel,
    ImportTransparentPNG,
    CropTransparentOperator,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # 注册场景属性
    if not hasattr(bpy.types.Scene, "transparent_threshold"):
        bpy.types.Scene.transparent_threshold = bpy.props.FloatProperty(
            name="透明度阈值",
            description="小于此值的透明度将被裁剪（0-1）",
            default=0.1,
            min=0.0,
            max=1.0
        )
    if not hasattr(bpy.types.Scene, "crop_precision"):
        bpy.types.Scene.crop_precision = bpy.props.IntProperty(
            name="裁剪精度",
            description="裁剪边缘的精度（点数）",
            default=32,
            min=8,
            max=256
        )

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    # 注销场景属性
    if hasattr(bpy.types.Scene, "transparent_threshold"):
        del bpy.types.Scene.transparent_threshold
    if hasattr(bpy.types.Scene, "crop_precision"):
        del bpy.types.Scene.crop_precision

if __name__ == "__main__":
    register() 