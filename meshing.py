#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


from tqdm import tqdm
import trimesh
import cv2
import numpy as np
import torch
from scene import Scene
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


class Mesh:
    def __init__(
        self,
        v=None,
        f=None,
        vn=None,
        fn=None,
        vt=None,
        ft=None,
        albedo=None,
        vc=None,  # vertex color
        device=None,
    ):
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        # only support a single albedo
        self.albedo = albedo
        # support vertex color is no albedo
        self.vc = vc

        self.ori_center = 0
        self.ori_scale = 1

    @classmethod
    def load(
        cls,
        path=None,
        resize=True,
        renormal=True,
        retex=False,
        front_dir="+z",
        **kwargs,
    ):
        # assume init with kwargs
        if path is None:
            mesh = cls(**kwargs)
        # obj supports face uv
        elif path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs)

        print(f"[Mesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")
        # auto-normalize
        if resize:
            mesh.auto_size()
        # auto-fix normal
        if renormal or mesh.vn is None:
            mesh.auto_normal()
            print(f"[Mesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")
        # auto-fix texcoords
        if retex or (mesh.albedo is not None and mesh.vt is None):
            mesh.auto_uv(cache_path=path)
            print(f"[Mesh loading] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        # rotate front dir to +z
        if front_dir != "+z":
            # axis switch
            if "-z" in front_dir:
                T = torch.tensor(
                    [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            elif "+x" in front_dir:
                T = torch.tensor(
                    [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            elif "-x" in front_dir:
                T = torch.tensor(
                    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            elif "+y" in front_dir:
                T = torch.tensor(
                    [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            elif "-y" in front_dir:
                T = torch.tensor(
                    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            else:
                T = torch.tensor(
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            # rotation (how many 90 degrees)
            if "1" in front_dir:
                T @= torch.tensor(
                    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            elif "2" in front_dir:
                T @= torch.tensor(
                    [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            elif "3" in front_dir:
                T @= torch.tensor(
                    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                    device=mesh.device,
                    dtype=torch.float32,
                )
            mesh.v @= T
            mesh.vn @= T

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None):
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        mtl_path = None

        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            prefix = split_line[0].lower()
            # mtllib
            if prefix == "mtllib":
                mtl_path = split_line[1]
            # usemtl
            elif prefix == "usemtl":
                pass  # ignored
            # v/vn/vt
            elif prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if len(normals) > 0
            else None
        )

        # see if there is vertex color
        use_vertex_color = False
        if mesh.v.shape[1] == 6:
            use_vertex_color = True
            mesh.vc = mesh.v[:, 3:]
            mesh.v = mesh.v[:, :3]
            print(f"[load_obj] use vertex color: {mesh.vc.shape}")

        # try to load texture image
        if not use_vertex_color:
            # try to retrieve mtl file
            mtl_path_candidates = []
            if mtl_path is not None:
                mtl_path_candidates.append(mtl_path)
                mtl_path_candidates.append(
                    os.path.join(os.path.dirname(path), mtl_path)
                )
            mtl_path_candidates.append(path.replace(".obj", ".mtl"))

            mtl_path = None
            for candidate in mtl_path_candidates:
                if os.path.exists(candidate):
                    mtl_path = candidate
                    break

            # if albedo_path is not provided, try retrieve it from mtl
            if mtl_path is not None and albedo_path is None:
                with open(mtl_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0:
                        continue
                    prefix = split_line[0]
                    # NOTE: simply use the first map_Kd as albedo!
                    if "map_Kd" in prefix:
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f"[load_obj] use texture from: {albedo_path}")
                        break

            # still not found albedo_path, or the path doesn't exist
            if albedo_path is None or not os.path.exists(albedo_path):
                # init an empty texture
                print(f"[load_obj] init empty albedo!")
                # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)
                albedo = np.ones((1024, 1024, 3), dtype=np.float32) * np.array(
                    [0.5, 0.5, 0.5]
                )  # default color
            else:
                albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
                albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                albedo = albedo.astype(np.float32) / 255
                print(f"[load_obj] load texture: {albedo.shape}")

                # import matplotlib.pyplot as plt
                # plt.imshow(albedo)
                # plt.show()

            mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)

        return mesh

    @classmethod
    def load_trimesh(cls, path, device=None):
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load ply/glb, assume only has one single RootMesh...
        _data = trimesh.load(path)
        if isinstance(_data, trimesh.Scene):
            if len(_data.geometry) == 1:
                _mesh = list(_data.geometry.values())[0]
            else:
                # manual concat, will lose texture
                _concat = []
                for g in _data.geometry.values():
                    if isinstance(g, trimesh.Trimesh):
                        _concat.append(g)
                _mesh = trimesh.util.concatenate(_concat)
        else:
            _mesh = _data

        if _mesh.visual.kind == "vertex":
            vertex_colors = _mesh.visual.vertex_colors
            vertex_colors = np.array(vertex_colors[..., :3]).astype(np.float32) / 255
            mesh.vc = torch.tensor(vertex_colors, dtype=torch.float32, device=device)
            print(f"[load_trimesh] use vertex color: {mesh.vc.shape}")
        elif _mesh.visual.kind == "texture":
            _material = _mesh.visual.material
            if isinstance(_material, trimesh.visual.material.PBRMaterial):
                texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
            elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
                texture = (
                    np.array(_material.to_pbr().baseColorTexture).astype(np.float32)
                    / 255
                )
            else:
                raise NotImplementedError(
                    f"material type {type(_material)} not supported!"
                )
            mesh.albedo = torch.tensor(texture, dtype=torch.float32, device=device)
            print(f"[load_trimesh] load texture: {texture.shape}")
        else:
            texture = np.ones((1024, 1024, 3), dtype=np.float32) * np.array(
                [0.5, 0.5, 0.5]
            )
            mesh.albedo = torch.tensor(texture, dtype=torch.float32, device=device)
            print(f"[load_trimesh] failed to load texture.")

        vertices = _mesh.vertices

        try:
            texcoords = _mesh.visual.uv
            texcoords[:, 1] = 1 - texcoords[:, 1]
        except Exception as e:
            texcoords = None

        try:
            normals = _mesh.vertex_normals
        except Exception as e:
            normals = None

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if texcoords is not None
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if normals is not None
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    # aabb
    def aabb(self):
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 1.2 / torch.max(vmax - vmin).item()
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self):
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            torch.sum(vn * vn, -1, keepdim=True) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = vn / torch.sqrt(
            torch.clamp(torch.sum(vn * vn, -1, keepdim=True), min=1e-20)
        )  # safe_normalize

        self.vn = vn
        self.fn = self.f

    def auto_uv(self, cache_path=None, vmap=True):
        # try to load cache
        if cache_path is not None:
            cache_path = os.path.splitext(cache_path)[0] + "_uv.npz"
        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np, vmapping = data["vt"], data["ft"], data["vmapping"]
        else:
            import xatlas

            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np, vmapping=vmapping)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)
        self.vt = vt
        self.ft = ft

        if vmap:
            # remap v/f to vt/ft, so each v correspond to a unique vt. (necessary for gltf)
            vmapping = (
                torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
            )
            self.align_v_to_vt(vmapping)

    def align_v_to_vt(self, vmapping=None):
        # remap v/f and vn/vn to vt/ft.
        if vmapping is None:
            ft = self.ft.view(-1).long()
            f = self.f.view(-1).long()
            vmapping = torch.zeros(
                self.vt.shape[0], dtype=torch.long, device=self.device
            )
            vmapping[ft] = f  # scatter, randomly choose one if index is not unique

        self.v = self.v[vmapping]
        self.f = self.ft
        # assume fn == f
        if self.vn is not None:
            self.vn = self.vn[vmapping]
            self.fn = self.ft

    def to(self, device):
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self

    def write(self, path):
        if path.endswith(".ply"):
            self.write_ply(path)
        elif path.endswith(".obj"):
            self.write_obj(path)
        elif path.endswith(".glb") or path.endswith(".gltf"):
            self.write_glb(path)
        else:
            raise NotImplementedError(f"format {path} not supported!")

    # write to ply file (only geom)
    def write_ply(self, path):
        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)

    # write to gltf/glb file (geom + texture)
    def write_glb(self, path):
        assert (
            self.vn is not None and self.vt is not None
        )  # should be improved to support export without texture...

        # assert self.v.shape[0] == self.vn.shape[0] and self.v.shape[0] == self.vt.shape[0]
        if self.v.shape[0] != self.vt.shape[0]:
            self.align_v_to_vt()

        # assume f == fn == ft

        import pygltflib

        f_np = self.f.detach().cpu().numpy().astype(np.uint32)
        v_np = self.v.detach().cpu().numpy().astype(np.float32)
        # vn_np = self.vn.detach().cpu().numpy().astype(np.float32)
        vt_np = self.vt.detach().cpu().numpy().astype(np.float32)

        albedo = self.albedo.detach().cpu().numpy()
        albedo = (albedo * 255).astype(np.uint8)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)

        f_np_blob = f_np.flatten().tobytes()
        v_np_blob = v_np.tobytes()
        # vn_np_blob = vn_np.tobytes()
        vt_np_blob = vt_np.tobytes()
        albedo_blob = cv2.imencode(".png", albedo)[1].tobytes()

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            # indices to accessors (0 is triangles)
                            attributes=pygltflib.Attributes(
                                POSITION=1,
                                TEXCOORD_0=2,
                            ),
                            indices=0,
                            material=0,
                        )
                    ]
                )
            ],
            materials=[
                pygltflib.Material(
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                    alphaCutoff=0,
                    doubleSided=True,
                )
            ],
            textures=[
                pygltflib.Texture(sampler=0, source=0),
            ],
            samplers=[
                pygltflib.Sampler(
                    magFilter=pygltflib.LINEAR,
                    minFilter=pygltflib.LINEAR_MIPMAP_LINEAR,
                    wrapS=pygltflib.REPEAT,
                    wrapT=pygltflib.REPEAT,
                ),
            ],
            images=[
                # use embedded (buffer) image
                pygltflib.Image(bufferView=3, mimeType="image/png"),
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(f_np_blob)
                    + len(v_np_blob)
                    + len(vt_np_blob)
                    + len(albedo_blob)
                )
            ],
            # buffer view (based on dtype)
            bufferViews=[
                # triangles; as flatten (element) array
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(f_np_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,  # GL_ELEMENT_ARRAY_BUFFER (34963)
                ),
                # positions; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob),
                    byteLength=len(v_np_blob),
                    byteStride=12,  # vec3
                    target=pygltflib.ARRAY_BUFFER,  # GL_ARRAY_BUFFER (34962)
                ),
                # texcoords; as vec2 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob),
                    byteLength=len(vt_np_blob),
                    byteStride=8,  # vec2
                    target=pygltflib.ARRAY_BUFFER,
                ),
                # texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob) + len(vt_np_blob),
                    byteLength=len(albedo_blob),
                ),
            ],
            accessors=[
                # 0 = triangles
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT,  # GL_UNSIGNED_INT (5125)
                    count=f_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(f_np.max())],
                    min=[int(f_np.min())],
                ),
                # 1 = positions
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,  # GL_FLOAT (5126)
                    count=len(v_np),
                    type=pygltflib.VEC3,
                    max=v_np.max(axis=0).tolist(),
                    min=v_np.min(axis=0).tolist(),
                ),
                # 2 = texcoords
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(vt_np),
                    type=pygltflib.VEC2,
                    max=vt_np.max(axis=0).tolist(),
                    min=vt_np.min(axis=0).tolist(),
                ),
            ],
        )

        # set actual data
        gltf.set_binary_blob(f_np_blob + v_np_blob + vt_np_blob + albedo_blob)

        # glb = b"".join(gltf.save_to_bytes())
        gltf.save(path)

    # write to obj file (geom + texture)
    def write_obj(self, path):
        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]} {1 - v[1]} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]} {v[1]} {v[2]} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                )

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")

        if self.albedo is not None:
            albedo = self.albedo.detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))


# https://github.com/dreamgaussian/dreamgaussian
def extract_mesh(
    gaussians, density_thresh, resolution=512, decimate_target=1e5, device="cuda"
):
    @torch.no_grad()
    def extract_fields(
        gaussians, resolution=128, num_blocks=16, relax_ratio=1.5, device="cuda"
    ):
        def gaussian_3d_coeff(xyzs, covs):
            # xyzs: [N, 3]
            # covs: [N, 6]
            x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
            a, b, c, d, e, f = (
                covs[:, 0],
                covs[:, 1],
                covs[:, 2],
                covs[:, 3],
                covs[:, 4],
                covs[:, 5],
            )

            # eps must be small enough !!!
            inv_det = 1 / (
                a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24
            )
            inv_a = (d * f - e**2) * inv_det
            inv_b = (e * c - b * f) * inv_det
            inv_c = (e * b - c * d) * inv_det
            inv_d = (a * f - c**2) * inv_det
            inv_e = (b * c - e * a) * inv_det
            inv_f = (a * d - b**2) * inv_det

            power = (
                -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f)
                - x * y * inv_b
                - x * z * inv_c
                - y * z * inv_e
            )

            power[power > 0] = -1e10  # abnormal values... make weights 0

            return torch.exp(power)

        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = gaussians.get_opacity
        print(opacities.size())

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask].to(device)
        xyzs = gaussians.get_xyz[mask].to(device)
        stds = gaussians.get_scaling[mask].to(device)

        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        center = (mn + mx) / 2
        scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - center) * scale
        stds = stds * scale

        covs = gaussians.covariance_activation(
            stds, 1, gaussians._rotation[mask].to(device)
        ).to(device)

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        # every block's resolution = split_size
        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)

        for xi, xs in enumerate(tqdm(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    # sample points [M, 3]
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask]  # [L, 3]
                    mask_covs = covs[mask]  # [L, 6]
                    mask_opas = opacities[mask].view(1, -1)  # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(
                        1, mask_covs.shape[0], 1
                    ) - mask_xyzs.unsqueeze(
                        0
                    )  # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(
                        pts.shape[0], 1, 1
                    )  # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(
                            g_pts[:, start:end].cuda().reshape(-1, 3),
                            g_covs[:, start:end].cuda().reshape(-1, 6),
                        ).reshape(
                            pts.shape[0], -1
                        )  # [M, l]
                        val += (mask_opas[:, start:end].cuda() * w).sum(-1).cpu()

                    # kiui.lo(val, mask_opas, w)

                    occ[
                        xi * split_size : xi * split_size + len(xs),
                        yi * split_size : yi * split_size + len(ys),
                        zi * split_size : zi * split_size + len(zs),
                    ] = val.reshape(len(xs), len(ys), len(zs))

        # kiui.lo(occ, verbose=1)
        return occ, center, scale

    def clean_mesh(
        verts,
        faces,
        v_pct=1,
        min_f=64,
        min_d=20,
        repair=True,
        remesh=True,
        remesh_size=0.01,
    ):
        # verts: [N, 3]
        # faces: [N, 3]

        _ori_vert_shape = verts.shape
        _ori_face_shape = faces.shape

        import pymeshlab as pml

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

        if v_pct > 0:
            ms.meshing_merge_close_vertices(
                threshold=pml.Percentage(v_pct)
            )  # 1/10000 of bounding box diagonal

        ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
        ms.meshing_remove_null_faces()  # faces with area == 0

        if min_d > 0:
            ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pml.Percentage(min_d)
            )

        if min_f > 0:
            ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

        if repair:
            # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
            ms.meshing_repair_non_manifold_edges(method=0)
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.AbsoluteValue(remesh_size)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

        print(
            f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
        )

        return verts, faces

    def decimate_mesh(
        verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
    ):
        # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

        _ori_vert_shape = verts.shape
        _ori_face_shape = faces.shape

        if backend == "pyfqmr":
            import pyfqmr

            solver = pyfqmr.Simplify()
            solver.setMesh(verts, faces)
            solver.simplify_mesh(
                target_count=target, preserve_border=False, verbose=False
            )
            verts, faces, normals = solver.getMesh()
        else:
            import pymeshlab as pml

            m = pml.Mesh(verts, faces)
            ms = pml.MeshSet()
            ms.add_mesh(m, "mesh")  # will copy!

            # filters
            # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=int(target), optimalplacement=optimalplacement
            )

            if remesh:
                # ms.apply_coord_taubin_smoothing()
                ms.meshing_isotropic_explicit_remeshing(
                    iterations=3, targetlen=pml.Percentage(1)
                )

            # extract mesh
            m = ms.current_mesh()
            verts = m.vertex_matrix()
            faces = m.face_matrix()

        print(
            f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
        )

        return verts, faces

    occ, center, scale = extract_fields(gaussians, resolution, device=device)
    print(f"occ: {occ.shape}")

    import mcubes

    vertices, triangles = mcubes.marching_cubes(
        occ.detach().cpu().numpy(), density_thresh
    )
    vertices = vertices / (resolution - 1.0) * 2 - 1

    # transform back to the original space
    vertices = vertices / scale + center.detach().cpu().numpy()

    print(f"marching cubes: {vertices.shape} {triangles.shape}")

    # vertices, triangles = clean_mesh(
    #     vertices, triangles, remesh=True, remesh_size=0.015
    # )
    # if decimate_target > 0 and triangles.shape[0] > decimate_target:
    #     vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

    v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
    f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

    print(f"clean: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}")
    mesh = Mesh(v=v, f=f, device="cuda")
    return mesh


def meshing(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    mesh_shape="./outs/gs_shape.obj",
    density_thresh=1,
    device="cuda",
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        mesh = extract_mesh(scene.gaussians, density_thresh, device=device)
        os.makedirs(os.path.dirname(mesh_shape), exist_ok=True)
        mesh.write(mesh_shape)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.device)

    meshing(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        device=args.data_device,
    )
