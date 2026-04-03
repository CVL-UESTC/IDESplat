from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)














# from math import isqrt
# from typing import Literal, Optional

# import torch
# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
# from einops import einsum, rearrange, repeat
# from jaxtyping import Float
# from torch import Tensor

# from ...geometry.projection import get_fov, homogenize_points


# # ---------- 工具 ----------
# def _infer_out_dtype(
#     out_dtype: Optional[torch.dtype],
#     *refs: Tensor,
# ) -> torch.dtype:
#     """
#     推断输出 dtype 的策略（优先级从高到低）：
#     1) 用户显示指定 out_dtype；
#     2) 若外层处于 autocast，使用当前 autocast 的 GPU dtype（fp16/bf16）；
#     3) 否则参考输入张量中第一个浮点 dtype；
#     4) 回退 torch.float32。
#     """
#     if out_dtype is not None:
#         return out_dtype

#     try:
#         if torch.is_autocast_enabled():
#             try:
#                 return torch.get_autocast_gpu_dtype()
#             except Exception:
#                 return torch.get_autocast_dtype()
#     except Exception:
#         pass

#     for t in refs:
#         if isinstance(t, torch.Tensor) and t.is_floating_point():
#             return t.dtype

#     return torch.float32


# # ---------- 透视投影矩阵 ----------
# def get_projection_matrix(
#     near: Float[Tensor, " batch"],
#     far: Float[Tensor, " batch"],
#     fov_x: Float[Tensor, " batch"],
#     fov_y: Float[Tensor, " batch"],
# ) -> Float[Tensor, "batch 4 4"]:
#     """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z axis."""
#     # 保持来什么 dtype 就用什么 dtype（避免提前转）
#     dtype = near.dtype
#     device = near.device

#     tan_fov_x = (0.5 * fov_x).tan()
#     tan_fov_y = (0.5 * fov_y).tan()

#     top = tan_fov_y * near
#     bottom = -top
#     right = tan_fov_x * near
#     left = -right

#     (b,) = near.shape
#     result = torch.zeros((b, 4, 4), dtype=dtype, device=device)
#     result[:, 0, 0] = 2 * near / (right - left)
#     result[:, 1, 1] = 2 * near / (top - bottom)
#     result[:, 0, 2] = (right + left) / (right - left)
#     result[:, 1, 2] = (top + bottom) / (top - bottom)
#     result[:, 3, 2] = 1.0
#     result[:, 2, 2] = far / (far - near)
#     result[:, 2, 3] = -(far * near) / (far - near)
#     return result


# # ---------- 透视渲染 ----------
# def render_cuda(
#     extrinsics: Float[Tensor, "batch 4 4"],
#     intrinsics: Float[Tensor, "batch 3 3"],
#     near: Float[Tensor, " batch"],
#     far: Float[Tensor, " batch"],
#     image_shape: tuple[int, int],
#     background_color: Float[Tensor, "batch 3"],
#     gaussian_means: Float[Tensor, "batch gaussian 3"],
#     gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
#     gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
#     gaussian_opacities: Float[Tensor, "batch gaussian"],
#     scale_invariant: bool = True,
#     use_sh: bool = True,
#     out_dtype: Optional[torch.dtype] = None,  # 若 None，将自动匹配外层 dtype
# ) -> Float[Tensor, "batch 3 height width"]:
#     assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

#     # --- 输出 dtype 推断（保持与外层一致） ---
#     desired_dtype = _infer_out_dtype(
#         out_dtype,
#         background_color, intrinsics, extrinsics
#     )

#     # --- 这里不统一转 FP32，让计算沿用外层 dtype ---
#     if scale_invariant:
#         scale = 1.0 / near
#         extrinsics = extrinsics.clone()
#         extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
#         gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
#         gaussian_means = gaussian_means * scale[:, None, None]
#         near = near * scale
#         far = far * scale

#     _, _, _, n = gaussian_sh_coefficients.shape
#     degree = isqrt(n) - 1
#     shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

#     b, _, _ = extrinsics.shape
#     h, w = image_shape

#     # FOV & 矩阵计算（保持外层 dtype）
#     fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
#     tan_fov_x = (0.5 * fov_x).tan()
#     tan_fov_y = (0.5 * fov_y).tan()

#     projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
#     projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
#     view_matrix = rearrange(extrinsics.float().inverse(), "b i j -> b j i")
#     full_projection = (view_matrix @ projection_matrix)

#     all_images = []

#     # 索引保持在 CPU（PyTorch 对 GPU 索引支持有限，CPU 更通用）
#     row, col = torch.triu_indices(3, 3)

#     for i in range(b):
#         # 屏幕空间梯度；如果不需要梯度可改为 requires_grad=False
#         mean_gradients = torch.zeros_like(
#             gaussian_means[i], dtype=torch.float32, requires_grad=True, device=gaussian_means[i].device
#         )

#         # —— 仅在“传入设置/内核前最后一跳”转为 float32 —— #
#         settings = GaussianRasterizationSettings(
#             image_height=int(h),
#             image_width=int(w),
#             tanfovx=float(tan_fov_x[i].float().item()),
#             tanfovy=float(tan_fov_y[i].float().item()),
#             bg=background_color[i].float(),
#             scale_modifier=1.0,
#             viewmatrix=view_matrix[i].float(),
#             projmatrix=full_projection[i].float(),
#             sh_degree=int(degree),
#             campos=extrinsics[i, :3, 3].float(),
#             prefiltered=False,
#             debug=False,
#         )
#         rasterizer = GaussianRasterizer(settings)

#         means3D = gaussian_means[i].float()
#         means2D = mean_gradients.float()
#         if use_sh:
#             shs_i = shs[i].float()
#             colors_precomp = None
#         else:
#             shs_i = None
#             colors_precomp = shs[i, :, 0, :].float()
#         opacities = gaussian_opacities[i, ..., None].float()
#         cov3D = gaussian_covariances[i, :, row, col].float()

#         image, _ = rasterizer(
#             means3D=means3D,
#             means2D=means2D,
#             shs=shs_i,
#             colors_precomp=colors_precomp,
#             opacities=opacities,
#             cov3D_precomp=cov3D,
#         )

#         all_images.append(image)

#     out = torch.stack(all_images)  # 此时为 FP32（内核输出）
#     if out.dtype != desired_dtype:
#         out = out.to(desired_dtype)
#     return out


# # ---------- 伪正交渲染 ----------
# def render_cuda_orthographic(
#     extrinsics: Float[Tensor, "batch 4 4"],
#     width: Float[Tensor, " batch"],
#     height: Float[Tensor, " batch"],
#     near: Float[Tensor, " batch"],
#     far: Float[Tensor, " batch"],
#     image_shape: tuple[int, int],
#     background_color: Float[Tensor, "batch 3"],
#     gaussian_means: Float[Tensor, "batch gaussian 3"],
#     gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
#     gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
#     gaussian_opacities: Float[Tensor, "batch gaussian"],
#     fov_degrees: float = 0.1,
#     use_sh: bool = True,
#     dump: dict | None = None,
#     out_dtype: Optional[torch.dtype] = None,  # 若 None，将自动匹配外层 dtype
# ) -> Float[Tensor, "batch 3 height width"]:

#     desired_dtype = _infer_out_dtype(out_dtype, background_color, extrinsics, width)

#     # 维持外层 dtype 计算
#     b, _, _ = extrinsics.shape
#     h, w = image_shape
#     assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

#     _, _, _, n = gaussian_sh_coefficients.shape
#     degree = isqrt(n) - 1
#     shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

#     # 伪正交投影参数（跟随 extrinsics 的 dtype/device）
#     fov_x = torch.tensor(fov_degrees, device=extrinsics.device, dtype=extrinsics.dtype).deg2rad()
#     tan_fov_x = (0.5 * fov_x).tan()
#     distance_to_near = (0.5 * width) / tan_fov_x
#     tan_fov_y = 0.5 * height / distance_to_near
#     fov_y = (2 * tan_fov_y).atan()
#     near = near + distance_to_near
#     far = far + distance_to_near

#     move_back = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
#     move_back[2, 3] = -distance_to_near
#     extrinsics = extrinsics @ move_back

#     if dump is not None:
#         dump["extrinsics"] = extrinsics
#         dump["fov_x"] = fov_x
#         dump["fov_y"] = fov_y
#         dump["near"] = near
#         dump["far"] = far

#     projection_matrix = get_projection_matrix(
#         near, far, repeat(fov_x, "-> b", b=b), fov_y
#     )
#     projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
#     view_matrix = rearrange(extrinsics.float().inverse(), "b i j -> b j i")
#     full_projection = (view_matrix @ projection_matrix)

#     all_images = []
#     all_radii = []

#     row, col = torch.triu_indices(3, 3)

#     for i in range(b):
#         mean_gradients = torch.zeros_like(
#             gaussian_means[i], dtype=torch.float32, requires_grad=True, device=gaussian_means[i].device
#         )

#         settings = GaussianRasterizationSettings(
#             image_height=int(h),
#             image_width=int(w),
#             tanfovx=float(tan_fov_x.float().item()),
#             tanfovy=float(tan_fov_y.float().item()),
#             bg=background_color[i].float(),
#             scale_modifier=1.0,
#             viewmatrix=view_matrix[i].float(),
#             projmatrix=full_projection[i].float(),
#             sh_degree=int(degree),
#             campos=extrinsics[i, :3, 3].float(),
#             prefiltered=False,
#             debug=False,
#         )
#         rasterizer = GaussianRasterizer(settings)

#         means3D = gaussian_means[i].float()
#         means2D = mean_gradients.float()
#         if use_sh:
#             shs_i = shs[i].float()
#             colors_precomp = None
#         else:
#             shs_i = None
#             colors_precomp = shs[i, :, 0, :].float()
#         opacities = gaussian_opacities[i, ..., None].float()
#         cov3D = gaussian_covariances[i, :, row, col].float()

#         image, radii = rasterizer(
#             means3D=means3D,
#             means2D=means2D,
#             shs=shs_i,
#             colors_precomp=colors_precomp,
#             opacities=opacities,
#             cov3D_precomp=cov3D,
#         )

#         all_images.append(image)
#         all_radii.append(radii)

#     out = torch.stack(all_images)  # FP32
#     if out.dtype != desired_dtype:
#         out = out.to(desired_dtype)
#     return out


# DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


# # ---------- 深度渲染 ----------
# def render_depth_cuda(
#     extrinsics: Float[Tensor, "batch 4 4"],
#     intrinsics: Float[Tensor, "batch 3 3"],
#     near: Float[Tensor, " batch"],
#     far: Float[Tensor, " batch"],
#     image_shape: tuple[int, int],
#     gaussian_means: Float[Tensor, "batch gaussian 3"],
#     gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
#     gaussian_opacities: Float[Tensor, "batch gaussian"],
#     scale_invariant: bool = True,
#     mode: DepthRenderingMode = "depth",
#     out_dtype: Optional[torch.dtype] = None,  # 若 None，将自动匹配外层 dtype
# ) -> Float[Tensor, "batch height width"]:

#     desired_dtype = _infer_out_dtype(out_dtype, intrinsics, extrinsics)

#     # 相机空间（保持外层 dtype 计算）
#     camera_space_gaussians = einsum(
#         extrinsics.float().inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
#     )
#     fake_color = camera_space_gaussians[..., 2]

#     if mode == "disparity":
#         fake_color = 1 / fake_color
#     elif mode == "log":
#         fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

#     b, _ = fake_color.shape

#     color3 = repeat(fake_color, "b g -> b g c ()", c=3)
#     result = render_cuda(
#         extrinsics,
#         intrinsics,
#         near,
#         far,
#         image_shape,
#         torch.zeros((b, 3), dtype=torch.float32, device=fake_color.device),  # 背景可直接给 FP32
#         gaussian_means,
#         gaussian_covariances,
#         color3,                      # 用伪颜色（外层 dtype），传入时内部会转 float()
#         gaussian_opacities,
#         scale_invariant=scale_invariant,
#         use_sh=False,
#         out_dtype=torch.float32,     # 先输出 FP32，最后统一回外层 dtype
#     )

#     out = result.mean(dim=1)  # [B, H, W]，当前为 FP32
#     if out.dtype != desired_dtype:
#         out = out.to(desired_dtype)
#     return out
