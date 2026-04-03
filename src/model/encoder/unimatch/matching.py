import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


# def warp_with_pose_depth_candidates(
#     feature1,
#     intrinsics,
#     pose,
#     depth,
#     clamp_min_depth=1e-3,
#     grid_sample_disable_cudnn=False,
# ):
#     """
#     feature1: [B, C, H, W]
#     intrinsics: [B, 3, 3]
#     pose: [B, 4, 4]
#     depth: [B, D, H, W]
#     """
#
#     assert intrinsics.size(1) == intrinsics.size(2) == 3
#     assert pose.size(1) == pose.size(2) == 4
#     assert depth.dim() == 4
#
#     b, d, h, w = depth.size()
#     c = feature1.size(1)
#
#     with torch.no_grad():
#         # pixel coordinates
#         grid = coords_grid(
#             b, h, w, homogeneous=True, device=depth.device
#         )  # [B, 3, H, W]
#         # back project to 3D and transform viewpoint
#
#         points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
#         points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(1, 1, d, 1) * depth.view(b, 1, d, h * w)  # [B, 3, D, H*W]
#         points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
#         # reproject to 2D image plane
#         points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(b, 3, d, h * w)  # [B, 3, D, H*W]
#         pixel_coords = points[:, :2] / points[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]
#
#         pixel_coords = pixel_coords.float()
#         print("pixel_coords: ", pixel_coords.min(), pixel_coords.max())
#         # normalize to [-1, 1]
#         x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
#         y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1
#
#         grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]
#         print("grid: ", grid.min(), grid.max())
#
#     # sample features
#     # ref: https://github.com/pytorch/pytorch/issues/88380
#     # print(feature1.shape, grid.shape)
#     # hardcoded workaround
#     if feature1.numel() > 1000000:
#         grid_sample_disable_cudnn = True
#     with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
#         warped_feature = F.grid_sample(
#             feature1,
#             grid.view(b, d * h, w, 2),
#             mode="bilinear",
#             padding_mode="zeros",
#             align_corners=True,).view(b, c, d, h, w)  # [B, C, D, H, W]
#
#     return warped_feature


def warp_with_pose_depth_candidates(
        feature1,
        intrinsics,
        pose,
        depth,
        clamp_min_depth=1e-3,
        grid_sample_disable_cudnn=False,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.autocast("cuda", enabled=False), torch.no_grad():
        # pixel coordinates (显式 FP32)
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        ).float()  # [B, 3, H, W]

        # —— 用 pinv 更稳，处理极小/近奇异内参；统一到 FP32 ——
        K = intrinsics.float().contiguous()
        Kinv = torch.linalg.pinv(K)  # [B,3,3]

        # back project to 3D and transform viewpoint
        points = Kinv.bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        R = pose[:, :3, :3].float()
        t = pose[:, :3, -1:].float()

        points = torch.bmm(R, points).unsqueeze(2).repeat(1, 1, d, 1) * depth.float().view(b, 1, d, h * w)  # [B,3,D,H*W]
        points = points + t.unsqueeze(-1)  # [B,3,D,H*W]

        # reproject to 2D image plane
        cam = K.bmm(points.view(b, 3, -1)).view(b, 3, d, h * w)  # [B,3,D,H*W]
        z = cam[:, -1:]  # [B,1,D,H*W]
        denom = z.clamp(min=clamp_min_depth)  # 仅为避免除零；不改原逻辑

        pixel_coords = cam[:, :2] / denom  # [B,2,D,H*W]
        # —— 关键：清理 inf/NaN，避免传入 grid_sample ——
        pixel_coords = torch.nan_to_num(pixel_coords, nan=0.0, posinf=0.0, neginf=0.0)

        # print("pixel_coords: ", pixel_coords.min(), pixel_coords.max())

        # normalize to [-1, 1]
        x_grid = 2.0 * pixel_coords[:, 0] / max(w - 1, 1) - 1.0
        y_grid = 2.0 * pixel_coords[:, 1] / max(h - 1, 1) - 1.0
        # print("x_grid: ", x_grid.min(), x_grid.max())
        # print("y_grid: ", y_grid.min(), y_grid.max())

        # 再次保底，确保 grid 完全有限
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=0.0, neginf=0.0)
        y_grid = torch.nan_to_num(y_grid, nan=0.0, posinf=0.0, neginf=0.0)

        grid = torch.stack([x_grid, y_grid], dim=-1).contiguous()  # [B, D, H*W, 2] (float32)
        grid = grid.view(b, d * h, w, 2)
        # print("grid 1 : ", grid.view(b, d, h, w, 2)[0, :, 10, 10, :])
        # print("grid 2 : ", grid.view(b, d, h, w, 2)[0, :, 30, 30, :])
        # print("grid 3 : ", grid.view(b, d, h, w, 2)[0, :, 50, 50, :])

    # sample features（保持原逻辑）
    if feature1.numel() > 1000000:
        grid_sample_disable_cudnn = True

    with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
        warped_feature = F.grid_sample(
            feature1,
            grid,
            # mode="bilinear",
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).view(b, c, d, h, w)

        # print("warped_feature 1 : ", warped_feature.view(b, c, d, h, w)[0, :, 10, 10, :])
        # print("warped_feature 2 : ", warped_feature.view(b, c, d, h, w)[0, :, 30, 30, :])
        # print("warped_feature 3 : ", warped_feature.view(b, c, d, h, w)[0, :, 50, 50, :])

    return warped_feature
















def warp_kv_with_pose_depth_candidates(
        feature_k,
        feature_v,
        intrinsics,
        pose,
        depth,
        clamp_min_depth=1e-3,
        grid_sample_disable_cudnn=False,
):
    """
    feature_k: [B, C, H, W]
    feature_v: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature_k.size(1)

    with torch.autocast("cuda", enabled=False), torch.no_grad():
        # pixel coordinates (显式 FP32)
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        ).float()  # [B, 3, H, W]

        # —— 用 pinv 更稳，处理极小/近奇异内参；统一到 FP32 ——
        K = intrinsics.float().contiguous()
        Kinv = torch.linalg.pinv(K)  # [B,3,3]

        # back project to 3D and transform viewpoint
        points = Kinv.bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        R = pose[:, :3, :3].float()
        t = pose[:, :3, -1:].float()

        points = torch.bmm(R, points).unsqueeze(2).repeat(1, 1, d, 1) * depth.float().view(b, 1, d, h * w)  # [B,3,D,H*W]
        points = points + t.unsqueeze(-1)  # [B,3,D,H*W]

        # reproject to 2D image plane
        cam = K.bmm(points.view(b, 3, -1)).view(b, 3, d, h * w)  # [B,3,D,H*W]
        z = cam[:, -1:]  # [B,1,D,H*W]
        denom = z.clamp(min=clamp_min_depth)  # 仅为避免除零；不改原逻辑

        pixel_coords = cam[:, :2] / denom  # [B,2,D,H*W]
        # —— 关键：清理 inf/NaN，避免传入 grid_sample ——
        pixel_coords = torch.nan_to_num(pixel_coords, nan=0.0, posinf=0.0, neginf=0.0)

        # print("pixel_coords: ", pixel_coords.min(), pixel_coords.max())

        # normalize to [-1, 1]
        x_grid = 2.0 * pixel_coords[:, 0] / max(w - 1, 1) - 1.0
        y_grid = 2.0 * pixel_coords[:, 1] / max(h - 1, 1) - 1.0
        # print("x_grid: ", x_grid.min(), x_grid.max())
        # print("y_grid: ", y_grid.min(), y_grid.max())

        # 再次保底，确保 grid 完全有限
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=0.0, neginf=0.0)
        y_grid = torch.nan_to_num(y_grid, nan=0.0, posinf=0.0, neginf=0.0)

        grid = torch.stack([x_grid, y_grid], dim=-1).contiguous()  # [B, D, H*W, 2] (float32)

        # print("grid 1 : ", grid.view(b, d, h, w, 2)[0, :, 10, 10, :])
        # print("grid 2 : ", grid.view(b, d, h, w, 2)[0, :, 30, 30, :])
        # print("grid 3 : ", grid.view(b, d, h, w, 2)[0, :, 50, 50, :])


    # 采样索引贴图 - 不使用循环
    # 将网格调整为 [B, D*H, W, 2] 格式
    grid_flat = grid.view(b, d * h, w, 2)
    # print("feature_k.shape: ", feature_k.shape)
    # print("grid_flat.shape: ", grid_flat.shape)
    # print("grid_flat[0, 9:10, :, :].shape: ", grid_flat[0, 9:10, :, :].shape)
    # print("grid_flat[0, 99:100, :, :].shape: ", grid_flat[0, 99:100, :, :].shape)
    # print("grid_flat[0, 500:501, :, :].shape: ", grid_flat[0, 500:501, :, :].shape)

    # sample features（保持原逻辑）
    if feature_k.numel() > 1000000:
        grid_sample_disable_cudnn = True

    with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
        warped_feature_k = F.grid_sample(
            feature_k,  # b, c, h, w
            grid_flat,
            mode="bilinear",
            # mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).view(b, c, d, h, w)

        warped_feature_v = F.grid_sample(
            feature_v,  # b, c, h, w
            grid_flat,
            mode="bilinear",
            # mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).view(b, c, d, h, w)

    return warped_feature_k, warped_feature_v



















@torch.no_grad()
def warp_indices_with_pose_depth(
        intrinsics,  # [B,3,3]
        pose,  # [B,4,4] (T_{source <- target})
        depth,  # [B,D,H,W]
        clamp_min_depth=1e-3,
        grid_sample_disable_cudnn=False,
):
    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()

    with torch.autocast("cuda", enabled=False), torch.no_grad():
        # 生成齐次像素坐标网格 [B,3,H,W]
        grid = coords_grid(b, h, w, homogeneous=True, device=depth.device).float()

        # 计算内参矩阵的伪逆 [B,3,3]
        K = intrinsics.float().contiguous()
        Kinv = torch.linalg.pinv(K)

        # 反投影到3D空间 [B,3,H*W]
        points = Kinv.bmm(grid.view(b, 3, -1))

        # 提取旋转和平移 [B,3,3], [B,3,1]
        R = pose[:, :3, :3].float()
        t = pose[:, :3, -1:].float()

        # 应用旋转并缩放深度 [B,3,D,H*W]
        points = torch.bmm(R, points).unsqueeze(2).repeat(1, 1, d, 1) * depth.float().view(b, 1, d, h * w)
        points = points + t.unsqueeze(-1)  # 应用平移

        # 重投影到2D图像平面 [B,3,D,H*W]
        cam = K.bmm(points.view(b, 3, -1)).view(b, 3, d, h * w)
        z = cam[:, -1:]  # 深度值 [B,1,D,H*W]
        denom = z.clamp(min=clamp_min_depth)  # 避免除零

        # 计算像素坐标并清理非法值 [B,2,D,H*W]
        pixel_coords = cam[:, :2] / denom
        pixel_coords = torch.nan_to_num(pixel_coords, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化到[-1, 1]范围
        x_grid = 2.0 * pixel_coords[:, 0] / max(w - 1, 1) - 1.0
        y_grid = 2.0 * pixel_coords[:, 1] / max(h - 1, 1) - 1.0

        # 再次清理非法值
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=0.0, neginf=-0.0)
        y_grid = torch.nan_to_num(y_grid, nan=0.0, posinf=0.0, neginf=-0.0)

        # 组合成网格坐标 [B,D,H*W,2]
        grid = torch.stack([x_grid, y_grid], dim=-1).contiguous()

    # 创建索引贴图 (1~HW) [B,1,H,W]
    idx_img = torch.arange(1, h * w + 1, device=depth.device, dtype=torch.float32).view(1, 1, h, w)
    idx_img = idx_img.expand(b, 1, h, w)

    # 采样索引贴图 - 不使用循环
    # 将网格调整为 [B, D*H, W, 2] 格式
    grid_flat = grid.view(b, d * h, w, 2)
    
    # grid_log = grid.view(b, d, h, w, 2)
    # print("feature_k.shape: ", idx_img.shape)
    # print("grid_flat.shape: ", grid_log.shape)
    # # print("idx_img: ", idx_img)
    # print("idx_img[0, :, 9:10, 9:10]: ", idx_img[0, :, 9:10, 9:10].int().reshape(-1))
    # print("idx_img[0, :, 19:20, 19:20]: ", idx_img[0, :, 19:20, 19:20].int().reshape(-1))
    # print("idx_img[0, :, 29:30, 29:30]: ", idx_img[0, :, 29:30, 29:30].int().reshape(-1))
    # print("grid_flat[0, :, 9:10, 9:10, :]: ", grid_log[0, :, 9:10, 9:10, :].reshape(-1, 2))
    # print("grid_flat[0, :, 19:20, 19:20, :]: ", grid_log[0, :, 19:20, 19:20, :].reshape(-1, 2))
    # print("grid_flat[0, :, 29:30, 29:30, :]: ", grid_log[0, :, 29:30, 29:30, :].reshape(-1, 2))


    if idx_img.numel() > 1000000:
        grid_sample_disable_cudnn = True

    with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
        # 使用 grid_sample 采样索引
        idx_plus1 = F.grid_sample(
            idx_img,
            grid_flat,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True
        )
    
    # 调整形状 [B,1,D*H,W] -> [B,D,H,W]
    idx_plus1 = idx_plus1.view(b, 1, d, h, w).squeeze(1)

    # print(idx_plus1.shape)
    # print("idx_plus1[0, :, 9:10, 9:10]: ", idx_plus1[0, :, 9:10, 9:10].reshape(-1))
    # print("idx_plus1[0, :, 19:20, 19:20]: ", idx_plus1[0, :, 19:20, 19:20].reshape(-1))
    # print("idx_plus1[0, :, 29:30, 29:30]: ", idx_plus1[0, :, 29:30, 29:30].reshape(-1))




    # 转换为索引 (0=无效, 1~HW=有效索引)
    idx = idx_plus1.round().to(torch.long) - 1  # [B,D,H,W]

    # 通过归一化坐标判断有效性 (在[-1,1]范围内)
    valid_x = (x_grid >= -1.0) & (x_grid <= 1.0)
    valid_y = (y_grid >= -1.0) & (y_grid <= 1.0)
    valid_z = (z > clamp_min_depth).squeeze(1)  # [B,D,H*W]

    # 组合有效性掩码 [B,D,H*W] -> [B,D,H,W]
    valid = valid_z & valid_x & valid_y
    valid = valid.view(b, d, h, w)

    ###################################################
    # 将无效位置的索引设为-1
    idx = torch.where(valid, idx, torch.full_like(idx, -1))
    ###################################################

    # print("valid[0, :, 9:10, 9:10]: ", valid[0, :, 9:10, 9:10].reshape(-1))
    # print("valid[0, :, 19:20, 19:20]: ", valid[0, :, 19:20, 19:20].reshape(-1))
    # print("valid[0, :, 29:30, 29:30]: ", valid[0, :, 29:30, 29:30].reshape(-1))


    # 调整维度顺序 [B,H,W,D]
    idx = idx.permute(0, 2, 3, 1).contiguous()
    valid = valid.permute(0, 2, 3, 1).contiguous()
    # print(idx.min(), idx.max())

    # print(idx.shape)
    # print("idx[0, 9:10, 9:10, :]: ", idx[0, 9:10, 9:10, :].reshape(-1))
    # print("idx[0, 19:20, 19:20, :]: ", idx[0, 19:20, 19:20, :].reshape(-1))
    # print("idx[0, 29:30, 29:30, :]: ", idx[0, 29:30, 29:30, :].reshape(-1))
    # print("\n\n\n\n")
    
    return idx, valid
















@torch.no_grad()
def warp_indices_with_pose_depth_candidates(
        intrinsics,  # [B,3,3]
        pose,  # [B,4,4]  (T_{source <- target})
        depth,  # [B,D,H,W]
        clamp_min_depth=1e-3,
        grid_sample_disable_cudnn=False,
):
    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()

    with torch.autocast("cuda", enabled=False):
        # 1) 像素齐次坐标 [B,3,H,W] -> FP32
        grid = coords_grid(b, h, w, homogeneous=True, device=depth.device).float()

        # 2) K 与 K^{-1}
        K = intrinsics.float().contiguous()
        Kinv = torch.linalg.pinv(K)  # 更稳

        # 3) 反投影 + 位姿变换
        pts = Kinv.bmm(grid.view(b, 3, -1))  # [B,3,H*W]
        R = pose[:, :3, :3].float()
        t = pose[:, :3, -1:].float()

        pts = torch.bmm(R, pts).unsqueeze(2).repeat(1, 1, d, 1)  # [B,3,D,H*W]
        pts = pts * depth.float().view(b, 1, d, h * w)  # 乘深度
        pts = pts + t.unsqueeze(-1)  # 加平移

        # 4) 重新投影到像平面
        cam = K.bmm(pts.view(b, 3, -1)).view(b, 3, d, h * w)  # [B,3,D,H*W]
        z = cam[:, 2:3, ...]  # [B,1,D,H*W]
        denom = z.clamp(min=clamp_min_depth)

        # 像素坐标（未归一化，单位：像素）
        pix = cam[:, :2, ...] / denom  # [B,2,D,H*W]
        pix = torch.nan_to_num(pix, nan=0.0, posinf=0.0, neginf=0.0)

        u = pix[:, 0, ...]  # x / 列索引，范围约 [0, w-1]
        v = pix[:, 1, ...]  # y / 行索引，范围约 [0, h-1]

        # 5) 归一化到 [-1,1]（给 grid_sample 用）
        x_grid = 2.0 * u / max(w - 1, 1) - 1.0
        y_grid = 2.0 * v / max(h - 1, 1) - 1.0
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=0.0, neginf=0.0)
        y_grid = torch.nan_to_num(y_grid, nan=0.0, posinf=0.0, neginf=0.0)

        grid_norm = torch.stack([x_grid, y_grid], dim=-1).contiguous()  # [B, D, H*W, 2]
        # 关键：reshape 成 grid_sample 期望的 [B, H_out, W_out, 2]
        grid_norm = grid_norm.view(b, d * h, w, 2)

        # 6) 构造“索引贴图”（1..HW；0 作为越界 padding）
        idx_img = torch.arange(h * w, device=depth.device, dtype=torch.float32).view(1, 1, h, w)  # [1,1,H,W] in 1..HW
        idx_img = idx_img.expand(b, 1, -1, -1)  # [B,1,H,W]   [0 .. HW-1]

        # 7) 最近邻采样索引贴图 -> [B,1,D*h,w] -> [B,D,H,W]
        with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
            idx_plus1 = F.grid_sample(
                idx_img, grid_norm, mode="nearest", padding_mode="zeros", align_corners=True
            ).view(b, 1, d, h, w)
        idx = idx_plus1.round().to(torch.long)  # - 1                         # [0 .. HW-1]
        idx = idx.view(b, d, h, w)

        # 8) 物理有效性：z>0 且坐标在图内
        valid = (z.view(b, 1, d, h * w) > clamp_min_depth).squeeze(1)
        in_w = (u >= 0.0) & (u <= (w - 1))
        in_h = (v >= 0.0) & (v <= (h - 1))
        valid = (valid & in_w & in_h).view(b, d, h, w)

        # 越界或无效处统一置为 -1
        # idx = torch.where(valid, idx, torch.full_like(idx, -1))

        # 9) 输出维度 [B,H,W,D] + 有效性 [B,H,W,D]
        idx = idx.permute(0, 2, 3, 1).contiguous()
        valid = valid.permute(0, 2, 3, 1).contiguous()

    return idx, valid




