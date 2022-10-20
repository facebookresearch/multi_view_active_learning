# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random

import kornia
import numpy as np
import torch
from fblearner.flow.projects.nimble.multi_view_active_learning.pose_estimators.loss import (
    Pose2DMeanSquaredError,
)
from PIL import Image
from scipy.optimize import least_squares

from .evaluation import get_scaled_pred_corrdinates


IMAGENET_MEAN, IMAGENET_STD = (
    np.array([0.485, 0.456, 0.406]),
    np.array([0.229, 0.224, 0.225]),
)


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        # Rotate first then translate
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_width, new_height):
        height, width = image_shape
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)
        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = (
            new_fx,
            new_fy,
            new_cx,
            new_cy,
        )

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def crop_image(image, bbox):
    """
    Crops area from image specified as bbox. Always returns area
    of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image

    """

    image_pil = Image.fromarray(image)
    image_pil = image_pil.crop(bbox)

    return np.asarray(image_pil)


def get_square_bbox(bbox):
    """Makes square bbox from any bbox by stretching of minimal length side

    Args:
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower


def scale_bbox(bbox, scale):
    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def normalize_image(image):
    """Normalizes image using ImageNet mean and std

    Args:
        image numpy array of shape (h, w, 3): image

    Returns normalized_image numpy array of shape (h, w, 3): normalized image
    """
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_image(image):
    """Reverse to normalize_image() function"""
    return np.uint8(np.clip(255.0 * (image * IMAGENET_STD + IMAGENET_MEAN), 0, 255))


def project_3d_points_with_camera(retval_camera, points_3d):
    if retval_camera.dist is not None:
        return _project_3d_points_to_image_plane_with_distortion(
            points_3d,
            retval_camera.K,
            retval_camera.R,
            retval_camera.t,
            retval_camera.dist,
        )[:, :2]
    else:
        return _project_3d_points_to_image_plane_without_distortion(
            retval_camera.projection, points_3d
        )


def triangulation(
    heatmaps,
    proj_matricies,
    stride,
    valid_joints,
    use_soft_argmax=False,
    use_reprojection_xe=False,
    sigma=None,
    n_iters=64,
    reprojection_error_epsilon=5,
    direct_optimization=False,
):
    """
    heatmaps: shape should be [# of views, # of joints, w, h].
    proj_matricies: [# of views, 4].
    stride: overall stride of the pose estimation model
    valid_joints: boolean tensor, shape should be [# of joints].
    n_iters: Default to 64. Number of pairs of views to triangulate.
        With 8 views, we should be getting all pairs.
    direct_optimization: solving least squared to directly minimize
        the reprojection error.
    """
    n_joints = heatmaps.shape[1]
    if use_soft_argmax:
        keypoints_2d = (
            (
                kornia.spatial_soft_argmax2d(heatmaps, normalized_coordinates=False)
                * stride
            )
            .cpu()
            .numpy()
        )
    else:
        keypoints_2d = get_scaled_pred_corrdinates(
            heatmaps, stride, n_joints, valid_joints
        )
    proj_matricies_np = proj_matricies.detach().cpu().numpy()
    keypoints_3d = np.zeros((n_joints, 3))
    reprojection_error = []
    inlier_counts = []
    for joint_i in range(n_joints):
        if not valid_joints[joint_i]:
            continue
        points = keypoints_2d[:, joint_i]
        keypoint_3d, reprojection_error_mean, inlier_count = _triangulate_ransac(
            proj_matricies_np,
            points,
            n_iters,
            reprojection_error_epsilon,
            direct_optimization,
        )
        inlier_counts.append(inlier_count)
        keypoints_3d[joint_i] = keypoint_3d
        reprojection_error.append(reprojection_error_mean)
    if use_reprojection_xe:
        metric = _compute_xe(keypoints_3d, proj_matricies_np, heatmaps, sigma)
    else:
        metric = np.mean(reprojection_error)
    results = {
        "keypoints_3d": keypoints_3d,
        "keypoints_2d": keypoints_2d,
        "metric": metric,
        "inlier_count": np.min(inlier_counts),
    }
    return results


def _compute_xe(keypoints_3d, proj_matricies, pred_heatmaps, sigma):
    loss = Pose2DMeanSquaredError()
    batch_size, num_kps, h, w = pred_heatmaps.shape
    mse_error = 0.0
    for view_id, proj_matrix in enumerate(proj_matricies):
        kp_2d = _project_3d_points_to_image_plane_without_distortion(
            proj_matrix, keypoints_3d
        )
        for kp_id, kp in enumerate(kp_2d):
            kp = torch.from_numpy(kp)
            grid = torch.zeros(size=(h, w, 2))
            grid[..., 0] = torch.from_numpy(np.arange(w)).unsqueeze(0)
            grid[..., 1] = torch.from_numpy(np.arange(h)).unsqueeze(1)
            grid = grid.unsqueeze(0)
            labels = kp.unsqueeze(-2).unsqueeze(-2)
            exponent = torch.sum((grid - labels) ** 2, dim=-1)
            reprojected_heatmap = torch.exp(-exponent / (2.0 * (sigma**2)))
            mse_error += loss.pose_2d_mse(
                pred_heatmaps[view_id, kp_id : kp_id + 1], reprojected_heatmap.cuda()
            )
    return mse_error


def _triangulate_ransac(
    proj_matricies,
    points,
    n_iters,
    reprojection_error_epsilon,
    direct_optimization,
):
    assert len(proj_matricies) == len(points)
    assert len(points) >= 2

    proj_matricies = np.array(proj_matricies)
    points = np.array(points)

    n_views = len(points)

    # determine inliers
    view_set = set(range(n_views))
    inlier_set = set()

    view_paris = list(itertools.combinations(view_set, 2))
    if len(view_paris) > n_iters:
        random.shuffle(view_paris)
        view_paris = view_paris[:n_iters]

    for sampled_views in view_paris:
        sampled_views = list(sampled_views)
        keypoint_3d_in_base_camera = _triangulate_dlt(
            proj_matricies[sampled_views], points[sampled_views]
        )
        reprojection_error_vector = _calc_reprojection_error_matrix(
            np.array([keypoint_3d_in_base_camera]), points, proj_matricies
        )[0]

        new_inlier_set = set(sampled_views)
        for view in view_set:
            current_reprojection_error = reprojection_error_vector[view]
            if current_reprojection_error < reprojection_error_epsilon:
                new_inlier_set.add(view)

        if len(new_inlier_set) > len(inlier_set):
            inlier_set = new_inlier_set

    # triangulate using inlier_set
    if len(inlier_set) == 0:
        inlier_set = view_set.copy()

    inlier_list = np.array(sorted(inlier_set))
    inlier_proj_matricies = proj_matricies[inlier_list]
    inlier_points = points[inlier_list]

    keypoint_3d_in_base_camera = _triangulate_dlt(inlier_proj_matricies, inlier_points)
    reprojection_error_vector = _calc_reprojection_error_matrix(
        np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies
    )[0]
    reprojection_error_mean = np.mean(reprojection_error_vector)

    # direct reprojection error minimization
    if direct_optimization:

        def residual_function(x):
            reprojection_error_vector = _calc_reprojection_error_matrix(
                np.array([x]), inlier_points, inlier_proj_matricies
            )[0]
            residuals = reprojection_error_vector
            return residuals

        x_0 = np.array(keypoint_3d_in_base_camera)
        res = least_squares(residual_function, x_0, loss="huber", method="trf")

        keypoint_3d_in_base_camera = res.x
        reprojection_error_vector = _calc_reprojection_error_matrix(
            np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies
        )[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

    return keypoint_3d_in_base_camera, reprojection_error_mean, len(inlier_set)


def _triangulate_dlt(proj_matricies, points):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).

    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates

    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh = np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = _homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def _calc_reprojection_error_matrix(keypoints_3d, keypoints_2d_list, proj_matricies):
    reprojection_error_matrix = []
    for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
        keypoints_2d_projected = _project_3d_points_to_image_plane_without_distortion(
            proj_matrix, keypoints_3d
        )
        reprojection_error = (
            1
            / 2
            * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
        )
        reprojection_error_matrix.append(reprojection_error)

    return np.vstack(reprojection_error_matrix).T


def _homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean
    z = 0 would cause division by zero exceptions. Likely due to invalid 3d keypoints.
    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        z = points.T[-1]
        z = np.where(z == 0, np.ones_like(z), z)
        return (points.T[:-1] / z).T
    elif torch.is_tensor(points):
        z = points.transpose(1, 0)[-1]
        z = torch.where(z == 0, torch.ones_like(z), z)
        return (points.transpose(1, 0)[:-1] / z).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def _euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat(
            [
                points,
                torch.ones(
                    (points.shape[0], 1), dtype=points.dtype, device=points.device
                ),
            ],
            dim=1,
        )
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def _project_3d_points_to_image_plane_with_distortion(X, K, R, t, Kd):
    """Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See cv2.projectPoints
    """
    x = np.asarray(R.dot(X.T) + t)
    x[0:2, :] = x[0:2, :] / x[2, :]
    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]
    x[0, :] = (
        x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
        + 2 * Kd[2] * x[0, :] * x[1, :]
        + Kd[3] * (r + 2 * x[0, :] * x[0, :])
    )
    x[1, :] = (
        x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
        + 2 * Kd[3] * x[0, :] * x[1, :]
        + Kd[2] * (r + 2 * x[1, :] * x[1, :])
    )
    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]
    return x.T


def _project_3d_points_to_image_plane_without_distortion(
    proj_matrix, points_3d, convert_back_to_euclidean=True
):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be
            converted to euclidean coordinates
        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    # NOTE: @ is matrix multiply
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = _euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = _homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = _euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = _homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")
