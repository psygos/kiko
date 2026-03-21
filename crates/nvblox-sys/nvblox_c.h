/// Minimal C API for nvblox Mapper.
/// This thin wrapper exposes just what kiko-slam needs:
/// create mapper, integrate depth, extract surface points.

#ifndef NVBLOX_C_H
#define NVBLOX_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an nvblox Mapper.
typedef struct NvbloxMapper NvbloxMapper;

/// Opaque handle to extracted surface points.
typedef struct NvbloxSurface NvbloxSurface;

/// Create a new Mapper with the given voxel size.
NvbloxMapper* nvblox_mapper_create(float voxel_size_m);

/// Destroy a Mapper.
void nvblox_mapper_destroy(NvbloxMapper* mapper);

/// Set the camera intrinsics. Must be called before integrateDepth.
void nvblox_mapper_set_camera(
    NvbloxMapper* mapper,
    float fx, float fy, float cx, float cy,
    int width, int height
);

/// Integrate a depth image (float32, row-major, meters).
/// T_L_C is a column-major 4x4 transform (layer-from-camera).
/// depth_data points to width*height floats on CPU.
void nvblox_mapper_integrate_depth(
    NvbloxMapper* mapper,
    const float* depth_data,
    int width, int height,
    const float* T_L_C_col_major_4x4
);

/// Extract surface mesh vertices as a point cloud.
/// Returns an opaque handle. Call nvblox_surface_* to read data.
NvbloxSurface* nvblox_mapper_extract_surface(NvbloxMapper* mapper);

/// Get the number of surface vertices.
size_t nvblox_surface_num_vertices(const NvbloxSurface* surface);

/// Get pointer to vertex positions (3 floats per vertex, x/y/z).
const float* nvblox_surface_positions(const NvbloxSurface* surface);

/// Get pointer to vertex colors (4 bytes per vertex, RGBA).
/// Returns NULL if no colors available.
const uint8_t* nvblox_surface_colors(const NvbloxSurface* surface);

/// Destroy a surface extraction result.
void nvblox_surface_destroy(NvbloxSurface* surface);

#ifdef __cplusplus
}
#endif

#endif // NVBLOX_C_H
