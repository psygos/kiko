/// C wrapper implementation for nvblox Mapper.

#include "nvblox_c.h"
#include <nvblox/nvblox.h>
#include <vector>
#include <cstring>

struct NvbloxMapper {
    nvblox::Mapper mapper;
    std::optional<nvblox::Camera> camera;
    NvbloxMapper(float voxel_size)
        : mapper(voxel_size, nvblox::MemoryType::kUnified) {}
};

struct NvbloxSurface {
    std::vector<float> positions;
    std::vector<uint8_t> colors;
};

extern "C" {

NvbloxMapper* nvblox_mapper_create(float voxel_size_m) {
    return new NvbloxMapper(voxel_size_m);
}

void nvblox_mapper_destroy(NvbloxMapper* mapper) {
    delete mapper;
}

void nvblox_mapper_set_camera(
    NvbloxMapper* mapper,
    float fx, float fy, float cx, float cy,
    int width, int height
) {
    mapper->camera = nvblox::Camera(fx, fy, cx, cy, width, height);
}

void nvblox_mapper_integrate_depth(
    NvbloxMapper* mapper,
    const float* depth_data,
    int width, int height,
    const float* T_L_C_col_major_4x4
) {
    if (!mapper->camera.has_value()) return;

    nvblox::DepthImage depth_image(height, width, nvblox::MemoryType::kUnified);
    std::memcpy(depth_image.dataPtr(), depth_data,
                width * height * sizeof(float));

    Eigen::Matrix4f mat;
    std::memcpy(mat.data(), T_L_C_col_major_4x4, 16 * sizeof(float));
    nvblox::Transform T_L_C;
    T_L_C.matrix() = mat;

    mapper->mapper.integrateDepth(depth_image, T_L_C, mapper->camera.value());
}

NvbloxSurface* nvblox_mapper_extract_surface(NvbloxMapper* mapper) {
    auto* surface = new NvbloxSurface();

    auto block_indices = mapper->mapper.tsdf_layer().getAllBlockIndices();
    if (block_indices.empty()) {
        return surface;
    }

    mapper->mapper.updateColorMesh();
    const auto& mesh_layer = mapper->mapper.color_mesh_layer();

    auto block_idx_list = mesh_layer.getAllBlockIndices();
    for (const auto& idx : block_idx_list) {
        auto block_ptr = mesh_layer.getBlockAtIndex(idx);
        if (!block_ptr) continue;

        size_t n = block_ptr->vertices.size();
        if (n == 0) continue;

        // Copy vertices from unified/device memory to host
        std::vector<nvblox::Vector3f> verts_host(n);
        cudaMemcpy(verts_host.data(), block_ptr->vertices.data(),
                   n * sizeof(nvblox::Vector3f), cudaMemcpyDefault);

        for (size_t i = 0; i < n; i++) {
            surface->positions.push_back(verts_host[i].x());
            surface->positions.push_back(verts_host[i].y());
            surface->positions.push_back(verts_host[i].z());
        }

        // Extract colors from vertex_appearances (nvblox::Color type)
        if (block_ptr->vertex_appearances.size() == n) {
            std::vector<nvblox::Color> colors_host(n);
            cudaMemcpy(colors_host.data(), block_ptr->vertex_appearances.data(),
                       n * sizeof(nvblox::Color), cudaMemcpyDefault);
            for (size_t i = 0; i < n; i++) {
                surface->colors.push_back(colors_host[i].r());
                surface->colors.push_back(colors_host[i].g());
                surface->colors.push_back(colors_host[i].b());
                surface->colors.push_back(255);
            }
        }
    }

    return surface;
}

size_t nvblox_surface_num_vertices(const NvbloxSurface* surface) {
    return surface->positions.size() / 3;
}

const float* nvblox_surface_positions(const NvbloxSurface* surface) {
    return surface->positions.data();
}

const uint8_t* nvblox_surface_colors(const NvbloxSurface* surface) {
    if (surface->colors.empty()) return nullptr;
    return surface->colors.data();
}

void nvblox_surface_destroy(NvbloxSurface* surface) {
    delete surface;
}

} // extern "C"
