//! Safe Rust bindings to nvblox Mapper via a thin C API.
//!
//! Parse-don't-validate: all types enforce invariants at construction.
//! The opaque `Mapper` handle is non-null and non-Send (GPU state).

use std::ptr::NonNull;

// Raw C FFI bindings
unsafe extern "C" {
    fn nvblox_mapper_create(voxel_size_m: f32) -> *mut std::ffi::c_void;
    fn nvblox_mapper_destroy(mapper: *mut std::ffi::c_void);
    fn nvblox_mapper_set_camera(
        mapper: *mut std::ffi::c_void,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        width: i32,
        height: i32,
    );
    fn nvblox_mapper_integrate_depth(
        mapper: *mut std::ffi::c_void,
        depth_data: *const f32,
        width: i32,
        height: i32,
        t_l_c_col_major_4x4: *const f32,
    );
    fn nvblox_mapper_extract_surface(mapper: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn nvblox_surface_num_vertices(surface: *const std::ffi::c_void) -> usize;
    fn nvblox_surface_positions(surface: *const std::ffi::c_void) -> *const f32;
    fn nvblox_surface_colors(surface: *const std::ffi::c_void) -> *const u8;
    fn nvblox_surface_destroy(surface: *mut std::ffi::c_void);
}

/// A validated voxel size. Must be positive and finite.
#[derive(Clone, Copy, Debug)]
pub struct VoxelSize(f32);

impl VoxelSize {
    pub fn new(meters: f32) -> Option<Self> {
        if meters > 0.0 && meters.is_finite() {
            Some(Self(meters))
        } else {
            None
        }
    }
    pub fn meters(self) -> f32 {
        self.0
    }
}

/// Camera intrinsics — validated at construction.
#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl CameraIntrinsics {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Option<Self> {
        if fx > 0.0
            && fy > 0.0
            && width > 0
            && height > 0
            && fx.is_finite()
            && fy.is_finite()
            && cx.is_finite()
            && cy.is_finite()
        {
            Some(Self {
                fx,
                fy,
                cx,
                cy,
                width,
                height,
            })
        } else {
            None
        }
    }
}

/// A 4×4 column-major transform (layer-from-camera).
#[derive(Clone, Copy, Debug)]
pub struct Transform([f32; 16]);

impl Transform {
    /// Create from a 3×3 rotation (row-major) and 3D translation.
    pub fn from_rt(rotation: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        // Convert to column-major 4×4
        let mut m = [0.0_f32; 16];
        for col in 0..3 {
            for row in 0..3 {
                m[col * 4 + row] = rotation[row][col];
            }
        }
        m[12] = translation[0];
        m[13] = translation[1];
        m[14] = translation[2];
        m[15] = 1.0;
        Self(m)
    }

    pub fn as_col_major(&self) -> &[f32; 16] {
        &self.0
    }
}

/// Depth image — validated dimensions. Owns the data.
pub struct DepthImage {
    data: Vec<f32>,
    width: u32,
    height: u32,
}

impl DepthImage {
    /// Create from raw float data. Length must equal width × height.
    pub fn new(data: Vec<f32>, width: u32, height: u32) -> Option<Self> {
        if data.len() == (width as usize) * (height as usize) && width > 0 && height > 0 {
            Some(Self {
                data,
                width,
                height,
            })
        } else {
            None
        }
    }
}

/// Safe wrapper around the nvblox Mapper.
/// Not Send — contains GPU state that must stay on the creating thread.
pub struct Mapper {
    ptr: NonNull<std::ffi::c_void>,
    _not_send: std::marker::PhantomData<*mut ()>,
}

// GPU state is thread-local — use PhantomData<*mut ()> to prevent Send/Sync
// (raw pointers are !Send + !Sync)

impl Mapper {
    /// Create a new Mapper with the given voxel size.
    pub fn new(voxel_size: VoxelSize) -> Option<Self> {
        let ptr = unsafe { nvblox_mapper_create(voxel_size.meters()) };
        NonNull::new(ptr).map(|ptr| Self {
            ptr,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Set camera intrinsics. Must be called before `integrate_depth`.
    pub fn set_camera(&mut self, cam: &CameraIntrinsics) {
        unsafe {
            nvblox_mapper_set_camera(
                self.ptr.as_ptr(),
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
                cam.width as i32,
                cam.height as i32,
            );
        }
    }

    /// Integrate a depth frame into the TSDF.
    pub fn integrate_depth(&mut self, depth: &DepthImage, t_l_c: &Transform) {
        unsafe {
            nvblox_mapper_integrate_depth(
                self.ptr.as_ptr(),
                depth.data.as_ptr(),
                depth.width as i32,
                depth.height as i32,
                t_l_c.as_col_major().as_ptr(),
            );
        }
    }

    /// Extract the current surface mesh as a point cloud.
    pub fn extract_surface(&mut self) -> Surface {
        let ptr = unsafe { nvblox_mapper_extract_surface(self.ptr.as_ptr()) };
        Surface {
            ptr: NonNull::new(ptr).expect("nvblox_mapper_extract_surface returned null"),
        }
    }
}

impl Drop for Mapper {
    fn drop(&mut self) {
        unsafe {
            nvblox_mapper_destroy(self.ptr.as_ptr());
        }
    }
}

/// Extracted surface mesh vertices.
pub struct Surface {
    ptr: NonNull<std::ffi::c_void>,
}

impl Surface {
    /// Number of vertices in the surface.
    pub fn num_vertices(&self) -> usize {
        unsafe { nvblox_surface_num_vertices(self.ptr.as_ptr()) }
    }

    /// Vertex positions as a slice of [x, y, z] triples.
    pub fn positions(&self) -> &[[f32; 3]] {
        let n = self.num_vertices();
        if n == 0 {
            return &[];
        }
        let ptr = unsafe { nvblox_surface_positions(self.ptr.as_ptr()) };
        unsafe { std::slice::from_raw_parts(ptr as *const [f32; 3], n) }
    }

    /// Vertex colors as a slice of [r, g, b, a] quads. May be empty.
    pub fn colors(&self) -> &[[u8; 4]] {
        let n = self.num_vertices();
        if n == 0 {
            return &[];
        }
        let ptr = unsafe { nvblox_surface_colors(self.ptr.as_ptr()) };
        if ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(ptr as *const [u8; 4], n) }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            nvblox_surface_destroy(self.ptr.as_ptr());
        }
    }
}
