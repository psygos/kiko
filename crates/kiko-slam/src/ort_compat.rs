//! Compatibility stubs for the ORT 1.24.2 static archive (aarch64).
//!
//! The pre-built archive was compiled with GCC 14 / glibc 2.38 which emit calls to
//! symbols that don't exist in Ubuntu 22.04's GCC 11 / glibc 2.35:
//!
//! - `__cxa_call_terminate` (GCC 14 noexcept ABI change)
//! - `__isoc23_strto{l,ll,ull}` (C23 aliases added in glibc 2.38)
//!
//! These stubs provide link-time compatible definitions so the binary links on older
//! toolchains. The behavior is equivalent: `__cxa_call_terminate` aborts (it's only
//! called on noexcept violation), and the C23 strto* variants are identical to their
//! C11 counterparts at the ABI level.

#[allow(unsafe_code)]
unsafe extern "C" {
    fn abort() -> !;
    fn strtol(nptr: *const i8, endptr: *mut *mut i8, base: i32) -> isize;
    fn strtoll(nptr: *const i8, endptr: *mut *mut i8, base: i32) -> i64;
    fn strtoull(nptr: *const i8, endptr: *mut *mut i8, base: i32) -> u64;
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn __cxa_call_terminate(_exception: *mut core::ffi::c_void) -> ! {
    unsafe { abort() }
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn __isoc23_strtol(nptr: *const i8, endptr: *mut *mut i8, base: i32) -> isize {
    unsafe { strtol(nptr, endptr, base) }
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn __isoc23_strtoll(nptr: *const i8, endptr: *mut *mut i8, base: i32) -> i64 {
    unsafe { strtoll(nptr, endptr, base) }
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn __isoc23_strtoull(nptr: *const i8, endptr: *mut *mut i8, base: i32) -> u64 {
    unsafe { strtoull(nptr, endptr, base) }
}
