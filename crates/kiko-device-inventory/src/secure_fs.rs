use std::ffi::OsStr;
use std::fmt;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};

use rustix::fd::OwnedFd;
use rustix::fs::{Mode, OFlags, open, openat};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenedPathKind {
    Directory,
    File,
}

#[derive(Debug)]
pub struct SecureOpenError {
    component_index: usize,
    source: rustix::io::Errno,
}

impl SecureOpenError {
    pub const fn component_index(&self) -> usize {
        self.component_index
    }

    pub const fn errno(&self) -> rustix::io::Errno {
        self.source
    }
}

impl fmt::Display for SecureOpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "no-follow open failed at component {}: {}",
            self.component_index, self.source
        )
    }
}

impl std::error::Error for SecureOpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub(crate) fn open_absolute_nofollow(
    path: &Path,
    final_kind: OpenedPathKind,
) -> Result<OwnedFd, SecureOpenError> {
    let mut current =
        open("/", directory_flags(), Mode::empty()).map_err(|source| SecureOpenError {
            component_index: 0,
            source,
        })?;
    let mut components = path.components().filter_map(|component| match component {
        Component::Normal(name) => Some(name),
        _ => None,
    });
    let mut next = components.next();
    let mut normal_index = 0_usize;
    while let Some(name) = next {
        next = components.next();
        normal_index += 1;
        let kind = if next.is_some() {
            OpenedPathKind::Directory
        } else {
            final_kind
        };
        current = open_component(&current, name, kind).map_err(|source| SecureOpenError {
            component_index: normal_index,
            source,
        })?;
    }
    Ok(current)
}

pub(crate) fn is_canonical_absolute_path(path: &Path) -> bool {
    let bytes = path.as_os_str().as_bytes();
    if bytes == b"/" {
        return true;
    }
    bytes.first() == Some(&b'/')
        && bytes.last() != Some(&b'/')
        && bytes[1..]
            .split(|byte| *byte == b'/')
            .all(|component| !component.is_empty() && component != b"." && component != b"..")
        && !bytes.contains(&0)
}

pub(crate) fn open_beneath_nofollow(
    root: &OwnedFd,
    relative: &Path,
) -> Result<OwnedFd, SecureOpenError> {
    let mut current: Option<OwnedFd> = None;
    let mut components = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name),
            _ => None,
        });
    let mut next = components.next();
    debug_assert!(next.is_some());
    let mut normal_index = 0_usize;
    while let Some(name) = next {
        next = components.next();
        normal_index += 1;
        let kind = if next.is_some() {
            OpenedPathKind::Directory
        } else {
            OpenedPathKind::File
        };
        let opened = match current.as_ref() {
            Some(directory) => open_component(directory, name, kind),
            None => open_component(root, name, kind),
        }
        .map_err(|source| SecureOpenError {
            component_index: normal_index,
            source,
        })?;
        current = Some(opened);
    }
    Ok(current.expect("parsed relative artifact path has a normal component"))
}

fn open_component(
    directory: &OwnedFd,
    name: &OsStr,
    kind: OpenedPathKind,
) -> rustix::io::Result<OwnedFd> {
    let flags = match kind {
        OpenedPathKind::Directory => directory_flags(),
        OpenedPathKind::File => file_flags(),
    };
    openat(directory, name, flags, Mode::empty())
}

fn directory_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK
}

fn file_flags() -> OFlags {
    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK
}
