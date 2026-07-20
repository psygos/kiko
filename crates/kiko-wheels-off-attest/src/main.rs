//! Atomically consume the three root-owned wheels-off bench attestations.
//!
//! The pending directory is renamed before any item is inspected. Therefore a
//! successful or rejected transaction cannot be replayed. A process killed
//! after the rename leaves only the consuming directory, which the next
//! invocation removes before refusing to proceed without a new pending set.

#![cfg(unix)]
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::time::{Duration, SystemTime};

const PENDING_DIRECTORY: &str = "/run/kiko-wheels-off.pending";
const CONSUMING_DIRECTORY: &str = "/run/kiko-wheels-off.consuming";
const REQUIRED_FILES: [&str; 3] = ["head-path-clear", "power-cut-reachable", "wheels-removed"];
const ROOT_UID: u32 = 0;
const ROOT_GID: u32 = 0;
const DIRECTORY_MODE: u32 = 0o700;
const FILE_MODE: u32 = 0o400;
const MAX_ATTESTATION_AGE: Duration = Duration::from_secs(60);

fn main() {
    if let Err(error) = consume_attestations(
        Path::new(PENDING_DIRECTORY),
        Path::new(CONSUMING_DIRECTORY),
        ROOT_UID,
        ROOT_GID,
    ) {
        eprintln!("wheels-off attestation transaction refused: {error}");
        std::process::exit(1);
    }
    eprintln!("wheels-off attestations verified and consumed");
}

fn consume_attestations(
    pending: &Path,
    consuming: &Path,
    required_uid: u32,
    required_gid: u32,
) -> Result<(), String> {
    if let Err(source) = remove_path_if_present(consuming) {
        let _ = remove_path_if_present(pending);
        return Err(format!(
            "cannot remove an incomplete prior transaction at {}: {source}",
            consuming.display()
        ));
    }

    if let Err(source) = fs::rename(pending, consuming) {
        // A failed invocation never leaves a reusable pending attestation set.
        let pending_cleanup = remove_path_if_present(pending);
        let consuming_cleanup = remove_path_if_present(consuming);
        return Err(format!(
            "cannot atomically claim {} as {}: {source}; pending_cleanup={pending_cleanup:?}; consuming_cleanup={consuming_cleanup:?}",
            pending.display(),
            consuming.display()
        ));
    }

    let observed_at = SystemTime::now();
    let verification = verify_claimed_directory(consuming, required_uid, required_gid, observed_at);
    let cleanup = remove_path_if_present(consuming);
    match (verification, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(source)) => Err(format!(
            "verified transaction could not be consumed at {}: {source}",
            consuming.display()
        )),
        (Err(error), Err(source)) => Err(format!(
            "{error}; transaction cleanup at {} also failed: {source}",
            consuming.display()
        )),
    }
}

fn verify_claimed_directory(
    directory: &Path,
    required_uid: u32,
    required_gid: u32,
    observed_at: SystemTime,
) -> Result<(), String> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|source| format!("cannot inspect {}: {source}", directory.display()))?;
    if !metadata.file_type().is_dir() {
        return Err(format!(
            "claimed transaction {} is not a real directory",
            directory.display()
        ));
    }
    require_owner_mode(
        directory,
        &metadata,
        required_uid,
        required_gid,
        DIRECTORY_MODE,
    )?;
    require_fresh_timestamp(directory, &metadata, observed_at)?;

    let mut actual_names = BTreeSet::new();
    let entries = fs::read_dir(directory)
        .map_err(|source| format!("cannot enumerate {}: {source}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|source| {
            format!(
                "cannot read an attestation entry in {}: {source}",
                directory.display()
            )
        })?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| format!("{} contains a non-UTF-8 entry name", directory.display()))?;
        if !actual_names.insert(name.clone()) {
            return Err(format!("duplicate attestation entry name: {name}"));
        }
    }

    let required_names = REQUIRED_FILES
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if actual_names != required_names {
        return Err(format!(
            "attestation set mismatch: required={required_names:?}, actual={actual_names:?}"
        ));
    }

    for name in REQUIRED_FILES {
        let path = directory.join(name);
        verify_attestation_file(&path, required_uid, required_gid, observed_at)?;
    }
    Ok(())
}

fn verify_attestation_file(
    path: &Path,
    required_uid: u32,
    required_gid: u32,
    observed_at: SystemTime,
) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|source| format!("cannot inspect {}: {source}", path.display()))?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "attestation {} is not a regular non-symlink file",
            path.display()
        ));
    }
    require_owner_mode(path, &metadata, required_uid, required_gid, FILE_MODE)?;
    require_fresh_timestamp(path, &metadata, observed_at)?;
    if metadata.nlink() != 1 {
        return Err(format!(
            "attestation {} has {} hard links; exactly one is required",
            path.display(),
            metadata.nlink()
        ));
    }
    if metadata.len() != 0 {
        return Err(format!(
            "attestation {} contains {} bytes; an empty file is required",
            path.display(),
            metadata.len()
        ));
    }
    Ok(())
}

fn require_fresh_timestamp(
    path: &Path,
    metadata: &fs::Metadata,
    observed_at: SystemTime,
) -> Result<(), String> {
    let modified = metadata
        .modified()
        .map_err(|source| format!("cannot read {} modification time: {source}", path.display()))?;
    let age = observed_at.duration_since(modified).map_err(|source| {
        format!(
            "attestation {} has a future modification time: {source}",
            path.display()
        )
    })?;
    if age > MAX_ATTESTATION_AGE {
        return Err(format!(
            "attestation {} is {age:?} old; maximum age is {MAX_ATTESTATION_AGE:?}",
            path.display()
        ));
    }
    Ok(())
}

fn require_owner_mode(
    path: &Path,
    metadata: &fs::Metadata,
    required_uid: u32,
    required_gid: u32,
    required_mode: u32,
) -> Result<(), String> {
    let actual_mode = metadata.permissions().mode() & 0o7777;
    if metadata.uid() != required_uid
        || metadata.gid() != required_gid
        || actual_mode != required_mode
    {
        return Err(format!(
            "{} requires uid={required_uid}, gid={required_gid}, mode={required_mode:#o}; actual uid={}, gid={}, mode={actual_mode:#o}",
            path.display(),
            metadata.uid(),
            metadata.gid()
        ));
    }
    Ok(())
}

fn remove_path_if_present(path: &Path) -> io::Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(source),
    };
    if metadata.file_type().is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{File, FileTimes};
    use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new(name: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("test clock after epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "kiko-wheels-off-attest-{name}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create isolated test root");
            Self(path)
        }

        fn pending(&self) -> PathBuf {
            self.0.join("pending")
        }

        fn consuming(&self) -> PathBuf {
            self.0.join("consuming")
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn create_valid_pending(root: &TestRoot) -> (u32, u32) {
        let pending = root.pending();
        fs::create_dir(&pending).expect("create pending directory");
        fs::set_permissions(&pending, fs::Permissions::from_mode(DIRECTORY_MODE))
            .expect("set pending mode");
        for name in REQUIRED_FILES {
            let path = pending.join(name);
            fs::write(&path, []).expect("create empty attestation");
            fs::set_permissions(&path, fs::Permissions::from_mode(FILE_MODE))
                .expect("set attestation mode");
        }
        let metadata = fs::symlink_metadata(&pending).expect("pending metadata");
        (metadata.uid(), metadata.gid())
    }

    #[test]
    fn valid_set_is_atomically_claimed_and_consumed() {
        let root = TestRoot::new("valid");
        let (uid, gid) = create_valid_pending(&root);
        consume_attestations(&root.pending(), &root.consuming(), uid, gid)
            .expect("valid transaction");
        assert!(!root.pending().exists());
        assert!(!root.consuming().exists());
    }

    #[test]
    fn symlink_is_rejected_and_entire_claimed_set_is_consumed() {
        let root = TestRoot::new("symlink");
        let (uid, gid) = create_valid_pending(&root);
        let target = root.0.join("outside");
        fs::write(&target, []).expect("create symlink target");
        let replaced = root.pending().join("head-path-clear");
        fs::remove_file(&replaced).expect("remove regular attestation");
        symlink(&target, &replaced).expect("install rejected symlink");

        assert!(consume_attestations(&root.pending(), &root.consuming(), uid, gid).is_err());
        assert!(!root.pending().exists());
        assert!(!root.consuming().exists());
        assert!(target.exists(), "cleanup must not follow the symlink");
    }

    #[test]
    fn failed_claim_removes_a_reusable_pending_set() {
        let root = TestRoot::new("failed-claim");
        let (uid, gid) = create_valid_pending(&root);
        fs::create_dir(root.consuming()).expect("force rename conflict");
        fs::set_permissions(root.consuming(), fs::Permissions::from_mode(DIRECTORY_MODE))
            .expect("set consuming mode");

        // The helper first removes a stale consuming transaction and can then
        // claim this valid set. A second call has no reusable pending files.
        consume_attestations(&root.pending(), &root.consuming(), uid, gid)
            .expect("stale transaction is cleared before a new claim");
        assert!(consume_attestations(&root.pending(), &root.consuming(), uid, gid).is_err());
        assert!(!root.pending().exists());
        assert!(!root.consuming().exists());
    }

    #[test]
    fn extra_file_rejects_and_consumes_the_whole_transaction() {
        let root = TestRoot::new("extra");
        let (uid, gid) = create_valid_pending(&root);
        let extra = root.pending().join("unexpected");
        fs::write(&extra, []).expect("create extra entry");
        fs::set_permissions(&extra, fs::Permissions::from_mode(FILE_MODE)).expect("set extra mode");

        assert!(consume_attestations(&root.pending(), &root.consuming(), uid, gid).is_err());
        assert!(!root.pending().exists());
        assert!(!root.consuming().exists());
    }

    #[test]
    fn stale_file_rejects_and_consumes_the_whole_transaction() {
        let root = TestRoot::new("stale");
        let (uid, gid) = create_valid_pending(&root);
        let stale = SystemTime::now()
            .checked_sub(MAX_ATTESTATION_AGE + Duration::from_secs(1))
            .expect("test clock supports a stale instant");
        File::open(root.pending().join("wheels-removed"))
            .expect("open stale fixture")
            .set_times(FileTimes::new().set_modified(stale))
            .expect("set stale modification time");

        assert!(consume_attestations(&root.pending(), &root.consuming(), uid, gid).is_err());
        assert!(!root.pending().exists());
        assert!(!root.consuming().exists());
    }

    #[test]
    fn freshness_boundary_is_exact_and_future_times_fail_closed() {
        let root = TestRoot::new("freshness-boundary");
        let (uid, gid) = create_valid_pending(&root);
        let path = root.pending().join("wheels-removed");
        let modified = fs::metadata(&path)
            .expect("fresh fixture metadata")
            .modified()
            .expect("fresh fixture modification time");

        verify_attestation_file(&path, uid, gid, modified + MAX_ATTESTATION_AGE)
            .expect("exact maximum age is accepted");
        assert!(
            verify_attestation_file(
                &path,
                uid,
                gid,
                modified + MAX_ATTESTATION_AGE + Duration::from_nanos(1),
            )
            .is_err()
        );
        assert!(
            verify_attestation_file(&path, uid, gid, modified - Duration::from_nanos(1),).is_err()
        );
    }
}
