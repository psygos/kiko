use std::fmt;

const MAX_ID_BYTES: usize = 64;

/// Allocation-free, checked ASCII identity retained after boundary parsing.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedId {
    bytes: [u8; MAX_ID_BYTES],
    len: u8,
}

impl BoundedId {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, IdentifierError> {
        Self::parse_str(field, &value)
    }

    pub(crate) fn parse_str(field: &'static str, value: &str) -> Result<Self, IdentifierError> {
        if value.is_empty()
            || value.len() > MAX_ID_BYTES
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
            })
        {
            return Err(IdentifierError { field });
        }
        let mut bytes = [0; MAX_ID_BYTES];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        Ok(Self {
            bytes,
            len: u8::try_from(value.len()).expect("identifier length is bounded to 64 bytes"),
        })
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("BoundedId contains checked ASCII")
    }
}

impl fmt::Debug for BoundedId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("BoundedId")
            .field(&self.as_str())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentifierError {
    pub field: &'static str,
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid bounded identifier in {}", self.field)
    }
}

impl std::error::Error for IdentifierError {}
