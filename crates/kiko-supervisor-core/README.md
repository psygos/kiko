# Kiko supervisor core

This crate owns Kiko's transport-independent lifecycle and exclusive motion
authority state machine. It performs no I/O, allocates no memory, and is usable
in simulation, replay, and the Nano runtime.

Important invariants:

- every process starts unarmed;
- inventory readiness and an explicit arm request are separate facts;
- commissioning, manual driving, point navigation, and exploration are
  mutually exclusive;
- a fresh, exact STM32-applied zero is required before initial authority,
  handover, lease-expiry recovery, or disarm completion;
- equality at a lease deadline is expired;
- controller identity changes, clock regression, and faults latch the
  supervisor and require inventory again; and
- clearing a fault never resumes a previous authority.

`ConfirmedBaseZero` means that a typed V2 applied result reported zero PWM,
safe output state, and clear controller faults. It is evidence of controller
state, not independently observed wheel motion.
