MEMORY
{
  /* Sector 7 (0x08060000..0x0807ffff) is reserved for the provisioned,
     append-only boot-identity journal. */
  FLASH : ORIGIN = 0x08000000, LENGTH = 384K
  RAM : ORIGIN = 0x20000000, LENGTH = 128K
}
