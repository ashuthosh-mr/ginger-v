🗓️ Feb 4 — RedMulE FPGA Synthesis Notes (Vivado)

1. While bringing up RedMulE for FPGA synthesis in Vivado (outside a full PULP system), I ran into a few integration-related issues worth noting:

2. Unsigned sentinel parameters (-1)
Several HWPE/HCI parameters historically default to -1 (meaning “must be overridden”).
In Vivado, int unsigned = -1 becomes 2^32−1, leading to absurd vector widths and synthesis failures.
Fix: ensure all such parameters are explicitly driven to concrete values.

3. Include-path / macro resolution
Critical parameters (e.g. DW, DATAW_ALIGN) are derived via macros in .svh files like hci_helpers.svh.
Missing include paths caused these macros to silently fail, collapsing parameters to -1 or 0.
Fix: properly set include directories (or temporarily use absolute includes).

4. Simulation vs synthesis mismatch
Some parameters looked correct in simulation (VCD) but failed in synthesis, since Vivado elaborates the full hierarchy.

Takeaway: RedMulE is synthesizable on FPGA, but requires explicit system-level parameterization and careful handling of sentinel values when used outside its usual auto-generated SoC context.
