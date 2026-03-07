# RedMulE FPGA Bring-up Notes  
**Date:** March 7 2026

---

## Progress Summary

### 1. `redmule_wrap` solved the `DW = -1` propagation issue

During synthesis, several modules were receiving invalid widths (`-1`), causing errors such as:

```
part-select [-1:0]
variable size too large
```

The issue was caused by incorrect propagation of the HCI interface size parameters.

Using the official **`redmule_wrap`** module fixed this by explicitly defining:

```systemverilog
localparam hci_size_parameter_t `HCI_SIZE_PARAM(tcdm)
```

This ensured that `DW`, `BW`, and other interface parameters were correctly propagated through the design hierarchy.

---

### 2. Incremental synthesis checkpoint issue

Vivado synthesis repeatedly failed due to stale incremental synthesis checkpoints created for a different FPGA part.

Example errors:

```
Incremental checkpoint part mismatch
GeneratedRun file for 'synth_1' not found
```

**Root cause**

A previously generated `.dcp` checkpoint was still referenced by the project.

**Fix**

- Removed stale synthesis artifacts
- Deleted `.runs` and `.cache` directories
- Removed leftover `.dcp` files
- Restarted synthesis

After cleanup, synthesis proceeded normally.

---

### 3. IO pin explosion (TCDM interface)

The `redmule_wrap` module exposes the **TCDM memory interface** externally.

The number of memory ports is determined by:

```
MP = DW / MemDw
```

With default RedMulE parameters this results in multiple parallel memory ports.

As a consequence, the design attempted to expose **~1182 IO pins**, while the target board (ZCU104) provides only ~300 usable pins.

This indicates that the TCDM interface is intended for **internal cluster memory**, not direct FPGA IO.

**Current workaround**

- Perform **Out-of-Context (OOC) synthesis**

**Planned solution**

- Move TCDM memory inside the FPGA
- Implement **BRAM-based TCDM banks**
- Add the host RISC-V and make it complete SoC (next steps)

---

### 4. Place and Route completed

After resolving synthesis issues, **place and route successfully completed**.

Hierarchy interpretation:

- **Pink** → RedMulE compute engine  
- **Blue** → memory subsystem  
- **Yellow** → control logic  

---

### Floorplan / Placement

![RedMulE standalone on ZCU104's FPGA device](assets/redmule_place.png)


---


## Next Steps

- Integrate **BRAM-backed TCDM memory**
- Reduce external IO usage by integrating the redmule_wrap with cv32x core to make it complete SoC
- Run a **DNN inference workload (genome basecalling)** on FPGA
