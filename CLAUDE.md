# CLAUDE.md

## Data schema — what a `cell_id` actually is

A `cell_id` (e.g. `UKVCHE102`, `BKVPAN1`) is not an opaque location token — it encodes three distinct pieces of information:

1. **1st letter — radio technology of the cell**
   - `B` or `D` → 2G
   - `U` or `V` → 3G

2. **Middle letters — root ID of the base station (the physical antenna/site)**
   - This is the actual physical location. The same physical site can appear under multiple `cell_id` values if it has several technology generations and/or sectors.

3. **Trailing digits — zone/sector of that base station**
   - 2G: `1`, `2`, `3`, or `4`
   - 3G: `101`, `102`, `103`, `104`, `201`, `202`, `203`, or `204`

So `cell_id = <tech letter><site root><zone digits>`. Two `cell_id` strings can share the same site root while differing only in technology letter and/or zone digits — that represents the same physical location, observed through a different radio cell.
