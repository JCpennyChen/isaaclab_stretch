import h5py

f = h5py.File(
    "/mnt/shared_ssd/JohnChen/stretch/datasets/stretch_cabinet_demo_reach.hdf5", "r"
)

# Check a single demo
demo = f["data/demo_0"]
obs = demo["obs/policy"][:]
actions = demo["actions"][:]

print(f"Obs shape: {obs.shape}, min: {obs.min():.4f}, max: {obs.max():.4f}")
print(
    f"Actions shape: {actions.shape}, min: {actions.min():.4f}, max: {actions.max():.4f}"
)
print(f"First obs: {obs[0]}")
print(f"First action: {actions[0]}")
print(f"Last action: {actions[-1]}")

f.close()
