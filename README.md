# amphibious-quadruped-optimization

## Building and installing the SPlisHSPlasH Gazebo plugin

After making changes to the plugin source, rebuild and install it with:

```bash
cmake --build /home/ws/splishsplash/build --target FluidSimulator -- -j$(nproc)
cmake --install /home/ws/splishsplash/build --prefix /home/moritz/.local
```

The install step copies `libFluidSimulator.so` to `/home/moritz/.local/lib/gazebo-11/plugins/`, which is on `GAZEBO_PLUGIN_PATH` and picked up automatically by Gazebo.