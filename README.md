# Pac-man (2D)

> Pac-Man is a maze action game developed and released by Namco for arcades in 1980. The player controls Pac-Man, who must eat all the dots inside an enclosed maze while avoiding four colored ghosts. Eating large flashing dots called "Power Pellets" causes the ghosts to turn blue, allowing Pac-Man to eat them for bonus points.

- **Topics**: AI, Tilemaps, Sprites
- **Version**: Unity 2021.3 (LTS)
- **Credits**: https://github.com/adamgraham
- [**Download**](https://github.com/zigurous/unity-pacman-tutorial/archive/refs/heads/main.zip)
- [**Watch Video**](https://youtu.be/TKt_VlMn_aA)

## Training performance dashboard

The Python DQN trainer writes episode rewards and training loss to TensorBoard
event files under `runs/`. To view those metrics in a live dashboard, start it
from the repository root:

```bash
python python/dashboard.py
```

Then open http://127.0.0.1:8000. The dashboard compares available runs, plots
episode reward and training loss, and refreshes every 10 seconds. It checks
both `runs/` and `python/runs/` by default, and shows an empty state while
training logs have not been created yet. To use a different log directory or
port, pass `--logdir` or `--port`, for example:

```bash
python python/dashboard.py --logdir path/to/runs --port 8080
```
