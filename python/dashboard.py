import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MAX_CHART_POINTS = 1000
EVENT_FILE_PREFIX = "events.out.tfevents"
HTML_PATH = Path(__file__).with_name("dashboard.html")


def discover_run_dirs(log_dir):
    if not log_dir.is_dir():
        return []

    if any(path.name.startswith(EVENT_FILE_PREFIX) for path in log_dir.iterdir()):
        return [log_dir]

    return sorted(
        (
            path
            for path in log_dir.iterdir()
            if path.is_dir()
            and any(
                child.name.startswith(EVENT_FILE_PREFIX)
                for child in path.iterdir()
            )
        ),
        key=lambda path: path.name.lower(),
    )


def default_log_dir():
    candidates = [Path("runs"), Path("python/runs")]
    return next(
        (
            path.resolve()
            for path in candidates
            if path.is_dir() and discover_run_dirs(path)
        ),
        Path("runs").resolve(),
    )


def downsample(points):
    if len(points) <= MAX_CHART_POINTS:
        return points

    last_index = len(points) - 1
    return [
        points[round(index * last_index / (MAX_CHART_POINTS - 1))]
        for index in range(MAX_CHART_POINTS)
    ]


def scalar_series(accumulator, tag):
    return [
        {"step": event.step, "value": event.value, "wall_time": event.wall_time}
        for event in accumulator.Scalars(tag)
        if math.isfinite(event.value)
    ]


def load_runs(log_dir):
    runs = []
    for run_dir in discover_run_dirs(log_dir):
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        rewards = scalar_series(accumulator, "episode/reward") if "episode/reward" in tags else []
        losses = scalar_series(accumulator, "train/loss") if "train/loss" in tags else []

        runs.append(
            {
                "name": run_dir.name,
                "episodes": len(rewards),
                "latest_reward": rewards[-1]["value"] if rewards else None,
                "latest_reward_time": rewards[-1]["wall_time"] if rewards else None,
                "best_reward": max((point["value"] for point in rewards), default=None),
                "recent_average_reward": (
                    sum(point["value"] for point in rewards[-20:]) / min(len(rewards), 20)
                    if rewards
                    else None
                ),
                "latest_step": max(
                    (points[-1]["step"] for points in (rewards, losses) if points),
                    default=0,
                ),
                "reward": downsample(rewards),
                "loss": downsample(losses),
            }
        )
    return runs


def dashboard_data(log_dir):
    runs = load_runs(log_dir)
    reward_runs = [run for run in runs if run["episodes"]]
    best_rewards = [run["best_reward"] for run in reward_runs]
    recent_averages = [run["recent_average_reward"] for run in reward_runs]
    latest_run = max(
        (run for run in reward_runs if run["latest_reward_time"] is not None),
        key=lambda run: run["latest_reward_time"],
        default=None,
    )

    return {
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "episode_count": sum(run["episodes"] for run in runs),
            "best_reward": max(best_rewards, default=None),
            "recent_average_reward": (
                sum(recent_averages) / len(recent_averages)
                if recent_averages
                else None
            ),
            "latest_reward": latest_run["latest_reward"] if latest_run else None,
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    log_dir = Path(".")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_file(HTML_PATH, "text/html; charset=utf-8")
        elif path == "/api/runs":
            self._send_runs()
        else:
            self.send_error(404, "Not found")

    def _send_file(self, file_path, content_type):
        try:
            content = file_path.read_bytes()
        except OSError as error:
            self.send_error(500, f"Could not read dashboard page: {error}")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_runs(self):
        try:
            content = json.dumps(dashboard_data(self.log_dir)).encode("utf-8")
        except FileNotFoundError as error:
            self.send_error(404, str(error))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        print(f"[dashboard] {self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(
        description="Show DQN training metrics from TensorBoard event files."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        help="TensorBoard log directory (default: find runs/ or python/runs/)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind to (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)"    )
    args = parser.parse_args()

    if args.logdir is None:
        DashboardHandler.log_dir = default_log_dir()
    else:
        DashboardHandler.log_dir = args.logdir.expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}  (logs: {DashboardHandler.log_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
