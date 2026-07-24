"""nuPlan simulation entry point with an NFS-safe NuBoard writer."""

from __future__ import annotations

import pickle
from collections import defaultdict

import pandas as pd

from nuplan.common.utils import io_utils
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback


async def _save_local_buffer(output_path, buf: bytes) -> None:
    """Synchronous local write behind nuPlan's async-compatible signature."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        fp.write(buf)


async def _read_local_binary(path) -> bytes:
    with path.open("rb") as fp:
        return fp.read()


io_utils._save_buffer_async = _save_local_buffer
io_utils.read_binary_async = _read_local_binary


def _save_nuboard_synchronously(self: NuBoardFile, filename) -> None:
    """Avoid aiofiles executor deadlocks observed in this server environment."""

    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("wb") as fp:
        pickle.dump(self.serialize(), fp, protocol=pickle.HIGHEST_PROTOCOL)


NuBoardFile.save_nuboard_file = _save_nuboard_synchronously


def _integrate_metric_files_synchronously(self: MetricFileCallback) -> None:
    metrics = defaultdict(list)
    for directory in self._scenario_metric_paths:
        if not directory.exists():
            continue
        for metric_file in directory.iterdir():
            if not metric_file.name.endswith(".pickle.temp"):
                continue
            with metric_file.open("rb") as fp:
                frames = pickle.load(fp)
            for frame in frames:
                metrics[frame["metric_statistics_name"]].append(pd.DataFrame(frame))
            if self._delete_scenario_metric_files:
                metric_file.unlink(missing_ok=True)
    for name, frames in metrics.items():
        pd.concat(frames, ignore_index=True).to_parquet(self._metric_file_output_path / f"{name}.parquet")


MetricFileCallback.on_run_simulation_end = _integrate_metric_files_synchronously

from sledge.script.run_simulation import main  # noqa: E402


if __name__ == "__main__":
    main()
