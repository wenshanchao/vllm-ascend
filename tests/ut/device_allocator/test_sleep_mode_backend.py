#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

import vllm_ascend
from vllm_ascend.device_allocator import camem
from vllm_ascend.device_allocator import sleep_mode_backend as backend_module
from vllm_ascend.device_allocator.sleep_mode_backend import (
    ASCEND_LEGACY_SLEEP_MODE_BACKEND,
    ASCEND_SLEEP_MODE_BACKEND,
    DEVICE_MEMORY_PHASE,
    AscendSleepBackend,
    SleepPhase,
    SuspendRequest,
)
from vllm_ascend.distributed.kv_transfer.utils import mooncake_transfer_engine


class PhaseRecorder:
    """Builds phases that log every call to one shared list."""

    def __init__(self):
        self.events: list[tuple[str, str, object]] = []
        self.failing: set[tuple[str, str]] = set()

    def phase(self, name: str) -> SleepPhase:
        def step(kind: str):
            def run(argument):
                self.events.append((name, kind, argument))
                if (name, kind) in self.failing:
                    raise RuntimeError(f"{name} {kind} failed")

            return run

        return SleepPhase(name=name, prepare=step("prepare"), suspend=step("suspend"), resume=step("resume"))

    def calls(self) -> list[tuple[str, str]]:
        return [(name, kind) for name, kind, _ in self.events]

    def resume_tags(self) -> list[object]:
        return [argument for _, kind, argument in self.events if kind == "resume"]


@pytest.fixture
def recorder(monkeypatch):
    recorder = PhaseRecorder()
    monkeypatch.setattr(backend_module, "DEVICE_MEMORY_PHASE", recorder.phase("device_memory"))
    return recorder


@pytest.fixture
def backend(recorder):
    backend = AscendSleepBackend()
    backend.add_phase(recorder.phase("graphs"))
    backend.add_phase(recorder.phase("model_state"))
    return backend


def test_plugin_entry_point_registers_the_backend_and_tolerates_reentry():
    vllm_ascend.register_sleep_mode_backend()
    vllm_ascend.register_sleep_mode_backend()

    assert SleepModeBackendFactory.get_backend_class(ASCEND_SLEEP_MODE_BACKEND) is AscendSleepBackend
    assert (
        SleepModeBackendFactory.get_backend_class(ASCEND_LEGACY_SLEEP_MODE_BACKEND)
        is backend_module.LegacyAscendSleepBackend
    )


def test_factory_builds_the_backend_on_npu(monkeypatch):
    vllm_ascend.register_sleep_mode_backend()
    monkeypatch.setattr(backend_module, "current_platform", SimpleNamespace(device_type="npu"))

    backend = SleepModeBackendFactory.create_backend(SimpleNamespace(sleep_mode_backend=ASCEND_SLEEP_MODE_BACKEND))

    assert isinstance(backend, AscendSleepBackend)
    assert backend.state() == "RUNNING"
    assert backend.phases == (DEVICE_MEMORY_PHASE,)


def test_factory_rejects_the_backend_off_npu(monkeypatch):
    vllm_ascend.register_sleep_mode_backend()
    monkeypatch.setattr(backend_module, "current_platform", SimpleNamespace(device_type="cuda"))

    with pytest.raises(ValueError, match="not supported on this platform"):
        SleepModeBackendFactory.create_backend(SimpleNamespace(sleep_mode_backend=ASCEND_SLEEP_MODE_BACKEND))


def test_unsupported_capabilities_are_reported_not_faked():
    backend = AscendSleepBackend()

    with pytest.raises(RuntimeError, match="does not support selective discard"):
        backend.discard(("kv_cache",))
    assert not AscendSleepBackend.preserves_communicators()
    assert not AscendSleepBackend.preserves_graphs_with_communicators()
    assert not AscendSleepBackend.preserves_compiled_artifacts()
    assert not AscendSleepBackend.supports_durable_storage()


def test_suspend_prepares_every_phase_first_and_resume_runs_backwards(backend, recorder):
    backend.suspend(2)

    assert recorder.calls() == [
        ("graphs", "prepare"),
        ("model_state", "prepare"),
        ("device_memory", "prepare"),
        ("graphs", "suspend"),
        ("model_state", "suspend"),
        ("device_memory", "suspend"),
    ]
    assert {argument for _, _, argument in recorder.events} == {SuspendRequest(level=2)}
    assert backend.state() == "SUSPENDED"

    recorder.events.clear()
    backend.resume()

    assert recorder.calls() == [("device_memory", "resume"), ("model_state", "resume"), ("graphs", "resume")]
    assert recorder.resume_tags() == [None, None, None]
    assert backend.state() == "RUNNING"


def test_failed_prepare_releases_nothing(backend, recorder):
    recorder.failing.add(("model_state", "prepare"))

    with pytest.raises(RuntimeError, match="model_state prepare failed"):
        backend.suspend(2)

    assert recorder.calls() == [("graphs", "prepare"), ("model_state", "prepare")]
    assert backend.state() == "RUNNING"


def test_failed_suspend_rolls_back_the_entered_phases_in_reverse(backend, recorder):
    recorder.failing.add(("model_state", "suspend"))

    with pytest.raises(RuntimeError, match="model_state suspend failed"):
        backend.suspend()

    # The failing phase may be half suspended, so it is resumed too; device
    # memory was never reached and is left alone.
    assert recorder.calls()[3:] == [
        ("graphs", "suspend"),
        ("model_state", "suspend"),
        ("model_state", "resume"),
        ("graphs", "resume"),
    ]
    assert recorder.resume_tags() == [None, None]
    assert backend.state() == "RUNNING"

    recorder.failing.clear()
    recorder.events.clear()
    backend.suspend()
    assert backend.state() == "SUSPENDED"


def test_rollback_keeps_going_when_a_phase_cannot_be_resumed(backend, recorder):
    recorder.failing.update({("device_memory", "suspend"), ("model_state", "resume")})

    with pytest.raises(RuntimeError, match="device_memory suspend failed"):
        backend.suspend()

    assert recorder.calls()[-3:] == [("device_memory", "resume"), ("model_state", "resume"), ("graphs", "resume")]


def test_rollback_after_a_staged_wake_leaves_sleeping_tags_asleep(backend, recorder):
    backend.suspend(2)
    backend.resume(["weights"])
    recorder.events.clear()
    recorder.failing.add(("device_memory", "suspend"))

    with pytest.raises(RuntimeError, match="device_memory suspend failed"):
        backend.suspend(2)

    assert recorder.resume_tags() == [["weights"], ["weights"], ["weights"]]
    assert backend.state() == "SUSPENDED"


def test_repeated_suspend_is_ignored_while_the_weights_are_asleep(backend, recorder):
    backend.suspend()
    recorder.events.clear()

    backend.suspend(2)
    assert recorder.events == []

    # Waking only the KV cache changes nothing here: the weights still sleep.
    backend.resume(["kv_cache"])
    recorder.events.clear()
    backend.suspend()

    assert recorder.events == []
    assert backend.state() == "SUSPENDED"


def test_suspend_runs_again_once_the_weights_are_awake(backend, recorder):
    backend.suspend(2)
    backend.resume(["weights"])
    assert backend.state() == "SUSPENDED"
    recorder.events.clear()

    backend.suspend(2)

    assert ("device_memory", "suspend") in recorder.calls()
    backend.resume()
    assert backend.state() == "RUNNING"


def test_repeated_resume_still_reaches_every_phase(backend, recorder):
    backend.suspend()
    backend.resume()
    recorder.events.clear()

    backend.resume(["kv_cache"])

    assert recorder.calls() == [("device_memory", "resume"), ("model_state", "resume"), ("graphs", "resume")]
    assert recorder.resume_tags() == [["kv_cache"]] * 3
    assert backend.state() == "RUNNING"


def test_failed_resume_stays_resuming_and_can_be_retried(backend, recorder):
    backend.suspend()
    recorder.failing.add(("model_state", "resume"))

    with pytest.raises(RuntimeError, match="model_state resume failed"):
        backend.resume()
    assert backend.state() == "RESUMING"

    recorder.failing.clear()
    backend.resume()
    assert backend.state() == "RUNNING"


def test_close_is_idempotent_and_ends_the_backend(backend):
    backend.close()
    backend.close()

    assert backend.phases == (backend_module.DEVICE_MEMORY_PHASE,)
    for call in (backend.suspend, backend.resume, lambda: backend.add_phase(SleepPhase(name="late"))):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_add_phase_rejects_duplicates_and_a_suspended_backend(backend):
    with pytest.raises(ValueError, match="'graphs' is already registered"):
        backend.add_phase(SleepPhase(name="graphs"))
    with pytest.raises(ValueError, match="'device_memory' is already registered"):
        backend.add_phase(SleepPhase(name="device_memory"))

    backend.suspend()
    with pytest.raises(RuntimeError, match="while the backend is suspended"):
        backend.add_phase(SleepPhase(name="late"))


@pytest.fixture
def device(monkeypatch):
    """Stand-ins for the device calls the built-in memory phase makes."""
    device = MagicMock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=device.synchronize), raising=False)
    monkeypatch.setattr(camem.CaMemAllocator, "get_instance", lambda: device.allocator)
    monkeypatch.setattr(mooncake_transfer_engine, "global_te", device.transfer_engine)
    return device


@pytest.mark.parametrize("backend_class", [AscendSleepBackend, backend_module.LegacyAscendSleepBackend])
@pytest.mark.parametrize(("level", "offload_tags"), [(1, ("weights",)), (2, ())])
def test_device_memory_drains_the_device_then_unregisters_before_unmapping(device, backend_class, level, offload_tags):
    backend = backend_class()

    backend.suspend(level)
    assert backend.state() == "SUSPENDED"

    assert [name for name, _, _ in device.mock_calls] == [
        "synchronize",
        "transfer_engine.unregister_buffer",
        "allocator.sleep",
    ]
    device.allocator.sleep.assert_called_once_with(offload_tags=offload_tags)


@pytest.mark.parametrize(
    ("tags", "reregisters"), [(None, True), (["kv_cache"], True), (["weights"], False), ([], False)]
)
@pytest.mark.parametrize("backend_class", [AscendSleepBackend, backend_module.LegacyAscendSleepBackend])
def test_device_memory_reregisters_only_when_the_kv_pool_is_back(device, backend_class, tags, reregisters):
    backend = backend_class()

    backend.resume(tags)
    assert backend.state() == "RUNNING"

    device.allocator.wake_up.assert_called_once_with(tags=tags)
    assert device.transfer_engine.reregister_buffer.called is reregisters
    if reregisters:
        assert [name for name, _, _ in device.mock_calls] == ["allocator.wake_up", "transfer_engine.reregister_buffer"]
