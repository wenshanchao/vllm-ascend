# Sleep Mode Backend

## 1. Background

vLLM selects the mechanism behind `sleep()` and `wake_up()` by name through `SleepModeBackend` (RFC #34303). Its default backend, `cumem`, finds its allocator with `get_mem_allocator_instance()`, which only resolves on CUDA and XPU and raises on NPU.

Before this design, vLLM Ascend ran its own sleep sequence inline in `NPUWorker`. That sequence releases more than allocator memory:

| Resource | Released on sleep | Restored on wake-up |
| --- | --- | --- |
| HCCL process groups | destroyed (RL extra cleanup only) | recreated |
| ACL graphs | workspaces and captured graphs dropped (RL extra cleanup only) | captured again once the KV cache is back |
| Model state kept by a level-2 sleep | parameters named by `--sleep-preserve-parameter-names` and all buffers copied to host | copied back in place |
| Mooncake buffer registration | unregistered | registered again once the KV cache is back |
| `CaMemAllocator` pools | weights offloaded (level 1) or dropped (level 2), everything else dropped | mapped again, offloaded weights copied back |

`AscendSleepBackend` moves this sequence behind vLLM's backend interface. The user guide is [Sleep Mode](../../user_guide/feature_guide/sleep_mode.md).

## 2. Design

### 2.1 Registration and Selection

`vllm_ascend:register_sleep_mode_backend` is a `vllm.general_plugins` entry point. It registers two names with `SleepModeBackendFactory`:

| `--sleep-mode-backend` | Class | Use |
| --- | --- | --- |
| `ascend` | `AscendSleepBackend` | Default on NPU. |
| `ascend_legacy` | `LegacyAscendSleepBackend` | Fallback, see [Migration](#4-migration). |

Registration lives in the package's entry-point metadata, so an editable install picks it up only after `pip install -e .` is run again.

When sleep mode is enabled and the option still holds upstream's default `cumem`, `NPUPlatform.check_and_update_config` replaces it with `ascend`. An explicit choice is left as it is.

`NPUWorker` creates the backend through the factory on the first `sleep()` or `wake_up()`, the same way upstream's GPU worker does.

### 2.2 Phases

`AscendSleepBackend` runs a sleep as an ordered list of `SleepPhase` objects. A phase has a name and three optional callables:

| Callable | When it runs | What it may do |
| --- | --- | --- |
| `prepare(request)` | For every phase, before any phase suspends | Reject the request, or copy out what `suspend` is about to destroy. It releases nothing. |
| `suspend(request)` | In registration order | Release the resource. |
| `resume(tags)` | In reverse registration order | Restore the resource. It must cope with a resource that is not suspended. |

`request` is a `SuspendRequest`. It carries the sleep level today and is the place for further request data.

The phases, in suspend order:

| Phase | Registered by | `prepare` | `suspend` | `resume` |
| --- | --- | --- | --- | --- |
| `hccl_and_acl_graph` | `NPUWorker`, only with `rl_config.sleep_mode_extra_cleanup` | none | `SleepWakeupManager.sleep()` | `SleepWakeupManager.wakeup(tags)` |
| `model_state` | `NPUWorker` | Level 2: copy retained parameters and buffers to host | none | Copy them back when `weights` wakes |
| `device_memory` | built in, always last | none | Device barrier, Mooncake unregister, `CaMemAllocator.sleep` | `CaMemAllocator.wake_up`, Mooncake register when `kv_cache` wakes |

The resulting order of device operations is the one the worker used before:

```text
sleep(level)                               wake_up(tags)
  model_state.prepare   copy to host         device_memory.resume       map, re-register
  hccl_and_acl_graph    tear down            model_state.resume         copy back
  device_memory         barrier,             hccl_and_acl_graph.resume  recreate groups,
                        unregister, unmap                               capture graphs
```

Model state is restored before ACL graphs are captured again, because capture runs the model.

### 2.3 Errors and Repeated Calls

| Situation | Behavior |
| --- | --- |
| `prepare` raises, for example a retention pattern that matches no parameter | The error propagates. Nothing was released, so nothing is rolled back. |
| `suspend` of a phase raises | Every phase entered so far, the failing one included, is resumed in reverse order, and the original error propagates. The state is left as it was before the call. |
| A phase fails during that rollback | The failure is logged, the remaining phases are still resumed, and the caller gets the original error. |
| `suspend` while the weights are asleep | Ignored with a warning. This is the rule vLLM's executor applies before it calls the workers. |
| `suspend` after a staged wake-up of `weights` only | Runs. A rollback of this call wakes only `weights`, so `kv_cache` stays asleep. |
| `resume` raises | The error propagates and the state stays `RESUMING`. `resume` can be called again. |
| `resume` when nothing is recorded asleep | Still delegated to every phase, as in upstream's `cumem` backend. A wake-up is never skipped on the strength of the backend's own bookkeeping. |
| `close()` | Drops the registered phases. Later `suspend` and `resume` calls raise `RuntimeError`. Calling it again does nothing. |

`state()` reports `SUSPENDED` while any of the tags `weights` and `kv_cache` is asleep, `RESUMING` during a wake-up, and `RUNNING` otherwise.

### 2.4 Capabilities

| Capability | Value | Reason |
| --- | --- | --- |
| `is_supported()` | `True` on the NPU platform only | The factory rejects the backend elsewhere. |
| `preserves_communicators()` | `False` | HCCL groups are destroyed under RL extra cleanup, and a class-level flag cannot see that option. |
| `preserves_graphs_with_communicators()`, `preserves_compiled_artifacts()`, `supports_durable_storage()` | `False` | Upstream defaults. ACL graphs are captured again after a wake-up. |
| `discard(tags)` | raises `RuntimeError` | Selective discard (`release_kv_cache_memory`) has no worker entry point on Ascend. |

## 3. Adding a Resource

A resource that needs worker or model-runner state is registered where that state lives. The factory builds the backend from the model config alone, so the worker adds its phases right after creating it:

```python
backend.add_phase(
    SleepPhase(
        name="my_resource",
        prepare=lambda request: check_or_back_up(request.level),
        suspend=lambda request: release(),
        resume=lambda tags: restore(tags),
    )
)
```

Phases suspend in the order they were added, ahead of `device_memory`. `add_phase` rejects a duplicate name and a backend that is suspended or closed.

## 4. Migration

What moved:

| Before | Now |
| --- | --- |
| `NPUWorker.sleep` called `SleepWakeupManager`, the Mooncake engine and `CaMemAllocator` in sequence | `NPUWorker.sleep` calls `backend.suspend(level)`; the sequence is the phase list |
| A failure half-way left the worker partly asleep | The phases already entered are resumed |
| Only vLLM's executor rejected a second sleep | The backend rejects it as well |
| The barrier ahead of the unmap was `NPUWorker.synchronize_device()` | The `device_memory` phase issues the same `torch.npu.synchronize()` |

`SleepWakeupManager` and `CaMemAllocator` are unchanged.

The previous sequence stays available as `--sleep-mode-backend ascend_legacy`. With it, `NPUWorker` saves model state and runs `SleepWakeupManager` itself, then calls a backend that only handles device memory. There are no phases, no rollback and no check for a repeated sleep. The worker treats a third-party backend the same way. `ascend_legacy` is removed once `ascend` has been validated on the supported models and hardware.

vLLM versions after v0.33.0 change `SleepModeBackend.suspend(level)` to `suspend(discardable)`, where `discardable` holds the storage addresses the caller rewrites after waking. When vLLM Ascend moves to such a version, `AscendSleepBackend.suspend` and `NPUWorker.sleep` change with it and `SuspendRequest` carries the new field. Phases do not change.

## 5. Tests

| File | Covers |
| --- | --- |
| `tests/ut/device_allocator/test_sleep_mode_backend.py` | Registration, platform support, phase order, rollback, repeated suspend and resume, staged wake-up, `close`, unsupported capabilities, the `device_memory` phase |
| `tests/ut/worker/test_sleep_mode_backend_wiring.py` | Phases the worker registers, end-to-end order through `NPUWorker`, rollback through the worker, `ascend_legacy`, third-party backends, shutdown |
| `tests/ut/worker/test_sleep_parameter_retention.py`, `tests/ut/worker/test_worker_v1.py`, `tests/ut/worker/test_kv_cache_allocation_scope.py` | Existing sleep tests, now running through the backend |
| `tests/ut/test_platform.py` | Backend selection |
