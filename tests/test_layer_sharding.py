import unittest

from kaggle_h3.layer_sharding import (
    find_dispatchable_layers,
    find_dispatchable_text_encoder_layers,
    inspect_dispatched_map,
    is_comfy_quantized_model,
    parameter_bytes,
    plan_layer_device_map,
    plan_text_encoder_device_map,
)


class _FakeDtype:
    itemsize = 1


class _FakeParameter:
    dtype = _FakeDtype()

    def __init__(self, count: int):
        self.count = count

    def numel(self):
        return self.count

    def element_size(self):
        return 1


class QuantizedTensor(_FakeParameter):
    """Name-only stand-in for Comfy's optional quantized tensor type."""


class _FakeModule:
    def __init__(self, children=None, parameter_bytes=0):
        self._children = list(children or [])
        self._parameter = _FakeParameter(parameter_bytes) if parameter_bytes else None

    def named_children(self):
        return ((str(index), child) for index, child in enumerate(self._children))

    def named_modules(self):
        yield "", self
        for name, child in self.named_children():
            yield name, child
            for nested_name, nested in child.named_modules():
                if nested_name:
                    yield f"{name}.{nested_name}", nested

    def parameters(self, recurse=True):
        if self._parameter is not None:
            yield self._parameter
        if recurse:
            for child in self._children:
                yield from child.parameters(recurse=True)

    def buffers(self, recurse=True):
        return iter(())


class LayerShardingTests(unittest.TestCase):
    def test_discovers_top_level_h3_blocks(self):
        token_refiner = _FakeModule([_FakeModule(parameter_bytes=1)] * 2)
        blocks = _FakeModule([_FakeModule(parameter_bytes=100)] * 4)
        model = _FakeModule([token_refiner, blocks])
        # Rename the children to match realistic H3 names for the structural
        # discovery test without making the planner depend on torch classes.
        model.named_children = lambda: iter(
            [("token_refiner", token_refiner), ("blocks", blocks)]
        )
        model.named_modules = lambda: iter(
            [
                ("", model),
                ("token_refiner", token_refiner),
                ("token_refiner.blocks", token_refiner),
                ("blocks", blocks),
                ("blocks.0", blocks._children[0]),
                ("blocks.1", blocks._children[1]),
                ("blocks.2", blocks._children[2]),
                ("blocks.3", blocks._children[3]),
            ]
        )
        group = find_dispatchable_layers(model)
        self.assertEqual(group.container_path, "blocks")
        self.assertEqual(len(group.layers), 4)
        self.assertEqual(group.layers[0].path, "blocks.0")

    def test_plans_contiguous_layers_on_every_gpu(self):
        blocks = _FakeModule([_FakeModule(parameter_bytes=100)] * 4)
        model = _FakeModule([blocks])
        model.named_children = lambda: iter([("blocks", blocks)])
        model.named_modules = lambda: iter(
            [("", model), ("blocks", blocks)]
            + [(f"blocks.{i}", child) for i, child in enumerate(blocks._children)]
        )
        plan = plan_layer_device_map(
            model,
            device_ids=(0, 1),
            gpu_budgets={0: 250, 1: 250},
            cpu_budget_bytes=0,
            allow_disk=False,
        )
        layer_devices = [item["device"] for item in plan["layers"]]
        self.assertEqual(layer_devices, ["cuda:0", "cuda:0", "cuda:1", "cuda:1"])
        self.assertEqual(plan["uses_all_requested_gpus"], True)
        self.assertEqual(plan["strategy"], "contiguous_intact_transformer_blocks")

    def test_preferred_h3_t4_layout_uses_23_27_block_split(self):
        projections = _FakeModule(parameter_bytes=2)
        token_refiner = _FakeModule(parameter_bytes=3)
        blocks = _FakeModule([_FakeModule(parameter_bytes=1) for _ in range(50)])
        final_layer = _FakeModule(parameter_bytes=2)
        model = _FakeModule([projections, token_refiner, blocks, final_layer])
        model.named_children = lambda: iter(
            [
                ("proj_in", projections),
                ("token_refiner", token_refiner),
                ("transformer_blocks", blocks),
                ("norm_out", final_layer),
            ]
        )
        model.named_modules = lambda: iter(
            [("", model), ("proj_in", projections), ("token_refiner", token_refiner),
             ("transformer_blocks", blocks),
             ("norm_out", final_layer)]
            + [(f"transformer_blocks.{i}", child) for i, child in enumerate(blocks._children)]
        )
        plan = plan_layer_device_map(
            model,
            device_ids=(0, 1),
            gpu_budgets={0: 30, 1: 30},
            cpu_budget_bytes=0,
            allow_disk=False,
            preferred_layout="h3_two_t4_preferred",
        )
        layer_devices = [item["device"] for item in plan["layers"]]
        self.assertEqual(layer_devices[:23], ["cuda:0"] * 23)
        self.assertEqual(layer_devices[23:], ["cuda:1"] * 27)
        self.assertEqual(plan["device_map"]["proj_in"], 0)
        self.assertEqual(plan["device_map"]["token_refiner"], 0)
        self.assertEqual(plan["device_map"]["norm_out"], 0)
        self.assertEqual(plan["strategy"], "h3_two_t4_preferred_contiguous_blocks")
        self.assertIn("after transformer_blocks.22", plan["preferred_layout"]["activation_boundary"])

    def test_quantized_model_detection_is_structural_and_dependency_free(self):
        model = _FakeModule([_FakeModule(parameter_bytes=1)])
        model._children[0]._parameter = QuantizedTensor(1)
        self.assertTrue(is_comfy_quantized_model(model))

    def test_quantized_budget_uses_physical_payload_not_logical_dtype(self):
        value = QuantizedTensor(100)
        value._qdata = _FakeParameter(10)
        value._params = type(
            "Params",
            (),
            {
                "__dataclass_fields__": {"scale": object()},
                "scale": _FakeParameter(1),
            },
        )()
        module = _FakeModule()
        module._parameter = value
        self.assertEqual(parameter_bytes(module), 11)

    def test_execution_map_proves_both_gpus_even_when_overflow_weights_stay_cpu(self):
        blocks = _FakeModule([_FakeModule(parameter_bytes=100)] * 4)
        model = _FakeModule([blocks])
        model.named_children = lambda: iter([("blocks", blocks)])
        model.named_modules = lambda: iter(
            [("", model), ("blocks", blocks)]
            + [(f"blocks.{i}", child) for i, child in enumerate(blocks._children)]
        )
        plan = plan_layer_device_map(
            model,
            device_ids=(0, 1),
            gpu_budgets={0: 250, 1: 250},
            cpu_budget_bytes=0,
            allow_disk=False,
        )
        execution_map = {
            item["path"]: item["device"] for item in plan["layers"]
        }
        observed = inspect_dispatched_map(
            model, plan, execution_device_map=execution_map
        )
        self.assertEqual(observed["observed_execution_gpu_ids"], [0, 1])
        self.assertTrue(observed["observed_all_requested_gpus"])

    def test_text_encoder_prefers_language_layers_over_vision_blocks(self):
        vision_blocks = _FakeModule([_FakeModule(parameter_bytes=90)] * 6)
        language_layers = _FakeModule([_FakeModule(parameter_bytes=100)] * 4)
        vision = _FakeModule([vision_blocks])
        language = _FakeModule([language_layers])
        model = _FakeModule([vision, language])
        model.named_children = lambda: iter(
            [("vision_model", vision), ("language_model", language)]
        )
        model.named_modules = lambda: iter(
            [
                ("", model),
                ("vision_model", vision),
                ("vision_model.blocks", vision_blocks),
                ("language_model", language),
                ("language_model.layers", language_layers),
            ]
            + [
                (f"language_model.layers.{i}", child)
                for i, child in enumerate(language_layers._children)
            ]
            + [
                (f"vision_model.blocks.{i}", child)
                for i, child in enumerate(vision_blocks._children)
            ]
        )
        group = find_dispatchable_text_encoder_layers(model)
        self.assertEqual(group.container_path, "language_model.layers")
        plan = plan_text_encoder_device_map(
            model,
            device_ids=(0, 1),
            gpu_budgets={0: 250, 1: 250},
            cpu_budget_bytes=0,
            allow_disk=False,
        )
        self.assertEqual(
            [item["device"] for item in plan["layers"]],
            ["cuda:0", "cuda:0", "cuda:1", "cuda:1"],
        )
        self.assertEqual(plan["strategy"], "contiguous_qwen_language_layers")
        self.assertTrue(plan["uses_all_requested_gpus"])


if __name__ == "__main__":
    unittest.main()
