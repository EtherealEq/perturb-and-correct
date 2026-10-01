"""The construction itself: does correction preserve calibration behaviour?

These are the behavioural tests the paper's headline claim rests on, and they run
against both implementations. A P&C member should still differ from the base model in
its *hidden* parameters, and agree with the base model on calibration data far better
than the same perturbation without the repair. The uncorrected ablation is the control
throughout.
"""
import numpy as np
import pytest

from pnc.spec import CorrectionSpec

from .conftest import (
    asnp,
    calibration,
    cuda_usable,
    is_torch,
    make_conv_case,
    make_ffn,
    make_stack,
)

SPEC = CorrectionSpec(ridge=1e-3, bootstrap_fraction=0.0, seed=0)


# --------------------------------------------------------------------------
# dense
# --------------------------------------------------------------------------
def test_correction_is_exact_when_the_redundancy_is_purely_affine(backend):
    """The sharpest statement of the mechanism. With a linear activation the
    perturbed layer spans the same affine space as the original, so the following
    layer can absorb the perturbation completely and the member reproduces the base
    model exactly -- affine redundancy in its pure form. The identical perturbation
    without the repair moves predictions by O(1)."""
    stack = make_stack(backend, activation=lambda x: x)
    inputs = calibration()
    base = asnp(stack.predict(inputs)["mean"])
    kwargs = dict(perturb_layers=[0], rank=8, n_members=4, scale=1.0, seed=0,
                  spec=CorrectionSpec(ridge=1e-9, bootstrap_fraction=0.0, seed=0))
    corrected = backend.build_dense_pnc(stack, inputs, correct=True, **kwargs)
    uncorrected = backend.build_dense_pnc(stack, inputs, correct=False, **kwargs)

    assert np.abs(asnp(corrected.predict(inputs)["mean"]) - base).mean() < 1e-10
    assert np.abs(asnp(uncorrected.predict(inputs)["mean"]) - base).mean() > 1e-2


def test_correction_solves_its_interface_exactly(backend, dense_stack):
    """Whatever the end-to-end effect, the repair must be the exact ridge solution at
    the interface it is fit on."""
    inputs = calibration()
    ensemble = backend.build_dense_pnc(
        dense_stack, inputs, perturb_layers=[0], rank=8, n_members=3,
        scale=1.0, spec=SPEC, seed=0,
    )
    weights = [asnp(w) for w in dense_stack.weights]
    biases = [asnp(b) for b in dense_stack.biases]
    perturbed = np.maximum(inputs @ (weights[0] + asnp(ensemble.deltas[0])[0]) + biases[0], 0.0)
    base_hidden = np.maximum(inputs @ weights[0] + biases[0], 0.0)
    target = base_hidden @ weights[1] + biases[1]
    expected = asnp(
        backend.ridge_solve(
            backend.augment(perturbed), target, SPEC.ridge,
            prior=np.concatenate([biases[1][None, :], weights[1]], axis=0),
        )
    )
    weight, bias = ensemble.corrections[1]
    got = np.concatenate([asnp(bias)[0][None, :], asnp(weight)[0]], axis=0)
    assert np.allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("perturb_layers", [[0], [0, 2]])
def test_correction_reduces_the_id_visible_effect(backend, dense_stack, perturb_layers):
    """Figure 2: at matched perturbation, the repair suppresses the ID output change.
    On a randomly initialised trunk there is little affine redundancy to exploit, so
    the margin here is modest; the trained networks in the paper do far better."""
    inputs = calibration()
    base = asnp(dense_stack.predict(inputs)["mean"])
    kwargs = dict(perturb_layers=perturb_layers, rank=8, n_members=6, scale=1.0,
                  spec=SPEC, seed=0)
    corrected = backend.build_dense_pnc(dense_stack, inputs, correct=True, **kwargs)
    uncorrected = backend.build_dense_pnc(dense_stack, inputs, correct=False, **kwargs)
    corrected_error = np.abs(asnp(corrected.predict(inputs)["mean"]) - base).mean()
    uncorrected_error = np.abs(asnp(uncorrected.predict(inputs)["mean"]) - base).mean()
    assert corrected_error < 0.6 * uncorrected_error


def test_correction_matters_more_as_the_perturbation_grows(backend, dense_stack):
    """Appendix E.1: small perturbations need little correction; large ones expose a
    regime where the repair is what retains the base model's ID behaviour."""
    inputs = calibration()
    base = asnp(dense_stack.predict(inputs)["mean"])

    def ratio(scale):
        kwargs = dict(perturb_layers=[0], rank=8, n_members=6, scale=scale, spec=SPEC, seed=0)
        corrected = backend.build_dense_pnc(dense_stack, inputs, correct=True, **kwargs)
        uncorrected = backend.build_dense_pnc(dense_stack, inputs, correct=False, **kwargs)
        return float(
            np.abs(asnp(uncorrected.predict(inputs)["mean"]) - base).mean()
            / np.abs(asnp(corrected.predict(inputs)["mean"]) - base).mean()
        )

    assert ratio(2.0) > ratio(0.5) > 1.0


def test_hidden_perturbation_stays_large_after_correction(backend, dense_stack):
    """Figure 2(B): the repair suppresses the *output* effect, not the perturbation."""
    ensemble = backend.build_dense_pnc(
        dense_stack, calibration(), perturb_layers=[0], rank=8, n_members=4,
        scale=1.0, spec=SPEC, seed=0,
    )
    norms = np.linalg.norm(asnp(ensemble.deltas[0]).reshape(4, -1), axis=1)
    assert np.all(norms > 0.5)


def test_members_disagree_more_off_the_calibration_distribution(backend, dense_stack):
    """The mechanism claim: agreement on calibration data, disagreement away from it."""
    inputs = calibration(n=1024)
    ensemble = backend.build_dense_pnc(
        dense_stack, inputs, perturb_layers=[0, 2], rank=8, n_members=12,
        scale=1.0, spec=SPEC, seed=0,
    )
    shifted = calibration(n=1024, seed=99) * 6.0

    def spread(x):
        return asnp(ensemble.predict(x)["mean"]).std(axis=0).mean()

    assert spread(shifted) > spread(inputs)


def test_members_are_distinct(backend, dense_stack):
    inputs = calibration()
    ensemble = backend.build_dense_pnc(
        dense_stack, inputs, perturb_layers=[0], rank=8, n_members=5,
        scale=1.0, spec=SPEC, seed=0,
    )
    predictions = asnp(ensemble.predict(inputs)["mean"])
    assert not np.allclose(predictions[0], predictions[1])


def test_bootstrapped_members_use_different_correction_subsets(backend, dense_stack):
    inputs = calibration(n=1024)
    common = dict(perturb_layers=[0], rank=8, n_members=10, scale=1.0, seed=0)
    shared = backend.build_dense_pnc(
        dense_stack, inputs, spec=CorrectionSpec(1e-3, 0.0, seed=0), **common
    )
    bootstrapped = backend.build_dense_pnc(
        dense_stack, inputs, spec=CorrectionSpec(1e-3, 0.1, seed=0), **common
    )
    # Identical perturbations, different correction geometries -> different members.
    assert np.allclose(asnp(shared.deltas[0]), asnp(bootstrapped.deltas[0]))
    assert not np.allclose(
        asnp(shared.predict(inputs)["mean"]), asnp(bootstrapped.predict(inputs)["mean"])
    )


def test_prediction_batching_does_not_change_results(backend, dense_stack):
    inputs = calibration(n=500)
    ensemble = backend.build_dense_pnc(
        dense_stack, inputs, perturb_layers=[0], rank=8, n_members=4,
        scale=1.0, spec=SPEC, seed=0,
    )
    assert np.allclose(
        asnp(ensemble.predict(inputs, batch_size=64)["mean"]),
        asnp(ensemble.predict(inputs, batch_size=None)["mean"]),
        atol=1e-12,
    )


def test_output_heads_are_never_corrected(backend, dense_stack):
    with pytest.raises(ValueError, match="no following trunk layer"):
        backend.build_dense_pnc(
            dense_stack, calibration(), perturb_layers=[3], rank=4, n_members=2,
            scale=1.0, spec=SPEC, seed=0,
        )


def test_perturb_layers_must_be_ordered(backend, dense_stack):
    with pytest.raises(ValueError, match="network order"):
        backend.build_dense_pnc(
            dense_stack, calibration(), perturb_layers=[2, 0], rank=4, n_members=2,
            scale=1.0, spec=SPEC, seed=0,
        )


@pytest.mark.parametrize("perturb_layers", [[0, 1], [0, 0], [1, 2]])
def test_overlapping_perturb_layers_are_rejected(backend, dense_stack, perturb_layers):
    """Regression: targets one apart let the second perturb a layer the first has
    already corrected, so the fitted correction no longer matches its own design.
    That used to build silently and give a member ~1e0 away from the base model on a
    purely affine trunk, where the repair should be exact to ~1e-13.
    """
    with pytest.raises(ValueError, match="overlap"):
        backend.build_dense_pnc(
            dense_stack, calibration(), perturb_layers=perturb_layers, rank=4,
            n_members=2, scale=1.0, spec=SPEC, seed=0,
        )


def test_well_separated_perturb_layers_are_still_accepted(backend, dense_stack):
    ensemble = backend.build_dense_pnc(
        dense_stack, calibration(), perturb_layers=[0, 2], rank=4, n_members=2,
        scale=1.0, spec=SPEC, seed=0,
    )
    assert set(ensemble.deltas) == {0, 2}
    assert set(ensemble.corrections) == {1, 3}


def test_torch_stack_places_inputs_on_its_own_device():
    """Regression: only `build_dense_pnc` moved inputs to the stack's device, so a
    GPU-resident stack raised `Expected all tensors to be on the same device` for the
    host arrays callers naturally pass to `predict`.
    """
    import pnc.torch as pt

    stack = make_stack(pt)
    assert stack.as_input(calibration(n=8)).device == stack.device


@pytest.mark.skipif(not cuda_usable(), reason="no usable CUDA device")
def test_torch_accepts_host_inputs_on_a_cuda_stack():
    import pnc.torch as pt

    stack = make_stack(pt).to("cuda")
    inputs = calibration(n=128)  # host NumPy, as a caller would pass
    assert asnp(stack.predict(inputs)["mean"]).shape == (128, 3)
    ensemble = pt.build_dense_pnc(
        stack, inputs, perturb_layers=[0], rank=8, n_members=3,
        scale=1.0, spec=SPEC, seed=0,
    )
    assert asnp(ensemble.predict(inputs)["mean"]).shape == (3, 128, 3)


# --------------------------------------------------------------------------
# conv
# --------------------------------------------------------------------------
def test_conv_kernel_flattening_roundtrips(backend):
    shape = (7, 5, 3, 3) if is_torch(backend) else (3, 3, 5, 7)
    kernel = np.random.RandomState(0).standard_normal(shape)
    flat = backend.flatten_kernel_to_patch_order(kernel)
    assert tuple(flat.shape) == (45, 7)
    assert np.allclose(asnp(backend.unflatten_kernel_from_patch_order(flat, shape)), kernel)


def test_conv_correction_preserves_the_block_output(backend):
    case = make_conv_case(backend)
    kwargs = dict(rank=6, n_members=3, scale=1.0, spec=SPEC, seed=0)
    corrected = backend.build_conv_pnc(
        case["block"], case["chunks"], case["targets"], correct=True, **kwargs
    )
    uncorrected = backend.build_conv_pnc(
        case["block"], case["chunks"], case["targets"], correct=False, **kwargs
    )
    assert case["block_error"](corrected) < 0.7 * case["block_error"](uncorrected)
    # The perturbation survives the repair.
    assert not np.allclose(asnp(corrected[0].w1), asnp(case["w1_native"]))


def test_conv_members_learn_a_bias_the_base_convolution_lacks(backend):
    """Appendix B.3: the augmented constant column absorbs perturbation-induced mean
    shifts, so a Conv2 bias is fit even though the base convolution has none."""
    case = make_conv_case(backend, channels=3, size=5, n_images=16, chunk=16)
    members = backend.build_conv_pnc(
        case["block"], case["chunks"], case["targets"],
        rank=4, n_members=2, scale=0.5, spec=SPEC, seed=0,
    )
    assert case["block"].n_features == case["n_features"]
    assert np.any(np.abs(asnp(members[0].b2)) > 0)


# --------------------------------------------------------------------------
# transformer FFN
# --------------------------------------------------------------------------
def test_ffn_correction_preserves_the_block_output(backend):
    block = make_ffn(backend)
    hidden = np.random.RandomState(2).standard_normal((600, 16))
    targets = asnp(block.forward(hidden))

    kwargs = dict(rank=8, n_members=4, relative_magnitude=0.5, spec=SPEC, seed=0)
    corrected, basis, scale = backend.build_ffn_pnc(block, hidden, targets, correct=True, **kwargs)
    uncorrected, _, _ = backend.build_ffn_pnc(block, hidden, targets, correct=False, **kwargs)

    def error(members):
        values = []
        for member in members:
            perturbed = asnp(block.w1) + asnp(basis.delta(member.coefficient, scale))
            output = asnp(block.forward(hidden, w1=perturbed, w2=member.w2, b2=member.b2))
            values.append(np.abs(output - targets).mean())
        return float(np.mean(values))

    assert error(corrected) < 0.6 * error(uncorrected)


def test_ffn_relative_magnitude_hits_the_requested_operating_point(backend):
    block = make_ffn(backend, d_model=32, d_hidden=64)
    hidden = np.random.RandomState(3).standard_normal((400, 32))
    members, _, _ = backend.build_ffn_pnc(
        block, hidden, asnp(block.forward(hidden)), rank=20, n_members=41,
        relative_magnitude=2.0, spec=SPEC, seed=0,
    )
    achieved = [m.relative_magnitude for m in members]
    assert np.median(achieved) == pytest.approx(2.0, rel=1e-9)


def test_ffn_members_store_only_coefficients_and_the_second_map(backend):
    block = make_ffn(backend)
    hidden = np.random.RandomState(4).standard_normal((300, 16))
    members, basis, _ = backend.build_ffn_pnc(
        block, hidden, asnp(block.forward(hidden)), rank=5, n_members=3,
        relative_magnitude=1.0, spec=SPEC, seed=0,
    )
    assert tuple(members[0].coefficient.shape) == (5,)
    assert tuple(basis.directions.shape) == (5, 16 * 32)


def test_ffn_chunking_does_not_change_the_result(backend):
    block = make_ffn(backend)
    hidden = np.random.RandomState(5).standard_normal((512, 16))
    targets = asnp(block.forward(hidden))
    kwargs = dict(rank=6, n_members=2, relative_magnitude=0.8, spec=SPEC, seed=0)
    small, _, _ = backend.build_ffn_pnc(block, hidden, targets, chunk_size=37, **kwargs)
    large, _, _ = backend.build_ffn_pnc(block, hidden, targets, chunk_size=4096, **kwargs)
    assert np.allclose(asnp(small[0].w2), asnp(large[0].w2), atol=1e-9)
    assert np.allclose(asnp(small[0].b2), asnp(large[0].b2), atol=1e-9)
