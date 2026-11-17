import pytest
import torch
import torch.nn as nn
from PIL import Image

from plant_disease.gradcam import GradCAM, overlay
from plant_disease.model import build_model, freeze_backbone, load_checkpoint, save_checkpoint, target_layer, unfreeze_top
from plant_disease.predictor import Predictor, pretty_label

CLASSES = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model("resnet18", len(CLASSES), pretrained=False).eval()


def test_output_shape(model):
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, len(CLASSES))


def test_freeze_then_unfreeze():
    m = build_model("resnet18", 3, pretrained=False)
    freeze_backbone(m)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("fc.") for n in trainable)

    unfreeze_top(m, "resnet18")
    trainable = {n.split(".")[0] for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"fc", "layer3", "layer4"}


def test_gradcam_heatmap(model):
    cam = GradCAM(model, target_layer(model, "resnet18"))
    heatmap, logits = cam(torch.randn(1, 3, 224, 224))
    cam.remove()
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert logits.shape == (1, len(CLASSES))

    img = Image.new("RGB", (256, 256), "green")
    assert overlay(img, heatmap).size == img.size


def test_checkpoint_round_trip(tmp_path, model):
    path = tmp_path / "m.pth"
    save_checkpoint(path, model, "resnet18", CLASSES, 224)
    loaded, ckpt = load_checkpoint(path)
    x = torch.randn(1, 3, 224, 224)
    assert torch.allclose(model(x), loaded(x), atol=1e-5)
    assert ckpt["classes"] == CLASSES


def test_loads_legacy_checkpoint(tmp_path):
    # v1 checkpoints: plain resnet18 with a single Linear head, no 'arch' key
    from torchvision.models import resnet18

    old = resnet18(weights=None)
    old.fc = nn.Linear(old.fc.in_features, len(CLASSES))
    path = tmp_path / "old.pth"
    torch.save({"model_state_dict": old.state_dict(), "classes": CLASSES}, path)

    loaded, _ = load_checkpoint(path)
    old.eval()
    x = torch.randn(1, 3, 224, 224)
    assert torch.allclose(old(x), loaded(x), atol=1e-5)


def test_predictor(tmp_path, model):
    path = tmp_path / "m.pth"
    save_checkpoint(path, model, "resnet18", CLASSES, 224)
    pred = Predictor(path, device="cpu").predict(Image.new("RGB", (300, 200), "green"), explain=True)
    assert pred.label in {pretty_label(c) for c in CLASSES}
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-5
    assert pred.heatmap.shape == (224, 224)


def test_pretty_label():
    assert pretty_label("Tomato___Early_blight") == "Early blight"
    assert pretty_label("Tomato___healthy") == "Healthy"
