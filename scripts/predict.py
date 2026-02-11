# Third-party
import torch
from torch.amp import autocast

# Local
from cassava_model_comparison import config as cfg
from cassava_model_comparison.datasets import build_transforms
from cassava_model_comparison.engine import load_best_model
from cassava_model_comparison.io import make_batch_image_from_url

def main():
    runs_dir = cfg.RUNS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet18_model = load_best_model(
        model_name="resnet18",
        ckpt_path=runs_dir / "resnet18/best.pt",
        num_classes=cfg.NUM_CLASSES,
        device=device
    )
    resnet18_model.eval()

    url = "http://www.iita.org/wp-content/uploads/2017/09/1024_CBSD-cassava-root-1024x683.jpg"

    _, _, test_tf = build_transforms()
    img = make_batch_image_from_url(
        url=url, transform=test_tf
    )
    img = img.to(device)

    with torch.no_grad():
        logits = resnet18_model(img)
        pred = logits.argmax(dim=1)
        pred_idx = pred.item()

    print(f"pred_idx={pred_idx} | class={cfg.CLASS_NAMES[pred_idx]}")


if __name__ == "__main__":
    main()