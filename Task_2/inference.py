import argparse
import os
import torch
import cv2
from src.model import HuggingFaceLoFTR
from src.visualization import draw_matches


def run_inference(img0_path: str, img1_path: str, output_path: str, model_path: str = "models/loftr_finetuned"):
    """Runs image matching pipeline on an arbitrary pair of images and saves the visual result"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading the model from: {model_path}")

    if os.path.exists(os.path.join(model_path, "loftr_weights.pth")):
        model = HuggingFaceLoFTR.from_pretrained(model_path, device=device)
    else:
        print("No local weights found, loading the base checkpoint 'outdoor'...")
        model = HuggingFaceLoFTR(pretrained_config="outdoor").to(device)

    model.eval()

    # Read input images in grayscale
    img0_raw = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    if img0_raw is None or img1_raw is None:
        raise FileNotFoundError("Check the paths to the image files.")

    # Resize images to match model standard input dimensions (512x512)
    img0_res = cv2.resize(img0_raw, (512, 512))
    img1_res = cv2.resize(img1_raw, (512, 512))

    # Convert arrays to normalized tensors (1, 1, H, W)
    t0 = torch.from_numpy(img0_res).float()[None, None] / 255.0
    t1 = torch.from_numpy(img1_res).float()[None, None] / 255.0

    t0, t1 = t0.to(device), t1.to(device)

    # Perform inference forward pass
    with torch.no_grad():
        out = model(t0, t1)

    # Extract matching coordinates and confidence scores
    pts0 = out['keypoints0'].cpu().numpy()
    pts1 = out['keypoints1'].cpu().numpy()
    confs = out['confidence'].cpu().numpy()

    print(f"Matches found: {len(pts0)}")

    # Render matches and save result image to disk
    vis_img = draw_matches(img0_res, img1_res, pts0, pts1, confs, max_draw=150, conf_thresh=0.35)
    cv2.imwrite(output_path, vis_img)
    print(f"The imaging results are saved in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Image Matching Inference")
    parser.add_argument("--img0", required=True, help="Path to the first image")
    parser.add_argument("--img1", required=True, help="The path to the second image")
    parser.add_argument("--output", default="matches_result.png", help="Path to save the result")
    parser.add_argument("--model", default="models/loftr_finetuned", help="Path to the folder containing the model weights")

    args = parser.parse_args()
    run_inference(args.img0, args.img1, args.output, args.model)