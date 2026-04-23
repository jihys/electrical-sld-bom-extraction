"""Test: auto-select best reference image via LLM."""
from pathlib import Path
import cv2

from src.config import Settings
from src.agents.llm_caller import create_llm_client
from src.cad.panel_name_extractor import select_best_reference_image

def main():
    settings = Settings()
    client = create_llm_client(
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    deploy = settings.azure_openai_deployment

    # Load a sample page
    sample = cv2.imread("outputs/pages/page5.png")
    print(f"Sample page: {sample.shape[1]}x{sample.shape[0]}")

    candidates = [
        Path("data/panel_name_box_example1.png"),
        Path("data/panel_name_box_example2.png"),
    ]
    print(f"Candidates: {[c.name for c in candidates]}")

    best = select_best_reference_image(
        sample, candidates, client, deploy,
        category="panel name box",
    )
    print(f"\n>>> Best match: {best.name if best else 'None'}")

if __name__ == "__main__":
    main()
