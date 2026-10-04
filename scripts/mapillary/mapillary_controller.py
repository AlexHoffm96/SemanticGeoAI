from scripts.mapillary.mapillary_fetch_metadata import fetch_images
from scripts.mapillary.mapillary_load_image import download_images
from scripts.utils.config_loader import MAX_TOTAL

def run():
    print("MAPILLARY CONTROLLER START")

    print("\nFetch metadata from Mapillary API")
    fetch_images()

    print("\nDownload images from metadata.csv")
    download_images(max_images=MAX_TOTAL)

    print("\nMAPILLARY CONTROLLER FINISHED")

if __name__ == "__main__":
    run()