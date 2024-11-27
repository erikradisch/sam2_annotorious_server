import torch
import torchvision
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import cv2
from skimage import measure
import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon
from shapely import to_geojson
import json
# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from pycocotools import mask as mask_utils

hostName = "0.0.0.0"
serverPort = 5374
#Dictionary to store loaded images, so they don't have to be loaded each time
loadedImages = {}
def split_into_tiles(image, tile_size=1024):
    """
    devides an image into tiles.

    :param image: image to be devided
    :param tile_size: size of the tiles
    :return: Array of tiles and their positions
    """
    h, w, _ = image.shape
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, x, y))  # Speichere die Position für späteres Zusammenfügen
    return tiles


def fetch_tile(image_url, x, y, width, height):
    """
    Fetches an image tile from a specified URL and processes it to ensure it is in RGB format.

    :param image_url: The base URL of the image.
    :param x: The x-coordinate of the top-left corner of the tile.
    :param y: The y-coordinate of the top-left corner of the tile.
    :param width: The width of the tile.
    :param height: The height of the tile.
    :return: The processed image tile in RGB format, or None if an error occurs.
    """
    # Construct the URL for the image tile
    tile_url = f"{image_url}/{x},{y},{width},{height}/full/0/default.jpg"
    try:
        # Open the URL and read the contents into a byte array
        response = urlopen(tile_url)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)

        # Decode the byte array to an image (either grayscale or BGR)
        tile = cv2.imdecode(arr, -1)

        # Check if the image is grayscale and convert to RGB
        if len(tile.shape) == 2:  # Grayscale image (2D array)
            print(f"Grayscale image detected at {tile_url}. Converting to RGB.")
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
        else:
            # If the image is already in BGR format, convert to RGB
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        return tile
    except Exception as e:
        # Log an error message if the tile cannot be loaded
        print(f"Error loading tile: {e}")
        return None
def load_full_image_in_tiles(image_url, info):
    """
    Loads an entire image by fetching it in tiles from a given URL.

    :param image_url: Base URL of the image.
    :param info: Dictionary containing image metadata, including width, height, and tile size.
    :return: The full image reconstructed from its tiles.
    """
    width = info['width']
    height = info['height']
    tile_width = info['tiles'][0]['width']
    tile_height = tile_width  # Assuming square tiles

    # Initialize an empty array for the full image
    full_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate over the image area in steps of tile height and width
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            # Calculate the dimensions of the current tile
            w = min(tile_width, width - x)
            h = min(tile_height, height - y)

            # Fetch the tile and verify its presence
            tile = fetch_tile(image_url, x, y, w, h)
            if tile is not None:
                # Place the tile into the correct position in the full image
                full_image[y:y+h, x:x+w] = tile[:h, :w]

    return full_image

def fetch_image(image_url):
    """
    Fetches an image from a given URL and returns it in RGB format.

    :param image_url: URL of the image to fetch.
    :return: The fetched image in RGB format, or None if an error occurs.
    """
    try:
        response = urlopen(image_url)
        content_type = response.info().get_content_type()

        # JSON means it's a IIIF image
        if content_type == "application/json":
            print("json")
            # Read and analyze the JSON
            info_json = json.load(response)
            if "@id" in info_json and "width" in info_json and "height" in info_json:
                image_url = f"{info_json['@id']}/full/full/0/default.jpg"
                print(f"Bild-URL aus info.json extrahiert: {image_url}")
                img = load_full_image_in_tiles(info_json['@id'], info_json)
                print(f"Image type: {type(img)}, Shape: {getattr(img, 'shape', 'N/A')}")
                return img
            else:
                raise ValueError("info.json enthält keine gültigen IIIF-Daten.")
        # Image means it's a direct image file
        elif content_type.startswith("image/"):
            print("found image")
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)  # Load image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            return img
        else:
            raise ValueError(f"Unbekannter Inhaltstyp: {content_type}")
    except Exception as e:
        print(f"Fehler beim Laden der URL: {e}")
        return None

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """
    Converts a binary mask to a COCO polygon representation.

    Args:
        binary_mask: A 2D binary numpy array where '1's represent the object.
        tolerance: Maximum distance from original points of polygon to approximated
                   polygonal chain. If tolerance is 0, the original coordinate array is returned.

    Returns:
        A MultiPolygon object representing the detected polygons in the mask.
    """
    def close_contour(contour):
        """
        Closes a contour by connecting its last point to the first if they are not the same.

        Args:
            contour: A numpy array representing the contour points.

        Returns:
            A closed contour as a numpy array.
        """
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    # Find contours in the binary mask
    contours = measure.find_contours(binary_mask, 0.5)

    segmentations = []
    polygons = []

    # Process each contour
    for contour in contours:
        contour = close_contour(contour)  # Ensure contour is closed
        contour = measure.approximate_polygon(contour, tolerance)  # Approximate the contour

        if len(contour) < 3:  # Ignore contours that are too small
            continue

        # Flip the contour for correct (x, y) order
        contour = np.flip(contour, axis=1)

        # Create a polygon from the contour
        poly = Polygon(contour)
        poly = poly.simplify(0.5, preserve_topology=True)  # Simplify the polygon

        polygons.append(poly)  # Add to list of polygons

        # Convert polygon exterior coordinates to a flat list
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Create a MultiPolygon from all detected polygons
    multipolygon = MultiPolygon(polygons)

    return multipolygon


class Sam2Server(BaseHTTPRequestHandler):
    # choose device for inference
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    # load model
    sam2_checkpoint = "sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    def do_OPTIONS(self):
        """
        Handle OPTIONS requests. We need to return a 200 OK response and
        set the Access-Control-Allow-Methods header to allow the client
        to send GET and POST requests.
        """
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header('Access-Control-Allow-Headers', "Content-Type, Authorization, X-GWT-Module-Base, X-Requested-With")
        self.end_headers()

    def do_POST(self):
        """
        Handle POST requests to the server.

        This will load an image from a given URL, set the image in the predictor,
        get the input points and labels from the request, predict the mask using SAM2,
        convert the mask to a polygon, and return the polygon as a JSON response.

        :return: A JSON response containing the predicted polygon
        """
        content_len = int(self.headers.get('Content-Length'))
        body = self.rfile.read(content_len)
        str_body = body.decode('utf8')
        data = json.loads(str_body)
        print("incomming post: ", data)
        # The "test" is for testing connection. I couldn't make GWT to do just a simple options-request.
        if "test" in data:
            if data["test"]:
                print("responding 200")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header('Access-Control-Allow-Headers', "Content-Type, Authorization, X-GWT-Module-Base, X-Requested-With")
                self.end_headers()
        else:
            #this is the normal request
            img = None
            # look if image was already loaded. if not, load it and put it in the dictionary. This is important to prevent the server to be forced to load an image every time, a new request was made.
            # TO DO: imlement a cleaning of the loaded images, to prevent becomming to big.
            if data["image"] in loadedImages:
                print("image taken from storage")
                img = loadedImages[data['image']]
            else:
                print("image to be loaded")
                img = fetch_image(data['image'])
                loadedImages[data['image']] = img
            print(f"Image type: {type(img)}, Shape: {getattr(img, 'shape', 'N/A')}")
            #start infering
            self.predictor.set_image(img)
            input_point = np.array(data['input_points'])
            input_label = np.array(data['input_labels'])
            self.predictor.set_image(img)
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            m = masks[np.argmax(scores), :, :]  # Choose the model's best mask
            poly = binary_mask_to_polygon(np.array(m)) # converting mask zo polygon for annotorious
            #responding
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header('Access-Control-Allow-Headers', "Content-Type, Authorization, X-GWT-Module-Base, X-Requested-With")
            self.end_headers()
            polySvg = poly.svg()
            root = ET.fromstring(polySvg)
            path_strings = [child.attrib['d'].rstrip('Z').strip() for child in root if child.tag == "path"]

            # Bundle all cleaned 'd'-Attributes and add z at the end
            combined_path = " ".join(path_strings) + " Z"

            # wrap in path tag
            bundled_svg = f'<path d="{combined_path}" />'
            json_string = json.dumps({"mask": bundled_svg})
            self.wfile.write(json_string.encode(encoding='utf_8'))

if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), Sam2Server)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")



