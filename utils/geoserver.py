import requests
import uuid


# Defined functions
def retrieve_tiff(
        bbox: str,
        width: int,
        height: int,
        layer: str
    ):
    """
    Retrieve a GeoTIFF image from a GeoServer instance.
    """

    # Set the GeoServer URL
    url = 'https://landscapes.wearepal.ai/geoserver/wcs?'

    # Set the request parameters
    params = {
        'SERVICE': 'WCS',
        'VERSION': '1.0.0',
        'REQUEST': 'GetCoverage',
        'COVERAGE': layer,
        'BBOX': bbox,
        'WIDTH': width,
        'HEIGHT': height,
        'FORMAT': 'image/geotiff',
        'CRS': 'EPSG:3857',
        'RESPONSE_CRS': 'EPSG:3857'
    }

    # Send the request
    response = requests.get(url, params=params)

    # Generate a unique name for the image
    id = str(uuid.uuid4())

    # Save the image with a unique name
    img_path = f'tmp/{id}.tif'

    # Save the image to a temporary file
    with open(img_path, 'wb') as f:
        f.write(response.content)

    # Return the path to the image
    return img_path
