import json
import logging
import os

import click
import magic

from fer import face_check

#logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("fer-capture-log")

@click.command()
@click.option("--model", default = "model.h5", help = "Path to model binary.")
@click.option("--image", help = "Path to JPEG image.")
@click.option("--out", default = "raw", help = " 'raw': print dictionary to stdout, 'json': json to file ")
def cli(model, image, out):
    """
    This function acts as the interface between the command-line and the face_check function.

    Parameters:
        image (str): Valid file path to a JPEG image.
        model (str): Valid file path to an h5 model.
        out (str): 'raw' or 'json'

    Returns:
        data (dict): Dictionary containing the base64+ut8 encoding of the detected faces along with the predicted emotion.
        OR
        data (json): JSON containing the base64+ut8 encoding of the detected faces along with the predicted emotion
    """
    mime = magic.Magic(mime=True)
    if not mime.from_file(model) == "application/x-hdf":
        log.error("{} is not a valid application/x-hdf!".format(model))
        return 1
    if not mime.from_file(image) == "image/jpeg":
        log.error("{} is not a valid image/jpeg!".format(image))
        return 1
    try:
        data = face_check(image, model)
    except Exception as e:
        log.error(e)
        return 1
    if out == "json":
        with open(str(image.split("/")[-1].replace(".", "-") + ".json"), "w") as f:
            f.write(json.dumps(data))
            return 0
    print(data)
    return data
