import base64

def save_image(path: str, data: str):
    with open(
        path,
        "wb",
    ) as image_file:
        image_file.write(base64.b64decode(data))