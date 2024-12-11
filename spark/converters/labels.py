def default_to_yolo_format(
    image_w: int, image_h: int, r_min: int, c_min: int, r_max: int, c_max: int
) -> tuple[float, float, float, float]:
    x_center = (c_min + c_max) / (2 * image_w)
    y_center = (r_min + r_max) / (2 * image_h)

    width = (c_max - c_min) / image_w
    height = (r_max - r_min) / image_h

    return x_center, y_center, width, height


def yolo_to_default_format(
    image_w: int, image_h: int, x: float, y: float, w: int, h: int
) -> tuple[int, int, int, int]:
    c_min, r_min = (x - w / 2) * image_w, (y - h / 2) * image_h
    c_max, r_max = (x + w / 2) * image_w, (y + h / 2) * image_h

    return int(c_min), int(r_min), int(c_max), int(r_max)
