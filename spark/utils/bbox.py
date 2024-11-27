def convert_bbox_to_yolo(
    image_w: int, image_h: int, r_min: int, c_min: int, r_max: int, c_max: int
) -> tuple[float, float, float, float]:
    x_center = (c_min + c_max) / (2 * image_w)
    y_center = (r_min + r_max) / (2 * image_h)

    # Compute width and height
    width = (c_max - c_min) / image_w
    height = (r_max - r_min) / image_h

    return x_center, y_center, width, height
