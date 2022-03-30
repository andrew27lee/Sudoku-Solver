import cv2

def my_resize(image, width, height, inter=cv2.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        if height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            if w < h:
                r = width / float(w)
                dim = (width, int(h * r))
            else:
                r = height / float(h)
                dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
