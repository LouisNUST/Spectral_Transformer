from skimage import transform
import re

PREFIX = 'rot'
REGEX = re.compile(r"^" + PREFIX + "_(?P<angle>-?[0-9]+)")

class Rotate:
    def __init__(self, angle):
        self.angle = angle
        self.code = PREFIX + str(angle)

    def process(self, img):
        img=transform.rotate(img,90,resize=True)
        return img

    @staticmethod
    def match_code(code):
        match = REGEX.match(code)
        if match:
            d = match.groupdict()
            return Rotate(int(d['angle']))
