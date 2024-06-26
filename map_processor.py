import cv2
import os


class MapProcessor:
    def __init__(
        self,
        map_path: str,
        width: int = 800,
        height: int = 600,
        processed_name_prefix: str = "processed",
        show_map: bool = True,
    ) -> None:
        self.map_path = map_path
        self.track = cv2.imread(f"{map_path}", cv2.IMREAD_GRAYSCALE)
        self.width = width
        self.height = height
        self.processed_name_prefix = processed_name_prefix
        self.show_map = show_map

    def process_map(self):
        resized_track = cv2.resize(self.track, (800, 600))

        _, binary_track = cv2.threshold(resized_track, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary_track, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_image = cv2.cvtColor(resized_track, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        output_image_path = os.path.join(
            os.getcwd(), f"{self.processed_name_prefix}_{self.map_path}"
        )

        cv2.imwrite(output_image_path, contour_image)

        if self.show_map:
            self._show_map(contour_image)

    def _show_map(self, map):
        cv2.imshow("Track with Contours", map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
