import cv2


class ImageProcessor:
    def __init__(
        self,
        image_path: str,
        resize=[800, 600],
    ) -> None:
        self.image_path = image_path
        self.image = cv2.imread(f"{image_path}")

        if resize:
            self.image = cv2.resize(self.image, (resize[0], resize[1]))

    def _find_longest_list(self, lists: list[list[float]]) -> tuple[list[float], int]:
        longest_list = max(lists, key=len, default=None)
        if longest_list:
            length_difference = len(longest_list) - len(min(lists, key=len))
            return longest_list, length_difference
        else:
            return None, 0

    def find_contour_segments(self):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary track
        _, binary_track = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary track
        contours, _ = cv2.findContours(
            binary_track, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the two largest contours
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        segments = [[], []]
        # Loop through the largest contours to extract the line segments
        for j, contour in enumerate(largest_contours):
            epsilon = 0.004 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Loop through the edges of the polygon to extract the line segments
            for i in range(len(approx)):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % len(approx)][0]
                segments[j].append([p1[0], p1[1], p2[0], p2[1]])

        longest_segment, length_difference = self._find_longest_list(segments)

        if longest_segment:
            longest_segment[-1 - length_difference][-2:] = longest_segment[-1][-2:]
            del longest_segment[-length_difference:]

        return segments
