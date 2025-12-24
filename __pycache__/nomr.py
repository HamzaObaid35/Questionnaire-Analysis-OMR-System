import cv2
import os
import pandas as pd
import numpy as np

def preprocess_image(image):
    """Convert image to grayscale and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

def find_largest_contour(thresh):
    """Find the largest contour in the thresholded image."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def warp_image(image, contour):
    """Warp the image based on the largest contour."""
    x, y, w, h = cv2.boundingRect(contour)
    src_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    output_size = (w, h)
    dst_corners = np.array([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(image, M, output_size)
    return warped

def split_into_boxes(image, num_rows, num_cols):
    """Split the image into boxes based on the number of rows and columns."""
    height, width = image.shape
    box_height, box_width = height // num_rows, width // num_cols
    boxes = []
    for i in range(num_rows):
        for j in range(num_cols):
            top, bottom = i * box_height, (i + 1) * box_height
            left, right = j * box_width, (j + 1) * box_width
            box = image[top:bottom, left:right]
            boxes.append(box)
    return boxes

def evaluate_answers(boxes, num_rows, num_cols):
    """Evaluate the answers based on the marked boxes."""
    counts = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            box = boxes[i * num_cols + j]
            counts[i, j] = np.count_nonzero(box)
    
    marked_answers = []
    for i in range(num_rows):
        min_count = np.min(counts[i])
        marked = []
        for j in range(num_cols):
            if counts[i, j] > min_count * 1.8:
                marked.append(j + 1)  # Add 1 to convert index to answer number (1-5)
        if len(marked) == 1:
            marked_answers.append(marked[0])  # Single answer
        else:
            marked_answers.append('N')  # No answer or multiple answers
    return marked_answers

def process_image(image_file, num_rows, num_cols):
    """Process a single image to evaluate answers."""
    image = cv2.imread(os.path.join("images", image_file))
    if image is None:
        print(f"Error: Unable to read image {image_file}")
        return None

    thresh = preprocess_image(image)
    largest_contour = find_largest_contour(thresh)
    if largest_contour is None:
        print(f"Error: No contours found in image {image_file}")
        return None

    warped = warp_image(image, largest_contour)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    boxes = split_into_boxes(warped_thresh, num_rows, num_cols)
    marked_answers = evaluate_answers(boxes, num_rows, num_cols)
    return marked_answers

def main():
    questions = 5
    choices = 5

    results = []
    for image_file in os.listdir("images"):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            marked_answers = process_image(image_file, questions, choices)
            if marked_answers:
                # Create a table of questions and answers for this image
                for q_num, answer in enumerate(marked_answers, start=1):
                    results.append({
                        'Image': image_file,
                        'Question': q_num,
                        'Answer': answer
                    })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    
    # Display the table
    print(df)

    # Save results to a CSV file
    df.to_csv('question_answers.csv', index=False)
    print("Results saved to question_answers.csv")

if __name__ == "__main__":
    main()