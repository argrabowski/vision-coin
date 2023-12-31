# Coin Detection and Recognition

This project utilizes OpenCV to detect and recognize coins in an input image. The script employs the SIFT feature detector, FLANN matcher, and Hough Circle transform to achieve accurate coin detection and recognition.

https://github.com/argrabowski/vision-coin/assets/64287065/31dee19e-b15a-445f-bbc8-d8b497c516b9

## Dependencies

- Python 3.10
- OpenCV (`pip install opencv-python`)

## Usage

1. Ensure you have Python and the required dependencies installed.
2. Place the input image (`coins1.jpg`) and reference coin images (`penny.jpg`, `nickel.jpg`, `dime.jpg`, `quarter.jpg`) in the same directory as the script.
3. Adjust script parameters if necessary (e.g., file paths, Hough Circle transform parameters, FLANN matcher parameters).
4. Run the script:

   ```bash
   python vision-coin.py
   ```

5. The result will be saved as `result1.jpg` in the same directory.

## Parameters

- `minDist`: Minimum distance between detected circles in the Hough Circle transform.
- `param1` and `param2`: Parameters for the Hough Circle transform.
- FLANN matcher parameters can be adjusted for better matching results.
