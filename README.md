## Overview

The goal is to detect and follow the position of a soccer ball in still images captured from a short video. The project implements:

- **Normalized Cross‑Correlation (NCC)** – a template matching technique that slides a target patch (the ball) over each frame and measures similarity to find the best match.
- **Covariance Tracking** – uses covariance matrices of pixel features (e.g. x/y coordinates and RGB values) to model the target and compares this model with windows in the frame.
- **Hybrid Selection** – runs both methods on an image and picks the result that has the more plausible colour distribution (based on near‑black and near‑white pixel proportions).

A simple dataset of frames lives in the `Dataset` folder, and three template images (`target.jpg`, `target2.jpg`, `target3.jpg`) provide different views of the ball.

Dataset Source: [Dataset](https://github.com/AIS-Bonn/TemporalBallDetection)

This code is written for Python and uses the following libraries:

- `numpy`
- `scikit-image`
- `matplotlib`
- `scipy`
- `opencv-python`

